/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace xllm::test {

// The registry is constructed before fork. Only case indices cross the socket;
// callbacks and their state always remain in the process that owns them.
class ForkedRankSession final {
 public:
  struct Case {
    std::string name;
    std::function<int32_t(int32_t, int32_t)> run;
  };

  enum class Status { PASSED, FAILED, SKIPPED };

  struct Result {
    Status status = Status::PASSED;
    std::string message;
  };

  explicit ForkedRankSession(
      int32_t ranks,
      int32_t port,
      std::vector<Case> cases,
      std::function<void()> cleanup = {},
      std::chrono::milliseconds timeout = std::chrono::seconds{120})
      : ranks_(ranks),
        port_(port),
        cases_(std::move(cases)),
        cleanup_(std::move(cleanup)),
        timeout_(timeout) {}

  ForkedRankSession(const ForkedRankSession&) = delete;
  ForkedRankSession& operator=(const ForkedRankSession&) = delete;

  ~ForkedRankSession() { close(); }

  Result run(const std::string& name) {
    if (unavailable_) {
      return {Status::SKIPPED, "Earlier rank failure or insufficient devices"};
    }
    const auto found =
        std::find_if(cases_.begin(), cases_.end(), [&](const Case& item) {
          return item.name == name;
        });
    if (found == cases_.end()) {
      return fail("Unregistered rank scenario: " + name);
    }
    if (workers_.empty() && !start()) {
      return fail("Cannot start rank processes");
    }
    const int32_t command = static_cast<int32_t>(found - cases_.begin());
    for (const auto& worker : workers_) {
      if (::send(worker.fd, &command, sizeof(command), MSG_NOSIGNAL) !=
          sizeof(command)) {
        return fail("Cannot dispatch scenario " + name);
      }
    }
    std::vector<pollfd> pending;
    pending.reserve(workers_.size());
    for (const auto& worker : workers_) {
      pending.push_back({worker.fd, POLLIN, 0});
    }
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    int32_t remaining = ranks_;
    int32_t skipped = 0;
    while (remaining > 0) {
      const int64_t left =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now())
              .count();
      if (left <= 0) {
        return fail("Rank scenario timed out: " + name);
      }
      const int32_t ready =
          ::poll(pending.data(), pending.size(), static_cast<int>(left));
      if (ready < 0 && errno == EINTR) {
        continue;
      }
      if (ready < 0) {
        return fail("Cannot poll rank results: " + name);
      }
      for (size_t rank = 0; rank < pending.size(); ++rank) {
        if (pending[rank].fd < 0 || pending[rank].revents == 0) {
          continue;
        }
        Reply reply{};
        const ssize_t received =
            ::recv(pending[rank].fd, &reply, sizeof(reply), MSG_DONTWAIT);
        if (received < 0 && (errno == EINTR || errno == EAGAIN)) {
          continue;
        }
        if (received != sizeof(reply)) {
          return fail(name + ": rank " + std::to_string(rank) +
                      " exited without a result");
        }
        if (reply.code != 0 && reply.code != kSkip) {
          return fail(name + ": rank " + std::to_string(rank) + ": " +
                      reply.message.data());
        }
        skipped += reply.code == kSkip;
        pending[rank].fd = -1;
        --remaining;
      }
    }
    if (skipped == ranks_) {
      const Result closed = close();
      unavailable_ = true;
      if (closed.status != Status::PASSED) {
        return closed;
      }
      return {Status::SKIPPED, "Insufficient visible devices on every rank"};
    }
    if (skipped != 0) {
      return fail("Only some ranks skipped scenario " + name);
    }
    return {};
  }

  // Graceful cleanup is bounded even when a backend destructor blocks.
  Result close() {
    const int32_t quit = -1;
    for (const auto& worker : workers_) {
      ::send(worker.fd, &quit, sizeof(quit), MSG_NOSIGNAL);
    }
    return reap(/*terminate=*/false);
  }

 private:
  static constexpr int32_t kSkip = 77;

  struct Worker {
    pid_t pid = -1;
    int fd = -1;
  };

  struct Reply {
    int32_t code = 0;
    std::array<char, 4096> message{};
  };

  bool start() {
    if (ranks_ <= 0) {
      return false;
    }
    std::vector<std::array<int, 2>> channels(ranks_, {-1, -1});
    for (auto& channel : channels) {
      if (::socketpair(
              AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, channel.data()) != 0) {
        for (const auto& opened : channels) {
          for (int fd : opened) {
            if (fd >= 0) {
              ::close(fd);
            }
          }
        }
        return false;
      }
    }
    workers_.reserve(ranks_);
    const pid_t parent = ::getpid();
    bool started = true;
    for (int32_t rank = 0; rank < ranks_; ++rank) {
      const pid_t pid = ::fork();
      if (pid == 0) {
        if (::prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || ::getppid() != parent) {
          ::_exit(1);
        }
        for (int32_t index = 0; index < ranks_; ++index) {
          ::close(channels[index][0]);
          if (index != rank) {
            ::close(channels[index][1]);
          }
        }
        child_loop(rank, channels[rank][1]);
      }
      if (pid < 0) {
        started = false;
        break;
      }
      workers_.push_back({pid, channels[rank][0]});
    }
    for (int32_t index = 0; index < ranks_; ++index) {
      ::close(channels[index][1]);
      if (index >= static_cast<int32_t>(workers_.size())) {
        ::close(channels[index][0]);
      }
    }
    return started;
  }

  [[noreturn]] void child_loop(int32_t rank, int fd) {
    while (true) {
      int32_t command = -1;
      ssize_t received;
      do {
        received = ::recv(fd, &command, sizeof(command), 0);
      } while (received < 0 && errno == EINTR);
      if (received != sizeof(command) || command < 0) {
        if (cleanup_) {
          cleanup_();
        }
        ::_exit(0);
      }
      if (command >= static_cast<int32_t>(cases_.size())) {
        ::_exit(1);
      }
      Reply reply{};
      ::testing::TestPartResultArray assertions;
      {
        ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS,
            &assertions);
        try {
          reply.code = cases_[command].run(rank, port_);
        } catch (const std::exception& error) {
          reply.code = 1;
          std::snprintf(
              reply.message.data(), reply.message.size(), "%s", error.what());
        }
      }
      for (int index = 0; index < assertions.size(); ++index) {
        const auto& assertion = assertions.GetTestPartResult(index);
        if (assertion.failed()) {
          reply.code = 1;
          std::snprintf(reply.message.data(),
                        reply.message.size(),
                        "%s:%d: %s",
                        assertion.file_name(),
                        assertion.line_number(),
                        assertion.message());
          std::fprintf(stderr,
                       "[%s rank %d] %s\n",
                       cases_[command].name.c_str(),
                       rank,
                       reply.message.data());
        } else if (assertion.skipped() && reply.code == 0) {
          reply.code = kSkip;
        }
      }
      if (reply.code != 0 && reply.message[0] == '\0') {
        std::snprintf(reply.message.data(),
                      reply.message.size(),
                      "scenario returned %d",
                      reply.code);
      }
      std::fflush(nullptr);
      if (::send(fd, &reply, sizeof(reply), MSG_NOSIGNAL) != sizeof(reply)) {
        ::_exit(1);
      }
    }
  }

  Result fail(std::string message) {
    unavailable_ = true;
    reap(/*terminate=*/true);
    return {Status::FAILED, std::move(message)};
  }

  Result reap(bool terminate) {
    for (const auto& worker : workers_) {
      if (terminate) {
        ::kill(worker.pid, SIGTERM);
      }
      ::close(worker.fd);
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{2};
    bool clean = true;
    for (const auto& worker : workers_) {
      int status = 0;
      pid_t result;
      while (true) {
        result = ::waitpid(worker.pid, &status, WNOHANG);
        if (result == worker.pid || (result < 0 && errno != EINTR)) {
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          ::kill(worker.pid, SIGKILL);
          do {
            result = ::waitpid(worker.pid, &status, 0);
          } while (result < 0 && errno == EINTR);
          clean = false;
          break;
        }
        ::poll(nullptr, 0, 5);
      }
      clean = clean && result == worker.pid && WIFEXITED(status) &&
              WEXITSTATUS(status) == 0;
    }
    workers_.clear();
    return clean ? Result{}
                 : Result{Status::FAILED, "Rank cleanup failed or timed out"};
  }

  int32_t ranks_;
  int32_t port_;
  std::vector<Case> cases_;
  std::function<void()> cleanup_;
  std::chrono::milliseconds timeout_;
  std::vector<Worker> workers_;
  bool unavailable_ = false;
};

inline void expect_rank_result(const ForkedRankSession::Result& result) {
  if (result.status == ForkedRankSession::Status::SKIPPED) {
    GTEST_SKIP() << result.message;
  }
  ASSERT_EQ(result.status, ForkedRankSession::Status::PASSED) << result.message;
}

}  // namespace xllm::test
