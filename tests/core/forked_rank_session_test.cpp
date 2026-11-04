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

#include "tests/core/forked_rank_session.h"

#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <memory>

namespace xllm::test {
namespace {

using Status = ForkedRankSession::Status;

TEST(ForkedRankSessionTest, ReusesRankStateAcrossCases) {
  auto counter = std::make_shared<int32_t>(0);
  ForkedRankSession session(2,
                            0,
                            {{"first",
                              [counter](int32_t, int32_t) {
                                EXPECT_EQ((*counter)++, 0);
                                return 0;
                              }},
                             {"second", [counter](int32_t, int32_t) {
                                EXPECT_EQ((*counter)++, 1);
                                return 0;
                              }}});
  EXPECT_EQ(session.run("first").status, Status::PASSED);
  EXPECT_EQ(session.run("second").status, Status::PASSED);
  EXPECT_EQ(*counter, 0) << "Scenario state must stay in the child";
  EXPECT_EQ(session.close().status, Status::PASSED);
}

TEST(ForkedRankSessionTest, PropagatesNonfatalChildAssertion) {
  ForkedRankSession session(2, 0, {{"assertion", [](int32_t rank, int32_t) {
                                      EXPECT_NE(rank, 1)
                                          << "sentinel assertion";
                                      return 0;
                                    }}});
  const auto result = session.run("assertion");
  EXPECT_EQ(result.status, Status::FAILED);
  EXPECT_NE(result.message.find("sentinel assertion"), std::string::npos);
  EXPECT_EQ(session.run("assertion").status, Status::SKIPPED);
}

TEST(ForkedRankSessionTest, StopsPeersWhenOneRankCrashes) {
  ForkedRankSession session(2, 0, {{"crash", [](int32_t rank, int32_t) {
                                      if (rank == 0) {
                                        ::raise(SIGKILL);
                                      }
                                      ::pause();
                                      return 0;
                                    }}});
  const auto started = std::chrono::steady_clock::now();
  EXPECT_EQ(session.run("crash").status, Status::FAILED);
  EXPECT_LT(std::chrono::steady_clock::now() - started,
            std::chrono::seconds{3});
  int status = 0;
  EXPECT_EQ(::waitpid(-1, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
}

TEST(AttentionMultiDeviceProcessTest, TimedOutChildIsTerminatedAndReaped) {
  ForkedRankSession session(2,
                            0,
                            {{"timeout",
                              [](int32_t, int32_t) {
                                ::pause();
                                return 0;
                              }}},
                            {},
                            std::chrono::milliseconds{50});
  const auto result = session.run("timeout");
  EXPECT_EQ(result.status, Status::FAILED);
  EXPECT_NE(result.message.find("timed out"), std::string::npos);
  int status = 0;
  errno = 0;
  EXPECT_EQ(::waitpid(-1, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD);
}

TEST(ForkedRankSessionTest, RequiresEveryRankToSkip) {
  ForkedRankSession all(2, 0, {{"skip", [](int32_t, int32_t) { return 77; }}});
  EXPECT_EQ(all.run("skip").status, Status::SKIPPED);
  ForkedRankSession mixed(2, 0, {{"mixed", [](int32_t rank, int32_t) {
                                    return rank == 0 ? 0 : 77;
                                  }}});
  EXPECT_EQ(mixed.run("mixed").status, Status::FAILED);
}

TEST(ForkedRankSessionTest, NewSessionWorksAfterFailure) {
  ForkedRankSession failed(
      2, 0, {{"fail", [](int32_t, int32_t) { return 1; }}});
  ASSERT_EQ(failed.run("fail").status, Status::FAILED);
  ForkedRankSession fresh(2, 0, {{"pass", [](int32_t, int32_t) { return 0; }}});
  EXPECT_EQ(fresh.run("pass").status, Status::PASSED);
  EXPECT_EQ(fresh.close().status, Status::PASSED);
}

TEST(ForkedRankSessionTest, CleanupFailureIsNotReportedAsSkipped) {
  ForkedRankSession session(
      2, 0, {{"skip", [](int32_t, int32_t) { return 77; }}}, [] {
        ::_exit(1);
      });
  EXPECT_EQ(session.run("skip").status, Status::FAILED);
}

TEST(ForkedRankSessionTest, ReapsRanksAfterParentExit) {
  int prior_subreaper = 0;
  if (::prctl(PR_GET_CHILD_SUBREAPER, &prior_subreaper) != 0 ||
      ::prctl(PR_SET_CHILD_SUBREAPER, 1) != 0) {
    const int32_t error = errno;
    if (error == EINVAL || error == ENOSYS || error == EPERM ||
        error == EACCES) {
      GTEST_SKIP() << "Child subreaper unavailable: " << std::strerror(error);
    }
    FAIL() << "Cannot configure child subreaper: " << std::strerror(error);
  }
  int channel[2];
  if (::pipe(channel) != 0) {
    const int error = errno;
    EXPECT_EQ(::prctl(PR_SET_CHILD_SUBREAPER, prior_subreaper), 0);
    FAIL() << "Cannot create PID pipe: " << std::strerror(error);
  }
  const pid_t supervisor = ::fork();
  if (supervisor < 0) {
    const int error = errno;
    ::close(channel[0]);
    ::close(channel[1]);
    EXPECT_EQ(::prctl(PR_SET_CHILD_SUBREAPER, prior_subreaper), 0);
    FAIL() << "Cannot fork supervisor: " << std::strerror(error);
  }
  if (supervisor == 0) {
    ::close(channel[0]);
    ForkedRankSession session(
        1, 0, {{"pid", [fd = channel[1]](int32_t, int32_t) {
                  const pid_t pid = ::getpid();
                  ssize_t written;
                  do {
                    written = ::write(fd, &pid, sizeof(pid));
                  } while (written < 0 && errno == EINTR);
                  if (written != sizeof(pid)) {
                    return 1;
                  }
                  // Stay in the callback so socket closure cannot race the
                  // parent-death signal by making the rank exit on its own.
                  while (true) {
                    ::pause();
                  }
                }}});
    session.run("pid");
    ::_exit(1);
  }
  ::close(channel[1]);
  pollfd readable{channel[0], POLLIN, 0};
  const auto timeout = std::chrono::seconds{10};
  const auto ready_deadline = std::chrono::steady_clock::now() + timeout;
  int ready = 0;
  do {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            ready_deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      ready = 0;
      break;
    }
    ready = ::poll(&readable, 1, static_cast<int>(remaining.count()));
  } while (ready < 0 && errno == EINTR);
  EXPECT_GT(ready, 0) << "Waiting for rank PID: "
                      << (ready == 0 ? "timed out" : std::strerror(errno));
  pid_t rank_pid = -1;
  if (ready > 0) {
    ssize_t received;
    do {
      received = ::read(channel[0], &rank_pid, sizeof(rank_pid));
    } while (received < 0 && errno == EINTR);
    EXPECT_EQ(received, sizeof(rank_pid))
        << "Reading rank PID: "
        << (received < 0 ? std::strerror(errno) : "incomplete PID");
  }
  ::close(channel[0]);
  EXPECT_EQ(::kill(supervisor, SIGKILL), 0) << std::strerror(errno);
  const auto wait_for_process = [](pid_t pid, int* status, int options) {
    pid_t result;
    do {
      result = ::waitpid(pid, status, options);
    } while (result < 0 && errno == EINTR);
    return result;
  };
  int status = 0;
  EXPECT_EQ(wait_for_process(supervisor, &status, 0), supervisor)
      << std::strerror(errno);
  EXPECT_GT(rank_pid, 0);
  if (rank_pid > 0) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    pid_t reaped = 0;
    while (reaped == 0 && std::chrono::steady_clock::now() < deadline) {
      reaped = wait_for_process(rank_pid, &status, WNOHANG);
      ::poll(nullptr, 0, 5);
    }
    if (reaped == 0) {
      ::kill(rank_pid, SIGKILL);
      EXPECT_EQ(wait_for_process(rank_pid, &status, 0), rank_pid)
          << std::strerror(errno);
    }
    EXPECT_EQ(reaped, rank_pid)
        << "Reaping rank after supervisor exit: "
        << (reaped == 0 ? "timed out" : std::strerror(errno));
    if (reaped == rank_pid) {
      EXPECT_TRUE(WIFSIGNALED(status)) << "Wait status: " << status;
      if (WIFSIGNALED(status)) {
        EXPECT_EQ(WTERMSIG(status), SIGKILL);
      }
    }
  }
  EXPECT_EQ(::prctl(PR_SET_CHILD_SUBREAPER, prior_subreaper), 0);
}

}  // namespace
}  // namespace xllm::test
