/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int* token_ids;
  size_t len;
} TokenizerEncodeResult;

static void* (*create_tokenizer)(const char*);
static void (*free_tokenizer)(void*);
static void (*encode)(void*, const char*, size_t, int, TokenizerEncodeResult*);
static void (*free_encoded)(TokenizerEncodeResult*, size_t);
static void (
    *decode)(void*, const uint32_t*, size_t, int, const char**, size_t*);

static const char* const prompts[] = {
    "First body content is measured from the same clock.",
    "请只回答计算结果：1.3＋2.9 等于多少？",
    " spaces\n\tASCII and Unicode π, 初始值, punctuation! "};

typedef struct {
  const char* model;
  void* handle;
  uint64_t fingerprint;
  size_t tokens;
} Probe;

static uint64_t fold(uint64_t h, uint8_t value) {
  return (h ^ value) * UINT64_C(1099511628211);
}

static void* run_worker(void* raw) {
  Probe* probe = (Probe*)raw;
  probe->handle = create_tokenizer(probe->model);
  if (probe->handle == NULL) {
    return (void*)1;
  }
  for (size_t i = 0; i < sizeof(prompts) / sizeof(prompts[0]); ++i) {
    for (int add_special = 0; add_special < 2; ++add_special) {
      TokenizerEncodeResult result = {0};
      encode(
          probe->handle, prompts[i], strlen(prompts[i]), add_special, &result);
      if (result.token_ids == NULL || result.len == 0) {
        return (void*)1;
      }
      probe->tokens += result.len;
      for (size_t token = 0; token < result.len; ++token) {
        uint32_t word = (uint32_t)result.token_ids[token];
        for (unsigned int byte = 0; byte < sizeof(word); ++byte) {
          probe->fingerprint =
              fold(probe->fingerprint, (uint8_t)(word >> (8 * byte)));
        }
      }
      const char* decoded = NULL;
      size_t decoded_len = 0;
      decode(probe->handle,
             (const uint32_t*)result.token_ids,
             result.len,
             0,
             &decoded,
             &decoded_len);
      if (decoded == NULL || decoded_len == 0) {
        free_encoded(&result, 1);
        return (void*)1;
      }
      for (size_t byte = 0; byte < decoded_len; ++byte) {
        probe->fingerprint = fold(probe->fingerprint, (uint8_t)decoded[byte]);
      }
      free_encoded(&result, 1);
    }
  }
  return NULL;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return EXIT_FAILURE;
  }
  create_tokenizer =
      (void* (*)(const char*))dlsym(RTLD_DEFAULT, "tokenizers_new_from_path");
  free_tokenizer = (void (*)(void*))dlsym(RTLD_DEFAULT, "tokenizers_free");
  encode =
      (void (*)(void*, const char*, size_t, int, TokenizerEncodeResult*))dlsym(
          RTLD_DEFAULT, "tokenizers_encode");
  free_encoded = (void (*)(TokenizerEncodeResult*, size_t))dlsym(
      RTLD_DEFAULT, "tokenizers_free_encode_results");
  decode =
      (void (*)(void*, const uint32_t*, size_t, int, const char**, size_t*))
          dlsym(RTLD_DEFAULT, "tokenizers_decode");
  if (!create_tokenizer || !free_tokenizer || !encode || !free_encoded ||
      !decode) {
    return EXIT_FAILURE;
  }
  enum { WORKERS = 4 };
  pthread_t workers[WORKERS];
  Probe probes[WORKERS] = {0};
  for (size_t i = 0; i < WORKERS; ++i) {
    probes[i].model = argv[1];
    probes[i].fingerprint = UINT64_C(14695981039346656037);
    if (pthread_create(&workers[i], NULL, run_worker, &probes[i]) != 0) {
      return EXIT_FAILURE;
    }
  }
  for (size_t i = 0; i < WORKERS; ++i) {
    void* status = NULL;
    if (pthread_join(workers[i], &status) != 0 || status != NULL ||
        probes[i].tokens != probes[0].tokens ||
        probes[i].fingerprint != probes[0].fingerprint) {
      return EXIT_FAILURE;
    }
    for (size_t prior = 0; prior < i; ++prior) {
      if (probes[i].handle == probes[prior].handle) {
        return EXIT_FAILURE;
      }
    }
  }
  printf("workers=%d tokens_each=%zu fingerprint=%016llx\n",
         WORKERS,
         probes[0].tokens,
         (unsigned long long)probes[0].fingerprint);
  for (size_t i = 0; i < WORKERS; ++i) {
    free_tokenizer(probes[i].handle);
  }
  return EXIT_SUCCESS;
}
