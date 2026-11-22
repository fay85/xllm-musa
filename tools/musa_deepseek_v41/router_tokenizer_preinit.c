/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_PREINITIALIZED_TOKENIZERS 32

typedef void* (*TokenizersCreate)(const char*);
typedef void (*TokenizersDestroy)(void*);

static TokenizersCreate create_tokenizer;
static TokenizersDestroy destroy_tokenizer;
static pthread_once_t api_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static char expected_path[PATH_MAX];
static struct stat expected_file;
static void* preinitialized[MAX_PREINITIALIZED_TOKENIZERS];
static size_t available;
static bool enabled;

static void resolve_api(void) {
  create_tokenizer =
      (TokenizersCreate)dlsym(RTLD_NEXT, "tokenizers_new_from_path");
  destroy_tokenizer = (TokenizersDestroy)dlsym(RTLD_NEXT, "tokenizers_free");
}

static bool same_file(const struct stat* lhs, const struct stat* rhs) {
  return lhs->st_dev == rhs->st_dev && lhs->st_ino == rhs->st_ino &&
         lhs->st_size == rhs->st_size &&
         lhs->st_mtim.tv_sec == rhs->st_mtim.tv_sec &&
         lhs->st_mtim.tv_nsec == rhs->st_mtim.tv_nsec &&
         lhs->st_ctim.tv_sec == rhs->st_ctim.tv_sec &&
         lhs->st_ctim.tv_nsec == rhs->st_ctim.tv_nsec;
}

static bool verify_library(const char* configured_library) {
  Dl_info api;
  char actual[PATH_MAX];
  char expected[PATH_MAX];
  return configured_library != NULL &&
         dladdr((void*)create_tokenizer, &api) != 0 &&
         realpath(api.dli_fname, actual) != NULL &&
         realpath(configured_library, expected) != NULL &&
         strcmp(actual, expected) == 0;
}

static void release_unassigned(void) {
  while (available != 0) {
    destroy_tokenizer(preinitialized[--available]);
  }
}

// Provision independent original tokenizer handles before Router starts.
// The original ABI retains ownership and executes every encode and decode.
static void initialize_tokenizers(void) __attribute__((constructor));
static void initialize_tokenizers(void) {
  const char* model = getenv("XLLM_ROUTER_PREINIT_TOKENIZER_PATH");
  const char* count_string = getenv("XLLM_ROUTER_PREINIT_TOKENIZER_COUNT");
  const char* library = getenv("XLLM_ROUTER_PREINIT_TOKENIZER_LIBRARY");
  if (model == NULL || count_string == NULL) {
    return;
  }
  char* end = NULL;
  long count = strtol(count_string, &end, 10);
  struct stat after;
  pthread_once(&api_once, resolve_api);
  if (end == count_string || *end != '\0' || count < 1 ||
      count > MAX_PREINITIALIZED_TOKENIZERS || create_tokenizer == NULL ||
      destroy_tokenizer == NULL || !verify_library(library) ||
      realpath(model, expected_path) == NULL ||
      stat(expected_path, &expected_file) != 0 ||
      !S_ISREG(expected_file.st_mode)) {
    fprintf(stderr,
            "XLLM_ROUTER_TOKENIZER_PREINIT status=configuration_rejected\n");
    return;
  }
  while (available < (size_t)count) {
    void* handle = create_tokenizer(expected_path);
    if (handle == NULL) {
      break;
    }
    preinitialized[available++] = handle;
  }
  if (available != (size_t)count || stat(expected_path, &after) != 0 ||
      !same_file(&expected_file, &after)) {
    release_unassigned();
    fprintf(stderr,
            "XLLM_ROUTER_TOKENIZER_PREINIT status=initialization_failed\n");
    return;
  }
  enabled = true;
  fprintf(stderr,
          "XLLM_ROUTER_TOKENIZER_PREINIT status=ready handles=%zu\n",
          available);
}

static void finalize_tokenizers(void) __attribute__((destructor));
static void finalize_tokenizers(void) {
  pthread_mutex_lock(&pool_lock);
  release_unassigned();
  pthread_mutex_unlock(&pool_lock);
}

__attribute__((visibility("default"))) void* tokenizers_new_from_path(
    const char* path) {
  pthread_once(&api_once, resolve_api);
  if (create_tokenizer == NULL) {
    return NULL;
  }
  if (!enabled || path == NULL) {
    return create_tokenizer(path);
  }
  char canonical[PATH_MAX];
  struct stat current;
  if (realpath(path, canonical) == NULL ||
      strcmp(canonical, expected_path) != 0 || stat(canonical, &current) != 0 ||
      !same_file(&expected_file, &current)) {
    return create_tokenizer(path);
  }
  pthread_mutex_lock(&pool_lock);
  void* handle = available == 0 ? NULL : preinitialized[--available];
  pthread_mutex_unlock(&pool_lock);
  return handle == NULL ? create_tokenizer(path) : handle;
}
