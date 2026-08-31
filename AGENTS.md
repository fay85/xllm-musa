# xLLM Coding Agent Instructions

## Directory Structure

```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── c_api/                     # code for c api
│   ├── cc_api/                    # code for cc api 
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── platform/              # adaption for various platform
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── function_call              # code for tool call parser
│   ├── models/                    # models impl
│   ├── parser/                    # parser reasoning
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
│   ├── pybind/                    # code for python bind
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

## Code Style Guide

### Mandatory Style Gate

* The upstream canonical rule is [xLLM custom-code-style.md](https://github.com/xLLM-AI/xllm/blob/main/.agents/skills/code-review/references/custom-code-style.md). Before the first code action, read the target checkout's complete local copy and verify it against upstream; when they differ, the upstream rule takes precedence and the discrepancy must not be ignored.
* Compliance with [.agents/skills/code-review/references/custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) is a **blocking requirement** for every task that writes, edits, generates, refactors, or reviews files under `xllm/`.
* Before the first code action, the primary agent **MUST read the entire style file from the current target checkout and the current upstream canonical version**. Cached knowledge, memory, summaries, or a copy from another checkout do not satisfy this requirement.
* If either style source is missing, unreadable, or only partially read, stop before modifying `xllm/` and report the blocker.
* Any delegated agent that writes or reviews `xllm/` code must be explicitly instructed to follow both style sources. Delegation does not transfer or waive this obligation.
* Before handoff, re-check the task diff against every applicable rule in the style file. Do not claim completion while a known violation remains.
* These project rules override conflicting generic style guidance. They may not be relaxed unless the user explicitly changes this project rule.

* Before editing, creating, refactoring, or reviewing any file under `xllm/`, you **MUST** read [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md).
* The file above is a **required instruction file**, not an optional reference. Do not skip reading it.
* Apply the rules in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) to **both code generation and code review**.
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* If [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) specifies a rule, that rule takes precedence over the Google C++/Python Style Guide.
* Use the Google C++/Python Style Guide only for cases not specified in [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md).

## Review Instructions

* For code review tasks, you **MUST** first read [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Then read [custom-code-style.md](.agents/skills/code-review/references/custom-code-style.md) and apply it during the review.
* Review code changes for quality, security, performance, correctness, and maintainability following the project-specific standards.
* Review code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Use the review workflow, checklist, severity rules, and output format defined in [code-review/SKILL.md](.agents/skills/code-review/SKILL.md).
* Apply the Google C++/Python Style Guide only when the project-specific style guide does not define the rule.
* Focus the review on the requested diff or changed files. Do not comment on unrelated code.
