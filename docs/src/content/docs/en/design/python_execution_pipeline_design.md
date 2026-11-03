---
title: "Python Staged Execution Pipeline"
sidebar:
  order: 6
---
<!--
Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## Unified Execution Contract

DSpark combines the backbone and LM head into `DSparkDecoderModel`, with sampling handled by a separate
`DSparkProposalModel`. Both execution stages are `nn.Module` instances and retain the
`forward(input_ids, positions, input_embedding=None)` signature. Each model consumes only the inputs it needs:
sampling does not have to provide Attention/KV capabilities, and logits are not passed as token IDs.

The Decoder Model runs the backbone and LM head in one forward pass and always returns `(logits, hidden)`.
Its output structure does not change depending on whether confidence is requested. Logits are passed to the
Proposal Model through the existing `input_embedding` argument. As in MTP, this argument can carry intermediate
features, not just token embeddings. `input_ids` always contains actual token IDs; the Proposal Model uses them
to obtain each request's anchor, so logits must not be substituted for IDs.

Module-specific auxiliary data is accessed through typed entries in `ForwardContext.execution_contexts`.
The pipeline puts backbone hidden states into `ProposalMetadata` only when the request needs confidence;
sampling parameters are still supplied by `ModelSamplingMetadataBuilder`. Runners establish and exit the context
for each forward pass, and models do not retain the current inputs across requests. Layer loading, layer completion
events, and EPLB inputs are passed only to the Decoder stage containing the backbone, not to the Proposal stage.

Having a metadata builder does not mean that every request must be graph-compatible. A model can provide
`can_execute_graph`; the graph backend combines this capability check with the batch layout and capacity checks.
The unified `Runner` consumes the result and selects graph or eager execution before running the computation.
Capture/replay failures are reported directly, rather than rerunning eagerly after side effects may have occurred.

## Hierarchy and Ownership

```text
ModelExecutor
  └─ DSparkExecutionPipeline
       ├─ entry_runner    → Runner(backbone + LM head) → EagerRunner / existing DecodeAclGraphRunner and other backends
       └─ proposal_runner → Runner(proposal) → EagerRunner / AclGraphRunner
```

`Runner` in `model_executor/runner.py` is the unified execution entry point for a single Model. It composes concrete
execution backends rather than inheriting from them. Each `Runner` binds one Model and its metadata builders, and
owns the eager backend and any graph backend allowed by the service configuration. Its `execute(ModelInput)` method
consumes the graph backend's `can_execute` result, selects a backend, and invokes it. The execution mode, the row count
from the most recent graph padding, and the history restoration callback also belong to this Runner, so different
Models do not overwrite one another's state.

During its construction, `ExecutionPipeline` creates the corresponding Runners through
`register_model(model, builders)`. Each Runner constructs its eager backend and any graph backend allowed by the
service configuration, then binds the metadata builders to those backends. All backends reference the same Model;
weights are neither reloaded nor copied. There is no phase that constructs an empty Runner and later calls
`bind_backends`, nor does the Executor iterate over the Pipeline's Runners to attach backends afterward.

`RunnerConfig` collects existing initialization parameters, including the device, model configuration, graph-enable
setting, and capacity limits. It introduces no new user-facing switches. The Executor retains Attention backend
preparation for the current single-main-model setup and passes that backend, the RunnerConfig, and an optional history
restoration callback to the Pipeline. The Pipeline forwards the configuration to Runners without interpreting backend
selection rules. The main-model Runner uses the existing decode backend; the Proposal stage has no Attention backend
and uses a non-Attention execution backend. Execution-environment compatibility and capacity constraints remain the
responsibility of the backends.

Constructing a backend is not the same as capturing a graph. KV caches are still bound once upstream resources are
ready; warmup, initial capture, and replay retain their existing timing. Runners and backends are not reconstructed
on every step. This change does not extend resource assembly to multiple independent Attention-based main models.

A concrete pipeline's `execute` method calls only `entry_runner.execute` and `proposal_runner.execute`, defining the
computation order, data flow, and waits. The Pipeline does not interpret the graph switch, construct `EagerRunner` or
`AclGraphRunner`, or decide the execution mode for the current step. The common `_execute_model` logic has moved into
`Runner.execute`. The existing ordering of warmup, Engram history restoration, and model-layer loading waits is
preserved, and the restoration callback is bound only to the entry Runner.

Models without a pipeline construction entry point use `SingleModelExecutionPipeline`, preserving their existing
Tensor/tuple outputs and eager/graph execution paths. The generic executor does not check for DSpark-specific types.

`DSparkModel` in `models/dspark_model.py` is a container for components and weights, without an overall `forward`.
It preserves access paths for checkpoints, the shared LM head, and C++ helper methods. The two computation Models
reference the container's weights without registering the entire container as their own submodule. The pipeline
holds the container to keep its resources alive during execution. `pipelines/dspark_pipeline.py` is the only DSpark
orchestration entry point, even when all stages execute eagerly.

## Encapsulation of Orchestration and Asynchrony

The common interface standardizes the input/output contract of `execute`, not the internal scheduling algorithm.
DSpark currently calls its two Models sequentially; other concrete pipelines may use threads, Futures, or multiple
streams internally. The common layer does not require a unified task graph, execution handle, or `ExecutionResult`,
and the Executor does not consume Futures.

When a Pipeline returns an ordinary Tensor, tuple, or `ModelOutput`, it must already have established the dependencies
needed for consumption on the caller's stream. This does not require device execution to have finished, much less a
device-wide synchronization on every call. The concrete pipeline owns error handling and resource lifetimes for its
internal asynchronous work. It must not leave the caller with unhandled failures or buffers that have been released
while still in use.

A model may express its input waits directly, for example by obtaining an H2D ready event from a typed context and
waiting at the layer that actually consumes the data. The pipeline owns when events are produced and recorded; the
model owns the wait at the consumption point. No additional deferred-input accessor is needed. Graph execution must
still preserve the capture bindings and replay contracts for events and persistent buffers. This section defines the
boundary for future reuse: the current change neither adapts Engram nor introduces threads, Futures, or new events
into the draft path.

## Input Resources and Graph Execution

- Eager: the runner passes forward arguments directly, and the builder's `build` constructs the auxiliary metadata
  visible to the current Model.
- Capture: the runner fixes the addresses of forward arguments, and the builder's `allocate_persistent` allocates
  its own auxiliary resources.
- Replay: the runner updates argument buffers, and the builder's `update_persistent` updates auxiliary resources.
  Each clears its own padding without replacing the Tensor addresses captured by the graph.

`ForwardContext` is only a data-access mechanism; it does not automatically replace old graph addresses with this
batch's new Tensors. Host-side context lookups and branches are not rerun during replay. A model's graph capability
declaration must therefore ensure consistent control flow for the same graph identity, or select eager execution
for that stage. Relaxing a boolean admission check alone cannot make a graph reusable across different computation
structures.

Models without Attention still reuse `EagerRunner`, but skip Attention preparation. `AclGraphRunner` fixes ordinary
forward arguments and reuses the metadata resource protocol described above. It invokes models in the same way as
the retained `DecodeAclGraphRunner`, which continues to handle Attention/KV, DP protocols, decode-specific layouts,
and state protection. This change does not rewrite the large decode graph implementation into a universal backend;
the two backends share the underlying capture entry point.

The non-Attention graph backend selects buckets by request count and preserves each request's complete query block.
For example, three requests with three tokens each can be padded to four requests and twelve tokens. The graph cache
also distinguishes query width, non-row Tensor dimensions, and dtype. This backend selects eager execution before
running an empty, dummy, ragged, or over-capacity batch, without introducing a new user-facing switch.

Cross-stage tensors are copied into the next graph's persistent inputs. Stream waits protect production and
consumption, and source Tensors have their stream usage recorded. The Runner removes padding according to the output
contract: hidden/logit Tensors are trimmed by token count, while proposals are trimmed by request count. Structured
eager outputs also use upstream real counts to trim dummy requests; this responsibility does not move into the Model
or pipeline.

## DSpark Computation and Fallback

1. The Decoder Model invokes the backbone and the original LM head in one forward pass, giving them a shared Runner
   and capture scope. It aggregates their graph capabilities and binds only the metadata builders needed by the
   backbone. Sampling conditions do not block this stage from using a graph. Weight sharing performed after pipeline
   construction remains visible, and the backbone's backend configuration and runtime hooks remain accessible.
2. The Proposal Model consumes logits, anchors, sampling metadata, and optional hidden states. It preserves per-position
   Markov score adjustment, native sampling, token remapping, and optional probability/confidence outputs.

| Condition | Proposal execution mode |
| --- | --- |
| Unconstrained greedy sampling, with no probability or confidence output | Can use its own ACL graph |
| Penalty/filter, random sampling, probability output, or confidence output | Eager for this stage |
| ACL graph disabled, or the graph backend cannot handle the batch layout/capacity | Eager for this stage |

Random sampling, full probabilities, and confidence are not computed unconditionally just to maintain a single graph.
The backbone and LM head select their execution mode together, while sampling selects its mode independently.
Falling back to eager for complex sampling does not affect the preceding stage's graph execution.

## Validation Scope

Related tests cover backend binding completed during construction, shared models and builders, Runner backend
selection and state isolation, Pipeline orchestration through the unified entry point, the common forward/context
contract, explicit intermediate arguments, persistent input addresses, padding updates, independent fallback and
recovery, Markov feedback, probabilities/confidence, weight sharing, dummy trimming, and executor assembly. The
two-Runner cases also cover the backbone and LM head sharing one capture, sampling fallback and recovery, fixed
logit/hidden outputs, and padding removal for the combined stage. CPU graph simulations validate orchestration and
buffer behavior only; they are not equivalent to real NPU capture/replay.

On non-ACL platforms, the Proposal stage initially uses eager execution, while the Decoder stage retains the existing
model backend. Stages containing collectives still require consistent input layouts and call ordering within their
communication group. The Decoder Model participates in admission coordination through the existing graph capability
interface without changing shared-layout or bucket rules. Real NPU execution, TP behavior, and performance require
separate validation. Splitting execution into stages introduces graph launches and intermediate Tensor transfers;
the architecture alone does not demonstrate a TPOT improvement.

## Design Review and Responsibility Summary

The central principle is:

> **Model defines what to compute, Runner decides how to execute it, ExecutionPipeline defines how computations are
> combined, and Executor connects them to the service execution entry point.**

This change is not merely a file split: it separates responsibilities that were previously mixed together.

### 1. Execution Hierarchy

```text
Executor
  └─ ExecutionPipeline
       ├─ Runner(Model A)
       │    ├─ EagerRunner ─┐
       │    └─ GraphRunner ┴─→ Same Model A
       └─ Runner(Model B)
            ├─ EagerRunner ─┐
            └─ GraphRunner ┴─→ Same Model B
```

GraphRunner is a generic label for graph backends in this diagram. The actual implementations are the existing
`DecodeAclGraphRunner`, `AclGraphRunner`, and related backends, constructed only when the service configuration allows
them. Their existence does not mean that the current request necessarily executes as a graph.

| Layer | Core responsibilities | Not responsible for |
| --- | --- | --- |
| Executor | Prepare the execution environment, inputs, and service-boundary information; invoke the Pipeline | Orchestrating internal stages or choosing a backend for each batch |
| ExecutionPipeline | Define stages, data flow, execution order, and asynchronous dependencies | Deciding whether each stage uses eager or graph execution |
| Runner | Bind one Model, construct backends, and select the current execution mode | Knowing which stages are upstream or downstream |
| Concrete execution backend | Eager execution, or capture/replay, persistent buffers, and padding handling | Defining application-specific orchestration |
| Model | Implement a stage's computation and declare its graph capabilities | Selecting an execution backend |

**Runner composes concrete backends, and those backends share the same Model without copying or reloading weights.**

### 2. Construction and Resource Ownership

```text
Executor prepares configuration and dependencies
  → Creates the concrete Pipeline
    → Pipeline constructs the Runners it needs
      → Each Runner constructs and binds its eager/graph backends
```

- **The concrete Pipeline knows which Models it needs**; the Executor does not need to know their count in advance.
- Once Pipeline construction finishes, Runners and their backends are fully bound. The Executor does not attach
  backends in a second pass.
- Whether a component is fixed or replaceable affects where it comes from, not the execution abstraction.
- `DSparkModel` retains ownership of components, weights, and shared-weight resources. Stage Models reference those
  resources, and the Pipeline holds the resource container to preserve its lifetime.

Here, the number of Models means the number of independently executed stages, not the number of weight-loading
operations or the total number of `nn.Module` instances in the model tree. There is no need to scan the model tree
to infer how many Runners to create.

Completing construction does not mean that every device resource is ready: **KV caches can still be bound after
upstream preparation, and graphs are first captured at the existing execution points.**

### 3. DSpark Implementation

```text
DSparkExecutionPipeline
  ├─ entry_runner    → Decoder Model (backbone + LM head) → (logits, hidden)
  └─ proposal_runner → Proposal Model → ModelOutput
```

Both stages retain the unified `forward` contract. There is no separate Model/Unit distinction or second Runner
interface.

- Primary intermediate Tensors are passed explicitly through `input_embedding`.
- Module-specific auxiliary data is accessed through typed `ForwardContext` entries.
- Since the Proposal stage does not need Attention, no artificial Attention/KV resources are created for it.

**Stages are divided by execution characteristics, not mechanically by module count.** The backbone and LM head are
captured together as one execution unit, while sampling forms another unit that can fall back independently. When
complex sampling requires eager execution, the preceding stage can still use a graph. Composition boundaries are
chosen during development; execution modes are selected at runtime. This is not dynamic per-batch graph merging
or splitting.

### 4. Graph and Metadata Semantics

The earlier hypothesis that a Model must support graph execution unconditionally has been dropped. The current
contract is:

- A Model declares the graph conditions for this step's computation and metadata through
  `GraphExecutionSupport.can_execute_graph(inputs: ModelInput)`.
- Modules receive the runtime information they need through their own `can_execute_graph` methods. The Model
  explicitly aggregates the modules it directly uses, then adds conditions imposed by their composition.
- The graph backend checks its own input layout, device, capacity, and execution context, then consumes the Model's
  aggregate result. It does not bypass the Model to query builders for admission.
- Runner selects a backend before execution. It does not catch a graph failure after side effects have occurred
  and rerun the computation eagerly.

Therefore, **simple requests are no longer forced to execute every sampling operation just to preserve a single
all-purpose graph.**

The three core metadata builder interfaces remain unchanged:

| Interface | Responsibility |
| --- | --- |
| `build` | Construct metadata for the current inputs |
| `allocate_persistent` | Allocate persistent resources owned by the module |
| `update_persistent` | Update the contents of persistent resources |

**Having a builder means resource management has an explicit owner, not that every input is inherently
graph-compatible.** The backend owns forward-argument buffers and output trimming; the builder owns its auxiliary
resources.

#### Admission Call Chain

```text
Executor.execute(step)
  └─ ExecutionPipeline.execute(step)
       └─ Each stage's Runner.execute(stage_input)
            ├─ Graph disabled → eager.execute(...)
            └─ graph_backend.can_execute(stage_input)
                 ├─ Backend conditions: layout, device, bucket, capacity, execution context
                 ├─ Model.can_execute_graph(stage_input)
                 │    └─ Module.can_execute_graph(runtime information needed by the module)
                 └─ Cross-rank admission coordination where required
            ├─ True  → graph.execute(...)
            └─ False → eager.execute(...)
```

**Request information flows downward and capability decisions flow upward; only Runner makes the final choice
of execution mode.** The Pipeline does not reevaluate graph conditions, and the backend does not repeat Model
admission or retry failures after execution has started.

`ModelInput`, defined in `model_executor/model_input.py`, is the data contract for a stage's inputs, not a new execution
layer. It is shared by Runner, graph-backend admission, and Model admission, and contains Tensor arguments, Attention
metadata, `InputBatch`, and existing execution context. A model that needs sampling information reads
`inputs.input_batch.sampling` and passes that information to the sampling builder. Modules do not need to depend on
the entire execution layer just to receive sampling parameters. The backend's computation entry point and the Model's
`forward` signature remain unchanged; Runner unpacks ModelInput when invoking the computation entry point.

Model/module admission must be a side-effect-free check. `False` means that the current step should use the supported
eager path; `True` means only that computation conditions are satisfied, not that the backend necessarily has a
suitable bucket or buffer. The framework does not recursively scan the entire `nn.Module` tree. The Model or module
that owns the computation path expresses its composition explicitly.

The initial integration is limited to the DSpark draft pipeline:

- The Decoder Model aggregates the capabilities of the backbone and shared LM head, not sampling restrictions.
- The DSpark Proposal Model calls `ModelSamplingMetadataBuilder.can_execute_graph(sampling)` and aggregates the
  Markov head's capability. The sampling stage uses a graph only for unconstrained greedy sampling without
  probabilities/confidence; other sampling modes remain eager without rejecting graph execution for the backbone
  and logits. The Model and backend hold the same sampling builder instance. Persistent-resource validation reuses
  its `can_execute_graph` method to avoid maintaining two sets of rules. This change neither alters sampling
  numerics nor expands the set of sampling modes eligible for graphs.

This change does not migrate target Models or modify the MoE and decoder layers shared by V4.1 target/draft models.
The existing V4.1 MoE quantization-mode guards and CSA2 startup checks remain in place; their ownership can be
consolidated in a dedicated migration later.

`GraphExecutionSupport` is a structural interface. When it is not implemented, the common query defaults to `True`,
meaning **no additional computation constraints**. This neither bypasses backend checks nor means that the framework
has proven graph safety for the module. Request-independent prohibitions can return `False` directly; conditionally
supported modules inspect the current inputs. Ordinary modules do not need boilerplate methods that always return
`True`. Submodules are not scanned automatically: a Model must explicitly aggregate the conditional modules it uses,
and adding dynamic computation paths requires maintaining that declaration. Existing ordinary Models therefore retain
backend-only admission, avoiding a migration of all models at once. Backends invoke only the builder's three resource
interfaces; invalid persistent inputs still raise errors rather than being swallowed as eager fallback.

Both ACL and CUDA backbone graph backends consume Model admission. Multimodal restrictions and the CUDA EPLB
restriction also belong to the backends, so Runner no longer maintains a separate set of vetoes outside `can_execute`.
For a backbone implementing `GraphExecutionSupport`, when `dp_size > 1`, `dp_all_true` combines each rank's complete
local admission result. Graph execution requires every rank to agree. A rejecting rank must still participate rather
than return early and leave other ranks waiting for its vote. Global token counts still determine shared buckets;
they do not replace dynamic capability voting. Only DP backbones adopting the new interface gain this admission
collective. Unmigrated models retain their existing shared-layout admission without an additional vote, and the
Proposal stage's local `AclGraphRunner` does not add DP voting either.

### 5. Encapsulating Asynchronous Orchestration

The public interface does not require serial execution internally, nor does it require Futures or a unified task
graph to be exposed.

Callers continue to invoke ordinary `pipeline.execute(...)`. The concrete Pipeline establishes the necessary
consumption dependencies and manages errors and resource lifetimes for internal asynchronous work. Returning does
not require the GPU/NPU to have finished, but the caller must not receive results that lack the required dependencies.

The intended adaptation for DSV4.1 is:

- The concrete Pipeline owns the semantics of combining Engram and DSV4.1.
- The Pipeline schedules preparation work and H2D transfers.
- The target model obtains ready events from the context and waits at the layers that consume the data.
- Expressing these dependencies does not require splitting the target model or turning fixed components into
  replaceable plugins.

**This is an agreed design direction, not an Engram adaptation completed by this change.**

### 6. Current Scope and Trade-offs

- The implementation still targets **one main model requiring complete Attention/KV resources**, without prematurely
  extending resource assembly to multiple independent main models.
- The Executor still handles graph-related service initialization information and existing KV bindings, so its source
  code is not entirely graph-agnostic. Backend construction and per-batch selection, however, have moved downward.
- Staging isolates fallback, but may add graph launches and intermediate Tensor transfers.
  **Performance gains must be measured, not inferred from architecture alone.**
- CPU unit tests and real NPU validation provide different evidence. Combining stages in this change does not, by
  itself, establish a TPOT improvement.

The overall principle is: **unify the execution entry point without forcing a single execution mode; keep
application-specific composition in the concrete Pipeline, execution-mode differences in Runner, and resource
management in the modules that actually own the resources.**
