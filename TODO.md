# TODO

## Direction

The goal is not to rebuild llama.cpp or vLLM under a different name.

NERVE remains a continuous stream inference engine:

```text
running stream =
    runtime graph
  + compiled permanent component circuits
  + mutable transient circuit
```

llama.cpp and vLLM should be used as engineering references for the hard practical parts that mature inference engines already solve: scheduling, memory/state management, batching, graph reuse, kernel selection, quantization, MoE routing, and speculative decoding.

The product shape remains the execution graph model described in `CONCEPT.md`.

## Core architectural work

### 1. Build a real stream scheduler

Requests are not the primary runtime object. A request is an event injected into a persistent stream.

Implement a scheduler that operates on streams and stream activations:

- Track active, idle, interrupted, and closing streams.
- Admit external input events into existing streams without destroying stream state.
- Schedule running stream activations before newly waiting work.
- Maintain a token/work budget per scheduler step.
- Distinguish prompt/prefill work from decode/feedback work.
- Support chunked prefill for long inputs.
- Keep the model mounted between turns.
- Report timings from normal chat runs without special profiling modes.

The scheduler should schedule stream events, not convert NERVE into a stateless request/response server.

Current status:

- Core stream scheduler exists and tracks persistent streams, queued input events, prefill chunks, decode feedback windows, active/idle/interrupted/closing state, and normal executor-driven runs.
- Streams can be admitted atomically with package-derived transient state declarations and an execution class identity.
- Decode activations can represent bounded feedback windows instead of forcing one-token host stepping.
- Prefill completion can account for the first emitted feedback token when the backend samples from the final prompt activation.
- The placed Vulkan prompt engine now routes normal prompt/chat execution through the core stream scheduler while keeping the model mounted.
- Normal prompt/chat reports include prefill/decode token counts, activation counts, scheduler batch counts, maximum batch width, and prefill/decode timing without a special profiling mode.

### 2. Replace flat transient context with block-managed transient circuit state

KV is not conceptually a disposable cache in NERVE. It is stream-owned transient circuit state.

Use vLLM-style block management as a practical implementation clue, but expose it in NERVE terms:

```text
node instance state
  -> allocated in blocks/pages
  -> owned by a stream
  -> referenced through state tables
  -> reusable, freeable, forkable, and snapshot-able
```

Required pieces:

- A block/page allocator for transient state.
- Per-stream state tables.
- Per-component or per-state block tables.
- Slot mappings for writing new state.
- Ref counts for shared or reused state blocks.
- Free lists and safe reclamation.
- State reset, snapshot, fork, and eventual merge semantics.
- Support for attention KV, recurrent state, Mamba state, conv state, and other component-owned transient state.

This should remove the architectural need for arbitrary tiny capacity limits while preserving bounded, explicit resource management.

Current status:

- A backend-neutral transient state arena and per-stream state tables exist.
- State blocks are page-like, reusable, ref-counted, resettable, forkable, and snapshot-able.
- The stream scheduler reserves transient state slots per scheduled activation for declared stream-owned state.
- Placed Vulkan packages now expose dynamic per-activation state declarations from resident package metadata, and placed streams register those declarations with the scheduler at stream admission.
- Scheduled placed activations expose a transitional binding plan from scheduler transient slots to the current resident state-buffer offsets; the backend still needs to use real page-backed bindings instead of fixed circular state buffers.

### 3. Preserve layer components as runtime/editing/placement boundaries

For the first practical architecture, each source model layer remains a standalone source component in the compiled package.

The backend may fuse, split, tile, or lower internals however it wants, but the logical layer-component boundary remains available to the runtime graph editor.

The layer component is the unit that can be:

- placed on a device;
- bypassed;
- duplicated;
- inspected;
- migrated;
- replaced;
- connected over a short in-device edge or a longer cross-device edge.

Optimization must not erase the user-facing execution graph contract.

### 4. Put the scheduler below the runtime graph editor

The runtime graph decides what exists and how it is wired.

The scheduler decides when activations happen.

Target shape:

```text
UI/API event
   |
   v
stream scheduler
   |
   v
runtime graph
   |
   v
node instances + transient state pages
   |
   v
backend execution plan
```

Placement remains a runtime concern. The compiler should produce a neutral component catalog and canonical topology, not a hardcoded execution placement.

### 5. Make batch execution mean multi-stream signal processing

Batching should not contradict the stream model.

Instead:

```text
many active streams
        |
        v
same mounted execution graph window
        |
        v
many stream outputs and state updates
```

Implement batch execution over active streams:

- Batch decode ticks across multiple streams.
- Batch compatible prefill chunks.
- Keep per-stream state tables separate.
- Keep public output and private feedback signals separate.
- Avoid rebuilding or remounting the model for each prompt.

This is the stream/execution graph equivalent of vLLM continuous batching.

Current status:

- The scheduler can emit backend-neutral activation batches.
- Batch compatibility includes execution class identity, so streams from different packages, placements, or context capacities are not grouped together accidentally.
- The placed Vulkan prompt engine consumes scheduler batch steps and sizes scheduler budgets by current stream count.
- Placed scheduler batches execute through a dedicated batch executor seam, so real multi-stream Vulkan execution can replace the current sequential internals without rewriting the outer scheduler loop.
- Current placed execution still runs each activation inside a batch sequentially; actual batched Vulkan kernels need to consume the batch plan next.

### 6. Keep the device-owned feedback loop as the long-term target

vLLM schedules each step host-side. That is useful as a starting point, but NERVE's concept wants active generation to live as close to the device as possible.

Short-term acceptable path:

- Host scheduler admits events.
- Host builds bounded execution windows.
- Device executes those windows.
- Host receives output/control events.

Long-term target:

- Host injects events and receives public outputs.
- Device owns bounded steady-state feedback windows.
- Feedback token/state production stays on device where possible.
- Host intervention is for control, interruption, stream admission, UI, and cross-device orchestration.

### 7. Make kernel dispatch shape-aware

Do not treat every compiled linear/component operation as the same kind of shader problem.

Use llama.cpp as inspiration:

- Decode often wants matvec-style kernels.
- Prefill often wants matmul-style kernels.
- Quantized formats need native paths, not wide-scalar expansion unless proven faster.
- Backend dispatch should pick kernels based on runtime shape, dtype, device features, and component contract.
- Graph/kernel reuse should avoid hot-path allocation and recompilation.

Initial required dispatch families:

- BF16 dense decode and prefill.
- FP8 dense decode and prefill.
- INT4 dense decode and prefill.
- Attention/state-update paths.
- Mamba/recurrent/conv state paths.
- MoE route-native paths.
- Sampler and speculative decode paths.

Current status:

- Scheduler activation batches now map explicitly onto backend execution modes:
  prefill chunks are causal sequences, while decode feedback batches are
  independent candidates.
- Vulkan component-batch kernel selection already uses execution-domain metadata
  to choose decode, prefill, or shared decode/prefill implementations by shape.
- This is still only the dispatch vocabulary and selection seam; dense FP8/INT4,
  attention/state, MoE route-native, and speculative decode optimized kernel
  families still need real implementations.

### 8. Make MoE route-native

MoE components must not behave like dense FFNs with a mask.

Routing is a first-class signal:

```text
hidden signal -> router -> selected expert routes -> active expert components -> reducer
```

Required pieces:

- Router/top-k component output as explicit route signal.
- Expert execution that only processes selected experts/routes.
- Route grouping or route batching to reduce wasted work.
- Expert-shard placement across devices.
- Correct reduction using route weights.
- Tests that prove work and output shapes scale with selected routes, not total experts.

For MoE models with a small active parameter count, performance should reflect active experts, not the full declared model size.

Current status:

- The compiler represents sparse MoE layers as explicit route-native components:
  router/top-k, sparse expert gate/up, sparse expert down, and reducer.
- BF16, FP8, and INT4 sparse expert shader families are generated from the
  selected-route contract, and compiler tests now guard that expert workgroup
  counts scale with `experts_per_token`, not the total expert pool.
- This is not finished: route grouping/batching across streams, expert-shard
  placement, runtime route counters, and benchmarks proving active-expert
  scaling on real MoE packages are still required.

### 9. Make MTP/speculative decoding first-class

Speculative decoding should not be treated as benchmark garnish.

For models that ship MTP/draft components, the compiled package should expose them as components or execution graphs with explicit topology:

```text
main stream state -> draft component(s) -> proposed feedback tokens
                                |
                                v
main execution graph verifies/accepts/rejects
```

Required pieces:

- Compile MTP/draft components as reusable source components.
- Runtime graph support for draft components.
- Scheduler support for lookahead slots.
- Transient state rollback on rejected draft tokens.
- Acceptance-rate and accepted-token stats in normal runtime output.
- Correct behavior with thinking/reasoning models enabled normally.

Current status:

- The compiler discovers structural MTP/draft graphs and lowers them as
  auxiliary execution graphs with draft input adapters, draft processors, draft
  output transducers, and explicit transactional state contracts.
- The runtime can mount speculative decoder packages, run draft steps, verify a
  target prefix, restore rejected tentative state, catch the draft decoder up to
  the accepted prefix, and report proposed/accepted draft tokens plus timing in
  normal chat output.
- This is not finished: scheduler-native lookahead slots, multi-draft routing,
  larger validation on real thinking models, and warmed benchmarks with MTP
  enabled are still required.

### 10. Treat prefill as a first-class workload

Long context is a normal use case, not an edge case.

Implement:

- Chunked prefill.
- Prefill/decode interleaving.
- Batch prefill where compatible.
- Block allocation before prefill chunks.
- No arbitrary small token limits in tests or benchmarks.
- Normal 64k+ output capacity and large context operation.

Prefill speed and decode speed should be measured separately by default.

Current status:

- The stream scheduler emits chunked prefill activations and can batch
  compatible prefill chunks across streams when activation/work budgets allow.
- Prefill and decode token counts, activation counts, batch counts, and timings
  are reported separately in normal prompt/chat output.
- The runtime default generation budget is 65,536 new tokens rather than a tiny
  benchmark-oriented cap.
- This is not finished: true page-backed resident state bindings, prefill/decode
  interleaving under memory pressure, and real long-context validation are still
  required.

### 11. Add prefix/state reuse after block-managed state exists

Do not start with clever prefix caching before the state allocator is solid.

Once block-managed transient state exists:

- Hash full state blocks by token prefix and relevant runtime modifiers.
- Reuse block-aligned prefix state across streams.
- Keep ref counts for shared blocks.
- Evict with LRU or better policy.
- Keep model/component identity and runtime graph identity in cache keys.
- Support future external or remote state connectors.

This maps to vLLM prefix caching while preserving NERVE's transient-circuit semantics.

### 12. Make graph/kernel reuse explicit

Avoid rebuilding execution shape on every prompt or token.

Required pieces:

- Separate prefill and decode execution plans.
- Reusable mounted execution graph plans.
- Reusable batch-size/token-count execution templates.
- Persistent descriptor/buffer layouts.
- Hot-path metadata updates without allocation.
- Stable graph identity based on runtime graph, placement, shape class, and state layout.

This is the Vulkan/SPIR-V analogue of graph reservation/reuse in llama.cpp and CUDA graph capture/replay in vLLM.

Current status:

- Mounted Vulkan dispatch segments keep pipelines, descriptors, command buffers,
  and fences resident for the lifetime of the mounted model.
- Resident kernel sequences are cached by execution variant/lane and replay
  recorded commands when their dispatch shape has no dynamic push constants.
- Normal chat output includes resident sequence record/reuse/submit counters, so
  graph/kernel reuse is visible without a special profiling mode.
- This is not finished: stable runtime graph identity, reusable prefill/decode
  template catalogs, and hot-path metadata updates for page-backed state still
  need to be made explicit.

## Validation expectations

Every meaningful architectural change must preserve model usability.

Validation should include:

- Teacher-forced source/compiled comparison where available.
- Free-running chat validation.
- Multi-turn memory checks.
- Thinking/reasoning mode enabled for thinking models.
- Warmup run discarded when reporting benchmark averages.
- Five normal chat requests for benchmark sanity:
  - `hi`
  - `Who are you?`
  - `what is the capital of Greece?`
  - `How many cities named "Corinth" are there?`
  - `What is your knowledge cutoff date?`
  - `I asked you earlier to tell me the capital of a country. Which country was that?`
- Per-model validation after changes that touch compiler/runtime behavior.
- No broad Vulkan test runs.
- No parallel tests.
- No NVIDIA NERVE workloads.

## Near-term implementation order

1. Define runtime stream/request/state scheduler data structures in Rust.
2. Introduce block-managed transient state tables independent of any one model architecture.
3. Route current single-chat execution through the scheduler with one stream.
4. Add decode batching across streams without changing the compiled package format more than necessary.
5. Add chunked prefill.
6. Make attention/recurrent/Mamba state use the block/state table abstraction.
7. Move current fixed-capacity feedback state onto the block-managed transient circuit.
8. Add shape-aware dispatch selection for decode versus prefill kernels.
9. Make MoE route execution truly active-route based.
10. Wire MTP/speculative decoding as real runtime flow.
11. Add prefix/state reuse.
12. Re-benchmark all available models and compare against llama.cpp using warmed runs.

## Non-goals

- Do not turn NERVE into a clone of llama.cpp.
- Do not turn NERVE into a clone of vLLM.
- Do not make compiler output depend on runtime placement.
- Do not hardcode model-specific behavior into core runtime files.
- Do not solve performance by adding arbitrary low limits.
- Do not optimize only benchmark prompts at the expense of real chat usability.
- Do not erase component boundaries to gain short-term speed.
- Do not add compatibility layers for old package formats unless there is a concrete current need.

## Success criteria

The engine is moving in the right direction when:

- A compiled package remains a reusable component catalog.
- Runtime graph editing controls placement, topology, duplication, and bypass.
- A stream survives across multiple input events without remounting the model.
- Transient state is explicit, stream-owned, and block-managed.
- Multiple active streams can share mounted permanent circuits.
- Decode and prefill use different optimized execution paths.
- MoE models execute proportional to active routes.
- MTP/speculative decoding works as part of normal generation.
- Normal chat output includes useful performance stats.
- Benchmarks discard warmup and report realistic multi-request averages.
- The implementation still feels like the continuous stream/execution graph architecture in `CONCEPT.md`.
