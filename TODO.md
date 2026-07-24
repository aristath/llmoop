# TODO

## Direction and constraints

NERVE is a continuous-stream execution engine:

```text
running stream =
    runtime graph
  + compiled permanent component circuits
  + mutable transient circuit state
```

The compiled package is a neutral component catalog plus canonical topology.
Runtime configuration owns graph edits, component placement, stream scheduling,
and device selection.

The logical source-layer component remains the graph-editing and placement
boundary. Backends may fuse, split, tile, or otherwise lower work inside that
boundary, but optimization must not erase the user's ability to place, duplicate,
bypass, replace, migrate, or reconnect the component.

Use llama.cpp, vLLM, and other mature engines as engineering references, not as
architectural templates. Do not add model-name-specific runtime behavior,
compiler-time placement, obsolete-format compatibility layers, arbitrary small
context/output limits, or benchmark-only shortcuts.

## Remaining work, in priority order

### 1. Build genuinely optimized shape- and dtype-specific kernel families

Kernel selection exists, but important paths still use generic or matvec-shaped
work where the workload calls for a different implementation. On a clean
Qwen3.6-27B-FP8 package, the current runtime produced 14.519 decode tokens/second
on the first measured turn before a later turn exposed the separate stochastic
correctness failure tracked in the final gate. This remains below the 20
tokens/second minimum.

#### Decode

- Fuse operations within a logical component where doing so removes intermediate
  traffic or dispatches without erasing the component boundary.
- Quantize an activation once per reusable scope instead of repeating activation
  quantization for every output tile.
- Reduce dispatch count in component hot paths; large Qwen components currently
  produce hundreds of primary kernel dispatches per tick.
- Add a native tiled/dot-product path for large BF16 projections, especially the
  output projection, without converting the stored BF16 weights.
- Evaluate fusing final projection, candidate reduction, and sampling where that
  preserves exact runtime semantics.
- Optimize attention and state-update kernels for increasing context length and
  remove avoidable serial reduction/softmax regions.

#### Prefill

- Implement true FP8 cooperative-matrix or tiled matrix-matrix kernels rather
  than executing prefill as repeated FP8 matrix-vector work.
- Add corresponding optimized BF16 and INT4 prefill families where the device
  supports them.
- Implement causal batched state updates for attention, recurrent, Mamba, and
  convolutional components.

#### Compilation and selection

- Compile kernel variants from operation shape, source dtype, and required device
  features—not model names.
- Preserve a model's native dtype whenever the selected device supports it.
- Select variants at runtime from shape, batch width, context state, and actual
  device capabilities.
- Maintain correctness tests and representative microbenchmarks for every
  optimized family.

### 2. Replace fixed one-component submission quanta with calibrated work quanta

The current conservative submission boundary avoids long graphics-ring jobs but
leaves substantial scheduling and submission overhead.

- Populate `RuntimeExecutionCost` from compiled component work, memory traffic,
  dispatch count, and kernel characteristics.
- Calibrate safe quantum limits per device and kernel family.
- Coalesce multiple adjacent components into one submission when the calibrated
  cost permits it.
- Preserve explicit yield, interruption, and transient-state commit boundaries.
- Adapt quanta without reintroducing graphics-ring timeouts.
- Expose quantum size, estimated cost, actual duration, and forced-yield metrics
  in normal runtime statistics.
- Decouple interruption responsiveness from a synchronous host wait after every
  small feedback window. The current adaptive 2-3-tick windows kept discarded
  work near zero, but a 1,908-tick real conversation turn still incurred 1,391
  fence waits and 1,492 queue-batch submissions.

### 3. Execute scheduler batches as real multi-stream Vulkan work

The scheduler can form compatible batches, but the Vulkan batch executor still
processes their activations sequentially.

- Consume compatible decode activations in batched kernels.
- Batch compatible prefill chunks.
- Keep transient state, control signals, and public output separate per stream.
- Support continuous stream admission, prefill/decode interleaving, cancellation,
  and fairness while the model remains mounted.
- Avoid increasing single-stream latency merely to report a wider logical batch.

### 4. Finish physical block-managed transient state

The backend-neutral allocator and logical state tables exist, but resident state
storage is still fundamentally flat and host offsets remain in the execution
path.

- Allocate GPU-resident physical page/chunk pools for every transient state kind.
- Bind scheduler block IDs to device-visible page tables.
- Implement allocation, rebinding, free, eviction, and safe reclamation.
- Perform hot-path state lookup on the device.
- Make reset, snapshot, fork, shared-prefix state, and copy-on-write semantics
  physically correct.
- Cover attention KV, recurrent, Mamba, convolutional, speculative, and future
  component-owned state through the same abstraction.
- Remove flat resident buffers as the authoritative state model.

### 5. Wire prefix/state reuse into normal stream admission

Prefix-state primitives exist, but normal chat does not automatically use them.

- Restore the longest compatible cached prefix when admitting prompt input.
- Insert reusable block-aligned state after normal prefill.
- Serialize every runtime modifier that can affect state into the cache key.
- Key reuse by canonical graph identity, exact placement, component/state layout,
  model/package identity, and token prefix.
- Connect cache references to physical page refcounts, copy-on-write, eviction,
  and reclamation.
- Report hits, misses, reused tokens, saved prefill work, and eviction behavior.
- Validate reuse with real multi-turn and branched conversations.

### 6. Define canonical runtime graph identity and reusable execution templates

Execution-class compatibility, prefix reuse, and command/template reuse need one
precise graph identity.

- Canonicalize source component references, node instances, topology, edge kinds,
  duplication/bypass edits, exact component placement, state layout, shape class,
  and selected kernel variants.
- Include runtime modifiers when they change execution or transient-state
  semantics.
- Use the identity consistently for scheduler compatibility, prefix-state keys,
  resident execution plans, and feedback templates.
- Maintain reusable prefill, decode, and batch template catalogs.
- Keep hot metadata in persistent buffers and update it without re-recording or
  reallocating unaffected work.
- Invalidate only templates affected by a graph edit, placement change, or shape
  transition.
- Eliminate per-turn recording of the unchanged 65-command resident sequence and
  replace the stream-local current-shape feedback template with a synchronized
  catalog that can safely replay every compatible shape. Timeline values must be
  rebased without giving independently recorded templates stale relative
  offsets.

### 7. Make cross-device execution efficient without making it mandatory

Everything may run on one device. Multi-device execution should become useful
when requested by placement or required by model size.

- Choose edge transport from runtime capabilities: same-device aliasing,
  peer/external memory, device-local transfer, host staging fallback, and
  eventually LAN transport.
- Remove host-backed activation edges from the fast path when peer/device-local
  transfer is available.
- Use asynchronous timeline synchronization and overlap transfers with independent
  device work.
- Support tensor or expert sharding inside a logical component when that is
  materially better than transferring a full activation between layer
  components.
- Keep the logical source-layer boundary intact even when its internal work is
  sharded.
- Report transfer route, bytes, waits, and overlap per graph edge.
- Compare single-device and necessary multi-device placements; do not force extra
  devices into benchmarks.

### 8. Complete route-native MoE execution

Sparse components and selected-route kernels exist, but routing is not yet a
fully optimized runtime signal path.

- Group and batch selected routes across tokens and streams.
- Execute only selected experts and prove this with runtime work counters.
- Place or shard experts across devices without dense all-expert work.
- Keep route weights and reduction on the device.
- Make route signals participate in resident execution templates and feedback
  control.
- Validate output correctness and active-expert scaling on real MoE packages.
- Make the 35B MoE model's performance reflect its active parameter count rather
  than its full declared size.

### 9. Integrate MTP into the steady-state scheduler and device loop

MTP compilation and transactional verification work, but speculative execution
is not yet part of the optimized steady-state path.

- Add scheduler-native lookahead slots and multi-draft routing.
- Keep draft proposal, target verification, acceptance, rollback, and catch-up on
  the device where practical.
- Ensure enabling MTP does not disable resident feedback execution or introduce a
  host synchronization point per token.
- Keep thinking/reasoning behavior enabled normally during validation.
- Report proposal count, acceptance, rollback, useful tokens, and timing in
  normal chat output.
- Enable MTP by default only where warmed, realistic workloads show a net
  improvement.

### 10. Finish long-context prefill and mixed-workload scheduling

- Interleave prefill and decode fairly under memory pressure.
- Derive prefill chunk size from available memory, device execution limits, and
  selected kernel shape.
- Batch compatible prefill work across streams.
- Preallocate, reclaim, and compact physical state pages safely around long
  prompts.
- Validate 64K/128K context and long agentic outputs without arbitrary low token
  limits.
- Report prefill and decode throughput separately by default.

### 11. Maintain adversarial correctness and performance gates

Every meaningful compiler, runtime, state, graph, or kernel change must be tested
against the supported model set rather than optimized around one model.

Correctness coverage must include:

- Teacher-forced source-versus-compiled comparisons where a source runner is
  available.
- Real free-running, multi-turn conversations.
- Thinking/reasoning enabled for thinking models.
- Graph duplication, bypass, rewiring, and placement changes.
- State reset, snapshot, fork, shared prefix, copy-on-write, and reclamation.
- EOS/cancellation tests proving unused feedback-window tail work did not execute.
- Long-context and long-output operation.

Performance runs must:

- Keep the model resident for the full run.
- Use `hi` only as the discarded warmup request.
- Average the following five measured conversation turns:
  1. `Who are you?`
  2. `what is the capital of Greece?`
  3. `How many cities named "Corinth" are there?`
  4. `What is your knowledge cutoff date?`
  5. `I asked you earlier to tell me the capital of a country. Which country was that?`
- Use a 65,536-token output allowance and a realistic context allocation unless
  the test explicitly measures another context size.
- Report setup separately; report prefill and decode throughput, useful versus
  executed ticks, placement, device identities, kernel variants, and MTP state.
- Compare equivalent warmed settings with llama.cpp or vLLM where applicable.
- Exercise one device when the model fits and only the devices actually required
  when it does not.
- Treat 20 decode tokens/second on Qwen3.6-27B-FP8 as the current minimum target,
  not the final optimization ceiling.
- Exercise multiple fixed seeds for stochastic samplers. A seed-1 27B run
  completed the full conversation correctly, including cross-turn recall, while
  seed 0 entered a multi-thousand-token repetition on the Corinth question;
  neither arbitrary output caps nor a single convenient seed is a valid
  correctness gate.
- Make fixed-seed sampling and execution reproducible. A fresh seed-1 package
  whose tensors, shaders, and executable artifacts were byte-identical to the
  earlier successful package entered an unbounded emoji loop on the second
  measured turn. Fail the gate on repeated final segments, malformed thinking
  boundaries, turn contamination, or failure to terminate after a valid answer;
  generating some meaningful text is insufficient.
- Make the structural Rust test gate self-contained. It currently panics when a
  deleted external 230M lowered-model fixture is absent; tests must use a
  checked-in or deterministically generated fixture, or skip with an explicit
  unsupported prerequisite rather than report an implementation failure.

Repository safety requirements remain mandatory: run tests sequentially, select
Vulkan tests individually, never run a NERVE workload on the NVIDIA GPU, and
verify AMD device residency before and after every GPU workload.
