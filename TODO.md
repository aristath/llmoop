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

### 1. Close the active feedback loop on the device

The device currently executes bounded feedback windows, but completion is still
observed at a host synchronization boundary and work already recorded after EOS
can still execute. Make the feedback loop behave like a real continuously running
signal path:

- Let sampling write device-resident continuation, interruption, and termination
  state.
- Predicate later ticks or use indirect dispatch so EOS, cancellation, or another
  stop condition prevents unused tail work from executing.
- Keep token feedback and transient-state updates on the device between useful
  host events.
- Keep persistent feedback submission templates alive across scheduler
  activations. Key them by canonical runtime graph, placement, execution shape,
  and state layout.
- Define exact state commit/rollback behavior at interruption and feedback-window
  boundaries.
- Size windows from safe execution cost and responsiveness, not from a semantic
  generation cap.
- Report planned, submitted, executed, retained, and discarded ticks so wasted
  work is visible.

The host should inject input/control events and drain public output/control
events. It should not be required to schedule every generated token.

### 2. Preserve modular layer anatomy through compilation

Make every logical source layer a modular pedal before changing its execution
further. Compilation must retain three related but distinct representations:

```text
semantic module tree
        |
        v
lowered semantic execution graph
        |
        v
optimized physical execution graph
```

The semantic module tree is the model anatomy exposed to the editor. The lowered
graph makes dataflow, parameters, state reads and writes, and transition ordering
explicit. The optimized graph may fuse, split, tile, reorder, or replace lowered
regions for a backend. A semantic module boundary is not automatically a kernel,
submission, placement, synchronization, or intermediate-buffer boundary.

- Keep the source layer as the top-level graph-editing, repetition, removal,
  placement, and state-policy pedal.
- Give each layer a stable, recursively expandable module tree. At minimum,
  distinguish its token-mixer block, feature-transform block, normalizations,
  residual paths, projections, gates, position operations, and state
  attachments.
- Represent attention KV, recurrent matrices, convolution history, and similar
  memory as state owned by the relevant mixer module, not as peer top-level
  pedals.
- Represent sparse MoE routing, selected expert bank, shared expert, reduction,
  and residual as submodules of the layer's feature-transform block. Allow the
  expert bank to expand into individual experts without making every expert a
  top-level pedal.
- Define a versioned module-tree schema with stable module IDs, roles, child
  modules, source-node membership, parameter references, state ownership, and
  module input/output contracts.
- Construct exact module membership while lowering each supported layer family;
  do not recover it later from node-name prefixes.
- Validate unique module IDs, complete and unambiguous source-node coverage,
  parent/child containment, valid parameter and state references, and root
  coverage of the layer.
- Preserve the semantic tree when producing optimized candidate circuits.
  Continue using source-node provenance such as `compiled_from` to associate a
  fused physical node with one or more semantic modules.
- Carry normalized source-node provenance into packaged kernel metadata so
  profiling, inspection, and future transformations can attribute physical work
  to semantic modules even when fusion crosses their boundaries.
- Add the module tree to the Rust circuit schema and editor model without
  changing flat-node execution planning or the one-kernel-per-optimized-node
  package contract.
- Make the layer pedal expandable in the UI. Show each submodule's responsibility,
  parameters, owned state, semantic nodes, optimized nodes, kernels, and measured
  cost where available.
- Treat layer clusters as composite editor groupings over layer pedals rather
  than replacing the layer as the compiled component boundary.
- Recompile packages after the compiler contract changes and add sequential,
  non-GPU schema, lowering, optimizer-provenance, package, editor, and UI tests
  for attention, convolution, gated-delta, RG-LRU, dense FFN, and sparse MoE
  layers.

This representation is the prerequisite for component-by-component
experimentation. A future transformation can select a semantic module, replace
its expression or representation, lower it again, verify its boundary behavior,
and still allow the backend to optimize across semantic boundaries.

### 3. Build genuinely optimized shape- and dtype-specific kernel families

Kernel selection exists, but important paths still use generic or matvec-shaped
work where the workload calls for a different implementation.

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

### 4. Replace fixed one-component submission quanta with calibrated work quanta

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

### 5. Execute scheduler batches as real multi-stream Vulkan work

The scheduler can form compatible batches, but the Vulkan batch executor still
processes their activations sequentially.

- Consume compatible decode activations in batched kernels.
- Batch compatible prefill chunks.
- Keep transient state, control signals, and public output separate per stream.
- Support continuous stream admission, prefill/decode interleaving, cancellation,
  and fairness while the model remains mounted.
- Avoid increasing single-stream latency merely to report a wider logical batch.

### 6. Finish physical block-managed transient state

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

### 7. Wire prefix/state reuse into normal stream admission

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

### 8. Define canonical runtime graph identity and reusable execution templates

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

### 9. Make cross-device execution efficient without making it mandatory

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

### 10. Complete route-native MoE execution

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

### 11. Integrate MTP into the steady-state scheduler and device loop

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

### 12. Finish long-context prefill and mixed-workload scheduling

- Interleave prefill and decode fairly under memory pressure.
- Derive prefill chunk size from available memory, device execution limits, and
  selected kernel shape.
- Batch compatible prefill work across streams.
- Preallocate, reclaim, and compact physical state pages safely around long
  prompts.
- Validate 64K/128K context and long agentic outputs without arbitrary low token
  limits.
- Report prefill and decode throughput separately by default.

### 13. Maintain adversarial correctness and performance gates

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

Repository safety requirements remain mandatory: run tests sequentially, select
Vulkan tests individually, never run a NERVE workload on the NVIDIA GPU, and
verify AMD device residency before and after every GPU workload.
