# NERVE

**Neural Execution & Rewiring Virtual Engine**

NERVE is an experimental inference engine for running language models as long-lived, editable execution graphs instead of one-shot request/response jobs.

The design is documented in [`CONCEPT.md`](CONCEPT.md). The short version is:

- A model becomes a compiled, self-contained package.
- The package exposes reusable components, parameters, ports, kernels, transducers, state declarations, and a canonical topology.
- Runtime creates a concrete execution graph from that package.
- A stream is the primary runtime object.
- User prompts are events injected into an existing stream.
- Transient state belongs to the stream and is part of the running circuit.
- Placement and wiring are runtime decisions, not compiler decisions.

The original intuition was a guitar pedalboard with feedback, but the project now uses standard graph/compiler/runtime language: components, node instances, ports, edges, runtime graphs, placement, transports, streams, and transient state.

## Current status

NERVE is not a production inference engine yet. It is a real implementation of the architecture, but still under active construction.

### Currently implemented

- Safetensors source-model discovery and compilation.
- Self-contained compiled model directories under `compiled_models/`.
- A package manifest named `vulkan_resident_package.json`.
- Transpiled model graph artifacts under `transpiled/`.
- Lowered stream-circuit artifacts under `lowered/`.
- Packaged tensors, tokenizer files, SPIR-V shaders, runtime config, and artifact integrity metadata inside the compiled model directory.
- Rust runtime graph types with node instances, edges, placement, duplicate nodes, explicit chains, and state policies.
- Runtime device discovery and logical-to-physical device binding.
- Vulkan/SPIR-V resident package mounting.
- Interactive chat and one-shot prompt execution through `nerve-runtime`.
- A TUI entrypoint through `nerve-tui`.
- A stream scheduler with persistent streams, queued input events, chunked prefill, decode feedback windows, activation batching, timing counters, and normal chat performance reporting.
- Backend-neutral transient state arenas and per-stream state tables.
- Ref-counted transient state blocks with reset, fork, share, snapshot, and prefix-cache primitives.
- Route-native MoE compiler structures and shader families for selected expert routes.
- MTP/speculative decoding package structures and runtime flow.
- Shape-aware dispatch vocabulary for prefill versus decode.

### Important unfinished work

- The Vulkan backend still has transitional fixed/circular resident state buffers in places. The scheduler has page/block-managed transient-state semantics, but resident Vulkan bindings still need to become truly page-backed before automatic prefix reuse can be wired into normal prompt admission.
- Multi-stream activation batches are scheduled, but some placed Vulkan batch paths still execute activations sequentially internally.
- FP8, INT4, MoE, MTP, prefill, and decode paths all need more real-model benchmarking and kernel work.
- The TUI exists as a runtime surface, but the full graph-editing product experience is still being built.

## Repository map

### [`CONCEPT.md`](CONCEPT.md)

The architectural source of truth. Start here to understand the model: streams, compiled packages, components, runtime graphs, placement, transient state, feedback, and Vulkan/SPIR-V as the baseline backend.

### [`TODO.md`](TODO.md)

The active engineering backlog and current implementation status. This tracks scheduler work, block-managed state, batching, dispatch shape, MoE, MTP, prefill, prefix/state reuse, and graph/kernel reuse.

### [`nerve/`](nerve/)

Python compiler and CLI package.

| Path | Purpose |
| --- | --- |
| `nerve/cli.py` | User-facing command dispatcher. |
| `nerve/model_compiler.py` | High-level source discovery, staged compilation, and atomic publish into a self-contained compiled model directory. |
| `nerve/model_package.py` | End-to-end package build: transpile, lower, copy tokenizer/tensors, compile shaders, build manifest, validate package. |
| `nerve/model_transpiler*.py` | Source checkpoint discovery and conversion into model/circuit graph facts. |
| `nerve/circuit_*.py` | Stream-circuit IR, lowering system, lowering operators, and optimization. |
| `nerve/model_package_manifest.py` | Builds `vulkan_resident_package.json`. |
| `nerve/model_package_tensors.py` | Tensor packaging. |
| `nerve/model_package_shaders.py` | Shader packaging. |
| `nerve/model_package_shader_selection.py` | Shader selection. |
| `nerve/model_package_shader_templates.py` | Shader template rendering. |
| `nerve/model_package_shader_compiler.py` | SPIR-V artifact creation. |

CLI examples:

```bash
python -m nerve --discover-model MODEL_DIR
python -m nerve --compile-model MODEL_DIR
python -m nerve --run COMPILED_MODEL_DIR_OR_MANIFEST
```

### [`runtime-rs/`](runtime-rs/)

Rust runtime crate.

| Path | Purpose |
| --- | --- |
| `runtime-rs/src/bin/nerve_runtime.rs` | CLI runtime binary entrypoint. |
| `runtime-rs/src/bin/nerve_runtime/` | Prompt/chat execution, package inspection, runtime graph controls, placement flags, sampler options, device binding, chat templates, and reporting. |
| `runtime-rs/src/bin/nerve_tui.rs` | TUI binary entrypoint. |
| `runtime-rs/src/tui/` | Terminal UI application modules. |
| `runtime-rs/src/stream_circuit/` | Core graph and artifact model: components, ports, state ports, lowered graphs, runtime graph topology, runtime node instances, placement, routes, reports, and validation. |
| `runtime-rs/src/editor/` | Runtime graph editor schema and editor state used by UI-facing code. |
| `runtime-rs/src/stream_runtime.rs` | Backend-neutral stream scheduler. |
| `runtime-rs/src/stream_runtime_tests.rs` | Stream scheduler tests. |
| `runtime-rs/src/stream_state.rs` | Backend-neutral transient state arena and per-stream state tables. |
| `runtime-rs/src/stream_prefix_cache.rs` | Backend-neutral prefix/state reuse primitives: prefix keys, retained cache entries, longest-compatible-prefix lookup, block-aligned insertion, restore, ref counts, and eviction. |
| `runtime-rs/src/vulkan_compute/` | Vulkan device discovery, feature/capability handling, resident buffers, pipeline creation, dispatch, sequence submission, and buffer copies. |
| `runtime-rs/src/vulkan_stream_circuit/` | Vulkan resident package loading, placement, device slices, resident plan buffers, dispatch binding, prompt streams, placed prompt engine, token runtime, sampler, speculative decode, batching, distributed execution, and reusable kernel/sequence machinery. |
| `runtime-rs/shaders/` | GLSL compute shader templates and generated/compiled shader inputs for BF16, FP8, INT4, attention/state, recurrent/conv, sampler, MoE, and related runtime operations. |

### [`tests/`](tests/)

Python compiler/package tests.

## Concept to implementation

The table below maps the language in [`CONCEPT.md`](CONCEPT.md) to the implementation that currently carries it.

| Concept term | Implementation |
| --- | --- |
| Compiled model | `compiled_models/<slug>/`, built by `nerve/model_compiler.py` |
| Package manifest | `vulkan_resident_package.json`, built by `nerve/model_package_manifest.py` |
| Component | `StreamCircuit` in `runtime-rs/src/stream_circuit/graph.rs` |
| Port | `CircuitPort` and `StatePort` in `stream_circuit/graph.rs` |
| Node instance | `StreamCircuitNodeInstance` in `stream_circuit/runtime_graph/instances.rs` |
| Runtime graph | `StreamCircuitRuntimeGraph` in `stream_circuit/runtime_graph/graph.rs` |
| Placement | `stream_circuit/placement.rs` and runtime CLI placement flags |
| Edge / transport | `stream_circuit/runtime_routes.rs` and `vulkan_stream_circuit/edge_*.rs` |
| Stream | `RuntimeStreamScheduler` stream state plus placed prompt streams |
| Transient state | `stream_state.rs` and `stream_prefix_cache.rs` |
| Permanent circuits | Package tensors, resident buffers, SPIR-V shaders, component executions |
| Input transducer | Package manifest `input_transducer` and Vulkan resident package loader |
| Output transducer | Package manifest `output_transducer` and token output pipeline |
| Sampler | Package sampler spec/kernels and runtime sampler config |
| Device-owned loop | Placed prompt stream / feedback window execution in `vulkan_stream_circuit` |
| Runtime graph editor | `runtime-rs/src/editor/` and `runtime-rs/src/tui/` |
