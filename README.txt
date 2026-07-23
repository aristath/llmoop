NERVE
=====

Neural Execution & Rewiring Virtual Engine

NERVE is an experimental inference engine for running language models as
long-lived, editable execution graphs instead of one-shot request/response jobs.

The design is documented in CONCEPT.md. The short version is:

  - a model becomes a compiled, self-contained package;
  - the package exposes reusable components, parameters, ports, kernels,
    transducers, state declarations, and a canonical topology;
  - runtime creates a concrete execution graph from that package;
  - a stream is the primary runtime object;
  - user prompts are events injected into an existing stream;
  - transient state belongs to the stream and is part of the running circuit;
  - placement and wiring are runtime decisions, not compiler decisions.

The original intuition was a guitar pedalboard with feedback, but the project
now uses standard graph/compiler/runtime language: components, node instances,
ports, edges, runtime graphs, placement, transports, streams, and transient
state.


Current status
--------------

NERVE is not a production inference engine yet. It is a real implementation of
the architecture, but still under active construction.

Currently implemented:

  - Safetensors source-model discovery and compilation.
  - Self-contained compiled model directories under compiled_models/.
  - A package manifest named vulkan_resident_package.json.
  - Transpiled model graph artifacts under transpiled/.
  - Lowered stream-circuit artifacts under lowered/.
  - Packaged tensors, tokenizer files, SPIR-V shaders, runtime config, and
    artifact integrity metadata inside the compiled model directory.
  - Rust runtime graph types with node instances, edges, placement, duplicate
    nodes, explicit chains, and state policies.
  - Runtime device discovery and logical-to-physical device binding.
  - Vulkan/SPIR-V resident package mounting.
  - Interactive chat and one-shot prompt execution through nerve-runtime.
  - A TUI entrypoint through nerve-tui.
  - A stream scheduler with persistent streams, queued input events, chunked
    prefill, decode feedback windows, activation batching, timing counters, and
    normal chat performance reporting.
  - Backend-neutral transient state arenas and per-stream state tables.
  - Ref-counted transient state blocks with reset, fork, share, snapshot, and
    prefix-cache primitives.
  - Route-native MoE compiler structures and shader families for selected
    expert routes.
  - MTP/speculative decoding package structures and runtime flow.
  - Shape-aware dispatch vocabulary for prefill versus decode.

Important unfinished work:

  - The Vulkan backend still has transitional fixed/circular resident state
    buffers in places. The scheduler has page/block-managed transient-state
    semantics, but resident Vulkan bindings still need to become truly
    page-backed before automatic prefix reuse can be wired into normal prompt
    admission.
  - Multi-stream activation batches are scheduled, but some placed Vulkan batch
    paths still execute activations sequentially internally.
  - FP8, INT4, MoE, MTP, prefill, and decode paths all need more real-model
    benchmarking and kernel work.
  - The TUI exists as a runtime surface, but the full graph-editing product
    experience is still being built.


Repository map
--------------

CONCEPT.md
  The architectural source of truth. Start here to understand the model:
  streams, compiled packages, components, runtime graphs, placement, transient
  state, feedback, and Vulkan/SPIR-V as the baseline backend.

TODO.md
  The active engineering backlog and current implementation status. This tracks
  scheduler work, block-managed state, batching, dispatch shape, MoE, MTP,
  prefill, prefix/state reuse, and graph/kernel reuse.

nerve/
  Python compiler and CLI package.

  nerve/cli.py
    User-facing command dispatcher. Provides:

      python -m nerve --discover-model MODEL_DIR
      python -m nerve --compile-model MODEL_DIR
      python -m nerve --run COMPILED_MODEL_DIR_OR_MANIFEST

  nerve/model_compiler.py
    High-level source discovery, staged compilation, atomic publish into a
    self-contained compiled model directory.

  nerve/model_package.py
    End-to-end package build: transpile, lower, copy tokenizer/tensors, compile
    shaders, build manifest, validate package.

  nerve/model_transpiler*.py
    Source checkpoint discovery and conversion into model/circuit graph facts.

  nerve/circuit_*.py
    Stream-circuit IR, lowering system, lowering operators, and optimization.

  nerve/model_package_manifest.py
    Builds vulkan_resident_package.json.

  nerve/model_package_tensors.py
  nerve/model_package_shaders.py
  nerve/model_package_shader_selection.py
  nerve/model_package_shader_templates.py
  nerve/model_package_shader_compiler.py
    Tensor packaging, shader selection, shader template rendering, and SPIR-V
    artifact creation.

runtime-rs/
  Rust runtime crate.

  runtime-rs/src/bin/nerve_runtime.rs
  runtime-rs/src/bin/nerve_runtime/
    The CLI runtime binary. Handles prompt/chat execution, package inspection,
    runtime graph controls, placement flags, sampler options, device binding,
    chat templates, and reporting.

  runtime-rs/src/bin/nerve_tui.rs
  runtime-rs/src/tui/
    Terminal UI entrypoint and application modules.

  runtime-rs/src/stream_circuit/
    Core graph and artifact model: components, ports, state ports, lowered
    graphs, runtime graph topology, runtime node instances, placement, routes,
    reports, and validation.

  runtime-rs/src/editor/
    Runtime graph editor schema and editor state used by UI-facing code.

  runtime-rs/src/stream_runtime.rs
  runtime-rs/src/stream_runtime_tests.rs
    Backend-neutral stream scheduler. Streams persist across events. The
    scheduler emits prefill/decode activations and compatible activation batches.

  runtime-rs/src/stream_state.rs
    Backend-neutral transient state arena and per-stream state tables.

  runtime-rs/src/stream_prefix_cache.rs
    Backend-neutral prefix/state reuse primitives: prefix keys, retained cache
    entries, longest-compatible-prefix lookup, block-aligned insertion, restore,
    ref counts, and eviction.

  runtime-rs/src/vulkan_compute/
    Vulkan device discovery, feature/capability handling, resident buffers,
    pipeline creation, dispatch, sequence submission, and buffer copies.

  runtime-rs/src/vulkan_stream_circuit/
    Vulkan resident package loading, placement, device slices, resident plan
    buffers, dispatch binding, prompt streams, placed prompt engine, token
    runtime, sampler, speculative decode, batching, distributed execution, and
    reusable kernel/sequence machinery.

  runtime-rs/shaders/
    GLSL compute shader templates and generated/compiled shader inputs for
    BF16, FP8, INT4, attention/state, recurrent/conv, sampler, MoE, and related
    runtime operations.

tests/
  Python compiler/package tests.


Concept to implementation
-------------------------

The table below maps the language in CONCEPT.md to the implementation that
currently carries it.

  Concept term             Implementation
  ------------             --------------
  compiled model           compiled_models/<slug>/, built by nerve/model_compiler.py
  package manifest         vulkan_resident_package.json, built by nerve/model_package_manifest.py
  component                StreamCircuit in runtime-rs/src/stream_circuit/graph.rs
  port                     CircuitPort and StatePort in stream_circuit/graph.rs
  node instance            StreamCircuitNodeInstance in stream_circuit/runtime_graph/instances.rs
  runtime graph            StreamCircuitRuntimeGraph in stream_circuit/runtime_graph/graph.rs
  placement                stream_circuit/placement.rs and runtime CLI placement flags
  edge / transport         stream_circuit/runtime_routes.rs and vulkan_stream_circuit/edge_*.rs
  stream                   RuntimeStreamScheduler stream state plus placed prompt streams
  transient state          stream_state.rs and stream_prefix_cache.rs
  permanent circuits       package tensors, resident buffers, SPIR-V shaders, component executions
  input transducer         package manifest input_transducer and Vulkan resident package loader
  output transducer        package manifest output_transducer and token output pipeline
  sampler                  package sampler spec/kernels and runtime sampler config
  device-owned loop        placed prompt stream / feedback window execution in vulkan_stream_circuit
  runtime graph editor     runtime-rs/src/editor/ and runtime-rs/src/tui/


Compiled model layout
---------------------

By default:

  python -m nerve --compile-model /path/to/source/model

creates:

  compiled_models/<model_slug>/
    vulkan_resident_package.json
    config.json
    runtime_config.json
    tensors.json
    tokenizer/
    tensors/
    shaders/
    transpiled/
      model.json
      tensors.json
    lowered/
      execution_graph.circuits.json
      ...

The compiled model directory is intentionally self-contained. The runtime should
be able to load the compiled model as one artifact root rather than chasing
files scattered across unrelated folders.

The compiler publishes through a staged directory and then atomically swaps the
finished compiled model into place. See nerve/model_compiler.py.


Quick start
-----------

From the repository root:

  python -m nerve --discover-model /path/to/safetensors/model

Compile a model:

  python -m nerve --compile-model /path/to/safetensors/model

Compile to a specific package directory:

  python -m nerve --compile-model /path/to/safetensors/model \
    --compiled-model-dir compiled_models/my_model

Inspect a compiled package:

  python -m nerve --run compiled_models/my_model --inspect-package

Inspect the effective runtime graph:

  python -m nerve --run compiled_models/my_model --inspect-graph

Inspect runtime placement/device facts:

  python -m nerve --run compiled_models/my_model --inspect-runtime

Run a one-shot prompt:

  python -m nerve --run compiled_models/my_model \
    --prompt "What is the capital of Greece?"

Start interactive chat:

  python -m nerve --run compiled_models/my_model --chat

Open the TUI:

  python -m nerve


Runtime graph controls
----------------------

Runtime placement and wiring are supplied when the model is run. They are not
compiled into the model package.

Use a default logical device:

  python -m nerve --run compiled_models/my_model \
    --device gpu0 \
    --prompt "Hello"

Bind logical devices to physical targets:

  python -m nerve --run compiled_models/my_model \
    --device gpu0 \
    --bind-device gpu0=vulkan:0 \
    --prompt "Hello"

Place a specific node instance:

  python -m nerve --run compiled_models/my_model \
    --place-node layer_00=gpu0 \
    --place-node layer_01=gpu1 \
    --bind-device gpu0=vulkan:0 \
    --bind-device gpu1=vulkan:1 \
    --prompt "Hello"

Duplicate a node instance after an existing instance:

  python -m nerve --run compiled_models/my_model \
    --duplicate-after layer_05=layer_05_copy \
    --prompt "Hello"

Run an explicit source/component chain:

  python -m nerve --run compiled_models/my_model \
    --chain layer_00,layer_01,layer_02,layer_02_copy=layer_02,layer_03 \
    --prompt "Hello"

These controls correspond to CONCEPT.md's runtime graph idea: the compiled
model supplies reusable source components; runtime chooses the actual node
instances, placement, and wiring.


Generation and sampler controls
-------------------------------

Important runtime options:

  --max-new-tokens N
    Generation stop budget. Defaults to 65536. This is not context capacity.

  --context-size N
    Runtime transient-state context window. If omitted, runtime chooses an
    automatic size.

  --speculative-draft-tokens N
    Number of MTP draft tokens per verification cycle. Defaults to 0.

  --seed N
    Explicit sampler seed. Defaults to 0.

  --temperature VALUE
  --top-k N
  --top-p VALUE
  --min-p VALUE
  --presence-penalty VALUE
  --repetition-penalty VALUE
    Runtime sampler overrides.

  --chat-template-var NAME=JSON
    Pass model-owned chat template variables, for example reasoning flags, to
    interactive chat. May be repeated.

  --generated-only
    Print only newly generated text.

  --json
    Print a machine-readable report for non-chat runs and inspection modes.


Development notes
-----------------

Run Python tests sequentially:

  python -m pytest -p no:xdist tests/test_cli.py -q

Run Rust tests sequentially:

  cargo test --manifest-path runtime-rs/Cargo.toml TEST_NAME -- --test-threads=1 --exact

Run Rust compile checks:

  cargo check --manifest-path runtime-rs/Cargo.toml

  cargo check --manifest-path runtime-rs/Cargo.toml --features vulkan,tokenizers,tui

This repository intentionally avoids broad parallel test runs. Some runtime
paths can initialize Vulkan or hold GPU resources, and reliability matters more
than shaving seconds off validation.


Design principles
-----------------

NERVE should stay clean and lean.

  - Core runtime code must not hardcode facts about one model family.
  - The compiler should discover structure and compile a package from that
    structure.
  - Model-specific facts belong in the compiled model package, not in the
    engine.
  - Placement is runtime configuration, not a compiler artifact.
  - Backwards compatibility with old internal package shapes is not a goal.
  - Removing or duplicating components should be a graph operation.
  - State sharing, cloning, and forking must be explicit.
  - KV is transient stream state, not disposable bookkeeping.
  - No performance solution should depend on arbitrary tiny token limits.
  - llama.cpp and vLLM are valuable references, but NERVE is not trying to be a
    clone of either.


Where to look next
------------------

For architecture:

  CONCEPT.md

For the current implementation backlog:

  TODO.md

For package compilation:

  nerve/model_compiler.py
  nerve/model_package.py
  nerve/model_package_manifest.py

For runtime graph semantics:

  runtime-rs/src/stream_circuit/
  runtime-rs/src/editor/

For stream scheduling and transient state:

  runtime-rs/src/stream_runtime.rs
  runtime-rs/src/stream_state.rs
  runtime-rs/src/stream_prefix_cache.rs

For Vulkan execution:

  runtime-rs/src/vulkan_compute/
  runtime-rs/src/vulkan_stream_circuit/

For CLI runtime behavior:

  nerve/cli.py
  runtime-rs/src/bin/nerve_runtime/

For TUI work:

  TUI.md
  runtime-rs/src/tui/
