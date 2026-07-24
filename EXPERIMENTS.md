# GPU and LLM Computational Design Space

## Purpose

This document is a map of the computational design space around GPU inference.
It is not a shortlist of experiments, a ranking of techniques, or a claim that
the listed alternatives are the only useful ones.

It separates four questions that are often collapsed into one:

1. What processes is a GPU built to perform efficiently?
2. What functional work does a language model perform?
3. How is that work conventionally expressed as numerical operations?
4. What other representations could express the same component or behavior?

The first three inventories provide raw material for the fourth. An alternative
expression may reuse one GPU process, compose several processes, introduce a
different state representation, or remove an operation that exists only because
of the conventional tensor formulation.

NERVE treats an existing model as a behavioral reference:

```text
(public_output, next_state, control) =
    source_model(external_input, current_state, randomness, control)
```

A compiled circuit may implement the same externally relevant transition using
different parameters, state, topology, operations, numerical representations,
and hardware pipelines:

```text
(compiled_output, compiled_next_state, compiled_control) =
    compiled_circuit(external_input, compiled_state, randomness, control)
```

The source implementation remains an executable specification. The compiled
circuit is not required to preserve its matrix multiplications, attention heads,
MLPs, KV layout, layer internals, or intermediate tensors. This follows the
behavioral-compilation contract in [CONCEPT.md](CONCEPT.md).

## Document Map

- **Part I** inventories programmable and fixed-function GPU processes.
- **Part II** describes LLM responsibilities without assuming a transformer
  implementation.
- **Part III** records the conventional tensor expressions used by transformer
  and state-space language models.
- **Part IV** develops an open vocabulary of alternative representations and
  expressions.
- **Part V** maps LLM responsibilities to conventional and alternative GPU
  processes in both directions.
- **Part VI** preserves worked expressions from the initial experiment document
  as examples rather than boundaries.
- **Part VII** contains questions for generating further expressions.
- **Part VIII** defines how any selected expression becomes a measured
  experiment.
- **Part IX** records the model-anatomy, pedal-boundary, lowering, and modular
  compilation design space.

## Vocabulary

This document uses the following distinctions:

- **Responsibility:** What a model component accomplishes, such as addressing
  context, transforming features, updating state, or selecting an output.
- **Expression:** The mathematical or symbolic form used to describe that
  responsibility, such as a dense linear map, lookup table, decision tree,
  recurrence, sampled field, or graph traversal.
- **Representation:** How signals, parameters, and state are encoded, such as
  dense tensors, bitplanes, geometry, textures, sparse coordinates, or programs.
- **Realization:** The concrete implementation of an expression on a device.
- **GPU process:** A kind of work the GPU exposes through programmable or
  fixed-function machinery.
- **Behavioral compilation:** Synthesis of another circuit from the source
  model's reachable behavior and error contract.
- **Experiment:** A measured realization of one point in the design space.

An expression is not tied to one realization. A lookup expression could use a
buffer, a sampled image, a tree, a perfect hash, or generated code. Conversely,
one GPU process such as texture sampling could participate in embeddings,
routing, nonlinear transformations, memory lookup, or state evolution.

# Part I: Processes a GPU Is Optimized to Perform

## 1. Throughput-Parallel Program Execution

### Workgroups, wavefronts, and subgroups

GPU programs create many invocations of the same shader. Invocations are
organized into workgroups, and hardware executes smaller groups of lanes
together. Vulkan calls the efficiently communicating lane group a **subgroup**;
AMD hardware commonly calls its corresponding execution group a **wavefront**.

This organization is suited to:

- applying the same transformation to many independent elements;
- processing tiles of a larger problem;
- dividing one reduction across lanes;
- evaluating several candidates together;
- compacting or classifying many items;
- comparing neighboring or corresponding elements;
- exchanging values among nearby lanes; and
- hiding memory latency by switching among ready wavefronts.

The organization also shapes the cost of a program:

- lanes in one wavefront share instruction issue;
- divergent branches may require different paths to execute separately;
- workgroup-local synchronization is cheaper than device-wide synchronization;
- subgroup shuffles and votes can exchange information without round trips to
  device memory; and
- occupancy depends on registers, local memory, workgroup size, and the
  instruction mix.

### Vector arithmetic

Vector arithmetic lanes execute ordinary numerical and logical instructions
across many lanes:

- floating-point add, subtract, multiply, fused multiply-add, compare, and
  conversion;
- integer arithmetic;
- shifts, masks, Boolean operations, bit counting, and bitfield extraction;
- min, max, clamp, and selection;
- reciprocal, reciprocal square root, exponent, logarithm, and other supported
  special functions;
- address calculations;
- branch predicates; and
- packing and unpacking numerical formats.

This machinery is not restricted to matrices. It can implement filters, finite
state machines, prefix scans, hashes, graph algorithms, cellular updates,
polynomials, numerical integration, sorting networks, codecs, interpreters, and
arbitrary shader programs within the execution and memory model.

### Scalar arithmetic and control

AMD RDNA compute units distinguish scalar work shared by a wavefront from vector
work whose values differ by lane. Scalar machinery can handle:

- uniform addresses and loop counters;
- common constants and parameters;
- wavefront-wide control decisions;
- descriptor and resource selection;
- uniform branches; and
- scalar memory reads.

An alternative inference representation may expose more wave-uniform decisions
than a dense tensor implementation, or it may create enough per-lane divergence
that scalarization is no longer available. The representation determines which
case applies.

### Matrix and dot-product operations

Modern GPUs include instructions or accelerators for matrix fragments and packed
dot products. They are optimized for regular multiply-accumulate patterns over
supported shapes and data types.

They are one GPU process among several. They can still participate in an
alternative circuit where a small projection, local expert, correction path, or
structured factor is naturally expressed as a matrix operation. Rejecting dense
matrix multiplication as the universal representation does not require
forbidding every matrix or dot-product instruction.

### Reductions, scans, votes, and lane exchange

Subgroup and workgroup operations support collective processes such as:

- sum, product, minimum, and maximum;
- inclusive and exclusive scans;
- ballot masks;
- all, any, and equality votes;
- lane shuffles and broadcasts;
- election of one lane;
- prefix allocation into a shared output; and
- cooperative loading and reuse of tiles.

These processes can express normalization, selection, routing, sparse
compaction, histogramming, state aggregation, beam management, and work
generation without treating the complete task as a matrix multiplication.

### Atomics and unordered contribution

Atomic operations let many invocations update shared locations without a
separate deterministic reduction tree. Depending on data type and device
features, operations can include:

- add;
- min and max;
- exchange;
- compare-and-swap;
- bitwise operations; and
- counters.

Atomics suit sparse scatter, graph propagation, histograms, queues, reference
counts, active-set construction, and accumulations whose ordering is either
irrelevant or explicitly accounted for.

## 2. Memory Access and Data Movement

### Registers

Registers hold invocation-local and wavefront-local working values at the
smallest directly addressable scope. They are suited to:

- temporary arithmetic state;
- small recurrent state;
- accumulators;
- coordinates and predicates;
- current candidate sets; and
- values reused across several instructions.

Register use competes with the number of simultaneously resident wavefronts.
An expression with less parameter traffic but much larger live state may trade
bandwidth for lower occupancy.

### Local data share and workgroup memory

Workgroup-shared memory provides an explicitly managed on-chip exchange area.
It supports:

- tiled reuse;
- collaborative reductions;
- local queues;
- shared dictionaries or codebook tiles;
- neighborhood exchange;
- software-managed caches; and
- staged transformations.

Its capacity and synchronization scope favor bounded local working sets rather
than model-wide state.

### Cache hierarchy

GPU caches exploit temporal, spatial, instruction, scalar, and texture locality.
The exact hierarchy is device-specific, but the processes it rewards include:

- repeated access to the same parameter or state region;
- neighboring accesses that occupy the same cache line;
- compact code and parameter working sets;
- shared read-only data across many lanes;
- reuse of recently selected experts or field tiles; and
- producer-consumer locality within a fused or resident circuit.

A representation can create locality rather than merely accept the locality of
the source tensors. Examples include clustering related parameters, storing
reachable cases together, arranging a graph by traversal order, and keeping
active state in a compact resident form.

### Coalesced buffer access

Buffer memory paths are suited to lanes reading or writing adjacent, aligned
locations. They can also perform gathers and scatters, but unrelated addresses
may require more memory transactions and provide less cache reuse.

Representations that expose sorted indices, compact active sets, tiled layouts,
or common base addresses can convert irregular logical access into regular
physical access.

### Texture addressing and sampling

Sampled-image machinery combines several processes:

- conversion from normalized or unnormalized coordinates to texel addresses;
- address modes such as clamp, wrap, and mirror;
- selection among image layers and mip levels;
- nearest lookup;
- linear interpolation between two, four, or eight neighboring texels for 1D,
  2D, and 3D images;
- gathering neighboring components;
- format conversion and component swizzle;
- depth comparison;
- filtered access to multiple resolution levels; and
- texture-oriented caching.

Depending on supported features and formats, sampling may also expose min/max
reduction, cubic filtering, sparse residency, or other specialized image
operations.

This is a hardware vocabulary for indexed lookup, interpolation, multiresolution
representation, neighborhood access, and approximate field evaluation. It is not
limited to visual color data, although the API's image formats and filtering
rules constrain what numerical representations it can consume.

### Image and buffer transfer

Transfer commands and copy-capable queues can:

- copy buffers and images;
- convert between some image arrangements;
- clear or fill resources;
- resolve multisampled images;
- blit and filter images where supported; and
- move resources between memory domains or devices.

Transfers can overlap execution when queue, dependency, and hardware conditions
permit. This provides a process for prefetch, migration, staging, snapshotting,
and construction of the next working set.

### Compression and packed storage

GPUs commonly consume packed vertex, texture, depth, color, and media formats.
Shader code can also operate on packed integers and bitplanes directly.

Potential processes include:

- fixed-function image-format conversion;
- block-compressed image fetch;
- shader-level quantized decoding;
- bit-sliced computation without scalar expansion;
- delta reconstruction;
- dictionary lookup; and
- codec-based reconstruction through a media engine.

Each path has a different contract. A format supported for sampling may not be
supported for storage writes or blending, and a media decoder emits constrained
surface layouts rather than arbitrary buffers.

## 3. Graphics Pipeline Processes

The graphics pipeline is a programmable and fixed-function dataflow processor.
Its inputs do not have to represent a visible scene, but they must obey graphics
pipeline contracts.

### Vertex and mesh processing

Programmable stages can:

- transform input records;
- generate coordinates and attributes;
- select or cull records;
- construct primitives;
- emit variable amounts of geometry through supported mesh/task mechanisms; and
- attach data that later stages interpolate.

These stages can express point generation, sparse record expansion, region
construction, routing geometry, or a graph of local domains.

### Primitive assembly, clipping, culling, and coverage

Fixed-function stages assemble vertices into points, lines, or triangles and
determine:

- whether a primitive is visible under configured rules;
- how it intersects the clip volume;
- which screen-space samples it covers; and
- which fragments are generated.

Abstractly, these are parallel classification, range rejection, projection, and
domain-coverage processes.

### Rasterization and interpolation

Rasterization converts primitives into covered sample locations. Fragment inputs
can be interpolated from per-vertex attributes using:

- perspective-correct interpolation;
- linear interpolation without perspective correction;
- flat, non-interpolated values;
- centroid or sample positions; and
- explicit barycentric coordinates where supported.

This provides a fixed pipeline for evaluating piecewise affine functions over
triangulated domains, expanding compact primitives into many covered samples,
and carrying local coefficients into fragment programs.

### Depth and stencil tests

Depth and stencil machinery performs comparisons, masks, and conditional
updates, often before or after fragment execution according to pipeline rules.
It can express:

- nearest or farthest selection;
- threshold acceptance;
- integer masks and state transitions;
- visibility;
- rejection before expensive work; and
- bounded per-location counters or classifications.

The available comparisons and updates are constrained by depth/stencil formats
and pipeline semantics.

### Render-target blending and raster operations

The output-merger stage can combine fragment outputs with existing attachment
values through supported operations such as:

- weighted addition;
- subtraction;
- minimum and maximum;
- component masks;
- logical operations on supported integer formats; and
- multisample resolve behavior.

This provides a fixed-function process for combining contributions at addressed
locations. The permitted formats, factors, ordering rules, and operations are
more restricted than arbitrary shader code.

## 4. Ray and Acceleration-Structure Processes

### Acceleration-structure construction

Vulkan acceleration structures organize:

- triangles;
- axis-aligned bounding boxes;
- instances of bottom-level structures; and
- transforms and instance metadata.

The API distinguishes construction from update. Updates can change specified
geometry, transform, and instance data under rules established when the
structure was built.

Abstractly, construction turns spatial records into a hierarchy that supports
pruned traversal.

### Ray traversal

Ray traversal searches an acceleration structure for intersections with a
directed interval. Hardware and drivers can accelerate:

- hierarchy traversal;
- ray-box testing;
- ray-triangle testing;
- first-hit or any-hit search;
- acceptance and rejection of candidates; and
- retrieval of instance, primitive, and intersection metadata.

Ray queries can be embedded in regular shader stages. Ray-tracing pipelines can
invoke programmable shaders at stages of the traversal.

The native domain is low-dimensional geometry. Using it for a language-model
responsibility requires an encoding from the model's signal or state into rays,
volumes, triangles, instances, or transforms.

## 5. Scheduling and Autonomous Work Generation

### Command buffers and queues

GPUs consume recorded commands describing:

- compute dispatches;
- graphics draws;
- ray-tracing dispatches;
- transfers;
- resource transitions;
- synchronization; and
- indirect commands.

Independent queues may overlap work where the device and dependency graph allow
it. The command processor therefore participates in batching, pipelining,
prefetch, and concurrent execution even though it does not implement the learned
function itself.

### Indirect execution

Indirect commands read dimensions or draw descriptions from device-visible
memory. A shader can therefore determine the amount or shape of later work
without returning that decision to the host.

This supports:

- compacted active sets;
- device-selected expert counts;
- zero-work termination;
- variable candidate counts;
- adaptive iteration;
- producer-generated consumers; and
- device-owned feedback windows.

### Execution graphs and shader-enqueued work

Where supported, execution-graph mechanisms allow shader nodes to enqueue other
nodes with payloads. This expresses dynamic GPU-side dataflow more directly than
a fixed list of host-recorded dispatches.

The same logical process can also be approximated with persistent shaders,
device queues, indirect dispatch, or bounded command templates when execution
graphs are unavailable.

### Persistent and resident programs

A shader can remain active across many state transitions, or a runtime can keep
pipelines, resources, and command templates resident across bounded
activations. This supports:

- device-resident state;
- event queues;
- feedback;
- local scheduling;
- reduced host round trips;
- continuous producers and consumers; and
- immediate reuse of code and working data.

Execution duration, fairness, cancellation, watchdog behavior, and state commit
boundaries remain part of the realization.

## 6. Media and Display Processes

### Media encode and decode

Dedicated media engines implement constrained algorithms for supported video
formats. On the Radeon AI PRO R9700, AMD lists H.264, H.265/HEVC, and AV1 encode
and decode support.

These engines perform processes including combinations of:

- block prediction;
- transform and inverse transform;
- quantization and reconstruction;
- entropy coding and decoding;
- motion-compensated reconstruction;
- color-surface handling; and
- frame-reference management.

They do not expose these internal steps as arbitrary programmable inference
instructions. An alternative representation would have to encode parameters or
state into a supported media bitstream and consume the reconstructed surfaces.

### Display

Display engines scan surfaces, compose supported planes, perform timing and
format operations, and drive physical outputs. They are generally not an
arbitrary compute path. They can still matter to an end-to-end system by
offloading presentation and format conversion, but using display scanout as a
learned inference component requires an API-visible transformation that
preserves retrievable numerical output.

## 7. A Concrete RDNA 4 Inventory

The Radeon AI PRO R9700 is one concrete target rather than the definition of the
architecture. AMD publishes the following device-level resources:

| Resource | Published quantity or property |
| --- | --- |
| Compute units | 64 |
| Stream processors | 4096 |
| AI accelerators | 128 |
| Ray accelerators | 64 |
| ROPs | 128 |
| Dedicated memory | 32 GB GDDR6 |
| Peak memory bandwidth | 640 GB/s |
| Infinity Cache | 64 MB |
| Supported media formats | H.264, H.265/HEVC, and AV1 encode/decode |

Published peak rates describe particular operation shapes and data types. They
do not imply that all units can be independently saturated at once, that every
unit is exposed through every API, or that a workload automatically maps to the
corresponding peak process.

## 8. Processes That Conflict with GPU Organization

The inverse inventory is also useful. Conventional GPUs do not naturally reward:

- short serial tasks with little parallel work;
- frequent host-device synchronization;
- unpredictable pointer chasing with no locality;
- heavy branch divergence within wavefronts;
- global communication after every small operation;
- working sets that are repeatedly streamed without reuse;
- tiny dispatches whose launch and synchronization dominate execution;
- state layouts that require continuous reformatting; or
- algorithms whose useful work is much smaller than their routing and
  bookkeeping overhead.

An alternative LLM expression need not eliminate all of these properties. It
must account for them when mapping its abstract operations to the device.

# Part II: What a Language Model Does

## 9. External Behavioral View

At its public boundary, an autoregressive language model assigns a conditional
distribution to possible continuations:

```text
P(next_symbol | preceding_symbols, current_input, control)
```

In a stateful stream formulation, the same behavior can be written:

```text
(distribution_t, S_{t+1}, control_t) =
    M(event_t, S_t, randomness_t, control_input_t)
```

The state `S_t` may be explicit KV tensors, recurrent state, an external memory,
or any behaviorally equivalent representation. The distribution may be exposed
directly, reduced to candidates, or sampled into a symbol that feeds the next
activation.

This boundary says what the model does without saying how it does it.

## 10. Functional Responsibilities Inside a Language Model

The following responsibilities appear in transformer LLMs and in alternative
sequence models, even when their internal expressions differ.

### Symbol transduction

Convert discrete token identifiers, bytes, patches, audio units, control events,
or other external symbols into an internal signal.

The inverse responsibility converts an internal signal into scores or
probabilities over public symbols.

### Feature representation

Maintain a signal in which distinctions relevant to future behavior can be
represented. Individual coordinates do not need stable human-readable meanings;
the representation only needs to support the transformations and state
transitions used by the model.

### Order, position, and time

Distinguish the order and relative placement of events. This may be represented
through explicit position features, phase, delays, recurrence, convolution,
graph structure, or state-transition order.

### Contextual addressing

Determine which prior information is relevant to the current signal. This
includes:

- content-based lookup;
- position-based lookup;
- local and global context;
- retrieval of one or several memories;
- weighting retrieved information; and
- deciding that stored information is irrelevant.

### Memory write and retention

Convert current activity into state that can affect future behavior. This
includes decisions about:

- what to store;
- where to store it;
- how long to retain it;
- how to update or overwrite it;
- how to separate fast and slow state; and
- how to expose it to later components.

### Feature transformation

Map the current signal into another signal that exposes different distinctions,
associations, or predictions. In conventional transformers, much of the learned
knowledge is encoded in the parameters of dense or sparse linear projections.
The responsibility itself is broader than that encoding.

### Nonlinear gating and selection

Make the effect of one signal depend on another signal or on its magnitude. This
supports conditional computation, inhibition, amplification, thresholding,
feature conjunction, and piecewise behavior.

### Routing

Select:

- experts;
- memories;
- graph branches;
- parameter subsets;
- output candidates;
- computation depth; or
- state destinations.

Routing may be soft, hard, stochastic, deterministic, local, hierarchical, or
distributed.

### Mixing and residual transport

Combine signals from different paths while preserving or modifying earlier
information. Addition is the conventional residual operation, but the
responsibility includes weighted mixture, concatenation, selection, competition,
and other compositions.

### Dynamic-range control

Keep signals within ranges that allow subsequent components to distinguish
meaningful differences. Normalization is one expression of this responsibility.
Other expressions can constrain, rescale, quantize, clip, rank, or encode
confidence differently.

### State transition

Compute the next internal condition of the running process. Attention with a KV
cache implements a state transition even though the cache is often described as
saved intermediates. Recurrent, state-space, cellular, graph, and memory-based
models make the state-transition role more explicit.

### Candidate scoring

Assign relative preference to possible outputs, routes, memories, or actions.
Scores may be exact real numbers, quantized ranks, partial orders, hierarchical
decisions, or candidate sets with a later refinement stage.

### Selection and randomness

Turn candidate scores into one or more choices according to temperature,
constraints, penalties, top-k, top-p, greedy selection, beam rules, or another
policy. Randomness is an explicit input to this responsibility.

### Feedback and stopping

Feed selected output or a richer private signal into the next activation,
respond to interruption, and determine whether the stream should continue,
yield, wait, or stop.

## 11. Responsibilities Are Not Transformer Layers

One transformer operation may serve several responsibilities:

- attention both addresses memory and mixes retrieved values;
- an MLP both transforms features and creates nonlinear gating;
- RMSNorm controls scale and changes the coordinates consumed by the next
  learned projection;
- a residual edge transports information and changes the effective function of
  the component it bypasses;
- a MoE router scores, selects, and schedules parameter subsets; and
- the output projection both decodes features and scores candidates.

Likewise, one responsibility can span several layers. Long-term memory behavior
emerges from repeated state writes, attention reads, feature transformations,
and feedback rather than belonging to a single named tensor operation.

A behavioral compiler can therefore choose replacement boundaries at an
operation, subcomponent, component, group of components, complete transient
state, or whole-model level.

# Part III: Traditional Numerical Expressions

## 12. Conventional Signal and Parameter Representation

A decoder-only transformer commonly represents:

- one token activation as a dense vector `x` of hidden width `d`;
- a batch or sequence as a dense matrix `X`;
- learned transforms as dense or block-sparse matrices;
- attention state as append-only key and value tensors;
- expert parameters as separate dense matrices;
- probabilities and routing weights as dense score vectors; and
- component boundaries as tensor shapes and layouts.

The same trained function may be stored in BF16, FP16, FP8, INT8, INT4, or
another quantized format. Quantization changes storage and arithmetic details but
usually preserves the dense tensor expression.

## 13. Token Embedding

### Tokenization boundary

Before embedding, a conventional text pipeline converts bytes or characters
into token identifiers using a tokenizer such as byte-pair encoding, Unigram,
WordPiece, or a byte-level scheme. Tokenization is commonly implemented outside
the neural model using CPU string processing, tries, tables, and merge rules.
The public language-model behavior nevertheless depends on this transducer
because it defines the symbols seen and emitted by the model.

Traditional expression:

```text
x_0 = E[token_id]
```

`E` is a vocabulary-by-hidden-width table. Runtime work is a row gather, optional
scaling, and sometimes addition or composition with other embeddings.

Responsibility:

- symbol transduction;
- initial feature representation; and
- possibly shared parameterization with the output projection.

## 14. Normalization

RMSNorm is commonly expressed as:

```text
rms(x) = sqrt(mean(x_i^2) + epsilon)
y_i    = weight_i * x_i / rms(x)
```

The conventional GPU realization performs:

1. elementwise square;
2. reduction;
3. reciprocal square root;
4. elementwise scale; and
5. optional fusion with an adjacent operation.

LayerNorm additionally computes and removes a mean.

Responsibility:

- dynamic-range control;
- coordinate conditioning for the next component; and
- learned per-feature rescaling.

## 15. Learned Linear Projection

Traditional expression:

```text
y = W x + b
```

For many tokens:

```text
Y = X W^T + b
```

Decode commonly presents matrix-vector or narrow matrix-matrix shapes. Prefill
and training present wider matrix-matrix shapes. Dense projections are used for:

- query, key, and value generation;
- attention output;
- MLP gate, up, and down paths;
- MoE routing;
- recurrent or state-space parameter generation;
- output logits; and
- adapters and multimodal transducers.

Responsibility:

- learned feature transformation;
- scoring;
- basis change;
- expansion or contraction of signal width; and
- generation of parameters for another operation.

## 16. Position Encoding

RoPE conventionally groups query and key coordinates into pairs and rotates
each pair by a position-dependent angle:

```text
[x'_{2i}    ]   [ cos(theta_i) -sin(theta_i) ] [x_{2i}    ]
[x'_{2i + 1}] = [ sin(theta_i)  cos(theta_i) ] [x_{2i + 1}]
```

Traditional realization uses elementwise loads, trigonometric tables or derived
coefficients, multiply-adds, and pairwise rearrangement.

Responsibility:

- order and relative position;
- phase-like modulation of contextual addressing; and
- distinction among otherwise identical content at different positions.

## 17. Scaled Dot-Product Attention

The conventional expression is:

```text
Q = X W_Q
K = X W_K
V = X W_V

scores = Q K^T / sqrt(head_width)
weights = softmax(scores + mask)
output  = weights V
```

Grouped-query attention shares one key/value head among several query heads.
Autoregressive decode generates a new query and compares it with stored keys.

Responsibility:

- contextual addressing;
- memory read;
- relevance scoring;
- normalized competition among memories; and
- weighted mixing of retrieved values.

## 18. Softmax

Traditional expression:

```text
m   = max_i(score_i)
z_i = exp(score_i - m)
y_i = z_i / sum_j(z_j)
```

The stable realization includes maximum reduction, exponentiation, sum
reduction, and normalization. Online and tiled formulations can compute the same
result without materializing the complete score matrix.

Responsibility:

- turn unbounded scores into positive normalized weights;
- create competition; and
- preserve relative score differences under a common offset.

Softmax is one expression of those responsibilities, not the responsibility
itself.

## 19. KV State

Traditional autoregressive state update:

```text
K_state <- append(K_state, k_t)
V_state <- append(V_state, v_t)
```

Traditional read:

```text
context_t = softmax(q_t K_state^T) V_state
```

The state grows with the number of retained tokens unless a window, eviction,
compression, or other policy changes it.

Responsibility:

- memory write;
- retention of content-addressable state;
- contextual addressing; and
- future state-dependent behavior.

## 20. Residual Transport

Traditional expression:

```text
y = x + component(norm(x))
```

or:

```text
y = norm(x + component(x))
```

Responsibility:

- preserve and transport an earlier signal;
- accumulate a component's change;
- provide a direct path through deep compositions; and
- define the effective coordinate system seen by later components.

## 21. Dense Gated MLP

A common SwiGLU-like expression is:

```text
gate = W_gate x
up   = W_up x
mid  = silu(gate) * up
y    = W_down mid
```

This consists of learned projections, an elementwise nonlinear function,
elementwise multiplication, and another projection.

Responsibility:

- feature transformation;
- nonlinear gating;
- conditional amplification or inhibition; and
- parameterized feature storage.

## 22. Mixture of Experts

A conventional sparse MoE layer is expressed as:

```text
router_scores = router(x)
selected      = top_k(router_scores)
expert_y_i    = expert_i(x) for i in selected
y             = weighted_sum(expert_y_i)
```

Runtime also performs dispatch, grouping of tokens by expert, expert execution,
and restoration of output order.

Responsibility:

- parameter routing;
- conditional computation;
- local feature transformation; and
- mixture of selected results.

## 23. Convolutional, Recurrent, and State-Space Components

Not every current LLM component is attention-based.

A causal convolution is conventionally expressed as:

```text
y_t = sum_{i=0}^{kernel_width-1} kernel_i * x_{t-i}
```

A recurrence is expressed as:

```text
S_{t+1} = f(A_t S_t + B_t x_t)
y_t     = g(C_t S_t + D_t x_t)
```

A selective state-space component makes some transition parameters depend on
the current input and may use a parallel scan for prefill plus a recurrent step
for decode.

Responsibilities:

- temporal mixing;
- bounded or structured memory;
- state transition;
- filtering; and
- input-dependent retention or forgetting.

## 24. Output Projection and Candidate Distribution

Traditional expression:

```text
logits = W_vocab x
```

An optional bias, scale, or tied embedding relationship may be present. The
result contains one score per vocabulary item.

Responsibility:

- convert internal features into output-symbol compatibility;
- score candidates; and
- expose a distribution for selection.

## 25. Sampling

Conventional sampling may perform:

1. penalties or constraints on logits;
2. temperature scaling;
3. top-k selection;
4. top-p or minimum-probability filtering;
5. normalization;
6. pseudorandom selection; and
7. token-history update.

Greedy decoding replaces stochastic selection with an argmax.

Responsibility:

- choose a public and/or feedback output;
- incorporate explicit randomness and policy; and
- produce continuation or termination control.

## 26. Conventional GPU Realization

Traditional runtimes lower the preceding expressions into:

- GEMM, GEMV, and packed dot-product kernels;
- reductions;
- elementwise kernels;
- gathers and scatters;
- append or copy operations;
- sorting or top-k selection;
- quantize and dequantize steps;
- kernel fusion;
- batched execution;
- cross-device collective communication; and
- host- or device-scheduled generation loops.

Many runtime optimizations preserve the mathematical expression:

- tiling;
- fusion;
- quantization;
- layout transformation;
- FlashAttention-style IO-aware exact attention;
- batching;
- speculative decoding;
- kernel specialization;
- tensor, pipeline, or expert parallelism; and
- prefix-state reuse.

These can be combined with alternative expressions. They should not be confused
with changing the expression itself.

## 27. Prefill and Decode Are Different Execution Regimes

Prefill processes many prompt positions together. It exposes:

- wide parallelism;
- matrix-matrix operations;
- causal dependencies;
- opportunities for scans and tiled attention; and
- substantial reusable work within one activation.

Decode commonly processes one new position per stream. It exposes:

- repeated use of large permanent parameters;
- state reads that grow or change over time;
- narrow matrix shapes;
- a strict feedback dependency between selected tokens; and
- sensitivity to dispatch, synchronization, and parameter bandwidth.

An alternative expression may use different realizations, state layouts, or
even different compiled circuits for these regimes while preserving one public
stream behavior.

# Part IV: Alternative Ways to Express LLM Work

## 28. This Is an Expression Vocabulary, Not a Catalog

The following sections define axes and reusable forms from which circuits can be
constructed. They are not mutually exclusive alternatives and they do not bound
future designs.

A component can combine:

- a dense projection for a small coordinate transform;
- a hash or tree for routing;
- sampled fields for local behavior;
- bit-level state for control;
- a recurrent state for continuity;
- a sparse correction path;
- rendering operations for interpolation and mixing; and
- device-generated work for feedback.

The relevant question is not “Which one replaces the transformer?” It is:

```text
What expression implements this responsibility
over the reachable signal and state domain
with acceptable future observable behavior?
```

## 29. Representation Axes

### Signal representation

A signal could be represented as:

- a dense floating-point vector;
- a quantized dense vector;
- a sparse set of active coordinates;
- a ranked list;
- a probability distribution;
- binary or ternary bitplanes;
- a symbolic record;
- a codebook identifier;
- a product code;
- a graph or set of graph activations;
- points, rays, primitives, or volumes;
- a sampled 1D, 2D, or 3D field;
- coefficients in a spectral or wavelet basis;
- a collection of particles or hypotheses;
- phases and amplitudes;
- events and timestamps;
- a program state;
- a finite-state-machine state; or
- several simultaneous representations connected by explicit transducers.

### Parameter representation

Learned behavior could be stored as:

- dense arrays;
- sparse arrays;
- low-rank factors;
- diagonal, permutation, and butterfly stages;
- convolution or spectral kernels;
- codebooks and indices;
- lookup tables;
- sampled fields;
- spline or polynomial coefficients;
- decision trees or forests;
- graph topology and edge attributes;
- geometry and instance metadata;
- Boolean or ternary circuits;
- generated code;
- procedural parameter generators;
- compressed bitstreams;
- local experts selected by a router;
- prototypes plus residuals;
- rules plus exceptions; or
- a hierarchy combining several of these forms.

### State representation

Transient state could be:

- append-only token-derived records;
- a bounded recurrent vector;
- fast and slow recurrent states;
- a ring buffer;
- a sparse associative store;
- key/value state;
- a hierarchy of summaries;
- a graph that grows or rewires;
- an attractor state;
- dynamic fast weights;
- a spatial field;
- a set of active concepts;
- an event log plus compact summaries;
- a multiresolution pyramid;
- external memory references;
- codec-compressed snapshots;
- probabilistic hypotheses; or
- different state forms owned by different components.

### Topology

A compiled component could be:

- a serial pipeline;
- a residual graph;
- a tree;
- a recurrent loop;
- a cellular neighborhood;
- a sparse message-passing graph;
- a mixture of local circuits;
- an event-driven dataflow graph;
- a producer-consumer queue network;
- a hierarchy of coarse and fine stages;
- an approximate path plus exact correction;
- several competing hypotheses; or
- a dynamically constructed topology.

### Time and activation

Execution could be:

- synchronous by layer;
- synchronous by token;
- streaming by event;
- recurrent until quiescence;
- activated only by changed state;
- clocked at several timescales;
- speculative with commit and rollback;
- asynchronous across graph branches;
- demand-driven by a query;
- scheduled by device-generated work; or
- split between a fast approximate circuit and slower reconciliation.

## 30. Turning Arithmetic into Addressing

Traditional learned transforms read many parameters and multiply them by many
signal elements. Another family of expressions uses the signal to select stored
behavior.

Forms include:

- direct lookup tables;
- perfect or learned hashes;
- product-quantized codebooks;
- multi-stage vector quantization;
- content-addressed dictionaries;
- decision trees;
- trie traversal;
- table cascades in which one lookup determines the next;
- sampled fields with interpolation;
- sparse page selection;
- local expert selection; and
- memoized reachable state transitions.

The stored entry may contain:

- a complete output;
- an output delta;
- a state delta;
- coefficients for a local function;
- another address;
- a candidate set;
- a short program; or
- parameters for a correction circuit.

GPU mappings include sampled images, buffers, descriptor arrays, cache-resident
tables, subgroup-cooperative lookup, and indirect work generation.

## 31. Turning Arithmetic into Interpolation

A function over the reachable signal domain can be represented by samples and
an interpolation rule.

Forms include:

- 1D, 2D, and 3D sampled fields;
- multiple factorized low-dimensional fields;
- tensor-product grids;
- sparse grids;
- adaptive meshes;
- radial basis functions;
- splines;
- multiresolution pyramids;
- prototypes with barycentric coordinates;
- local affine charts;
- mixtures of local polynomial patches; and
- field samples followed by a learned residual.

Possible GPU processes include:

- texture filtering;
- texel gathering;
- mip selection;
- sparse image residency;
- rasterization;
- barycentric interpolation;
- blending; and
- vector arithmetic for coordinate transforms and corrections.

High-dimensional model signals do not automatically possess spatial locality.
The compiler may need to discover projections, charts, factorization, or another
coordinate system over the reachable domain.

## 32. Turning Parameters into Structure

The learned function can be stored partly or wholly in a transform's structure
rather than in unrelated dense coefficients.

Forms include:

- diagonal and permutation stages;
- butterfly networks;
- Hadamard or Fourier transforms;
- block-circulant or Toeplitz operators;
- convolutions;
- wavelet transforms;
- tensor trains;
- Kronecker products;
- low-rank factors;
- repeated blocks with small modifiers;
- sparse graphs;
- routing topology;
- local receptive fields;
- hierarchical compositions; and
- structured base transforms plus sparse or low-rank exceptions.

These forms can reduce parameter storage, change arithmetic complexity, create
reuse, or expose locality. They can also be composed with ordinary matrix
fragments where a local dense transform remains part of the structure.

GPU mappings include vector ALUs, subgroup shuffles, shared-memory stages, FFT
or scan implementations, packed dot products, sparse gathers, and cooperative
matrix operations for remaining dense factors.

## 33. Turning Parameters into Programs

Instead of storing every coefficient, a component can store or synthesize a
program that produces behavior.

Forms include:

- procedural parameter generation from compact seeds;
- generated straight-line shader code;
- an interpreter over compact learned bytecode;
- decision programs;
- rewrite rules;
- finite-state transducers;
- grammar-like generation;
- cellular update rules;
- a library of reusable subroutines plus selectors;
- symbolic formulas;
- program sketches with learned constants; and
- rules plus exception tables.

Knowledge is then divided among:

- program topology;
- constants;
- control flow;
- state;
- selectors; and
- exceptions.

GPU realization can use instruction caches, scalar control, vector lanes,
indirect execution, callable shader mechanisms, or generated specialized
pipelines. Program divergence and code size become explicit costs.

## 34. Turning Similarity into Search

Attention and routing conventionally compute many similarities and then select
or weight results. Alternative expressions can search an index instead.

Forms include:

- trees;
- tries;
- locality-sensitive hashes;
- inverted indexes;
- product-key indexes;
- vector-quantized cells;
- hierarchical clustering;
- navigable graphs;
- Bloom filters and bitset intersections;
- sorted projections;
- spatial grids;
- k-d-like partitions;
- bounding-volume hierarchies;
- learned routing networks;
- multi-resolution coarse-to-fine search; and
- several low-dimensional indexes whose candidates are unioned.

The search result can be:

- an exact item;
- an approximate nearest set;
- a shortlist for exact scoring;
- a local expert;
- a memory page;
- a vocabulary partition; or
- a route through the next component.

GPU mappings include parallel tree traversal, bit operations, sorting and scans,
texture lookup, graph traversal, ray queries, and indirect dispatch.

## 35. Turning High-Dimensional Search into Geometry

Model keys, routes, or reachable regions can be encoded as:

- points;
- triangles;
- boxes;
- rays;
- curves;
- transforms of reusable geometry;
- layered depth ranges;
- primitive identifiers; or
- spatial occupancy.

Possible expressions include:

- a query projected into several 3D rays;
- keys represented as bounding volumes;
- experts represented as spatial regions;
- a decision boundary represented as a mesh;
- priority represented by depth;
- route acceptance represented by stencil or coverage;
- nearest candidates represented by first intersections; and
- local coefficients carried as vertex or instance attributes.

Possible GPU processes include acceleration-structure traversal, raster
coverage, clipping, depth/stencil, barycentric interpolation, and blending.

The encoding from high-dimensional signals into geometry is part of the learned
or synthesized circuit. Geometry is not assumed to preserve source-model
similarity without that encoding.

## 36. Turning Nonlinearity into Piecewise Behavior

Conventional activations use formulas such as sigmoid, SiLU, GELU, exponent, and
division. Alternative nonlinear expressions include:

- threshold logic;
- lookup tables;
- piecewise constants;
- piecewise affine functions;
- splines;
- polynomial or rational approximations;
- decision trees;
- quantized state transitions;
- winner-take-all competition;
- saturating counters;
- min/max morphological operations;
- stochastic acceptance;
- local expert selection; and
- event-triggered state changes.

GPU mappings include compare/select instructions, texture lookup, rasterized
charts, depth/stencil tests, min/max reductions, bitwise logic, and local
arithmetic.

## 37. Turning Dense Features into Sparse Events

A component can represent only changes or active features instead of rewriting a
complete dense vector every activation.

Forms include:

- active coordinate lists;
- sparse deltas;
- threshold crossings;
- event timestamps;
- concept activations;
- changed-state pages;
- sparse graph messages;
- expert activations;
- run-length encoded regions;
- bitmasks;
- priority queues; and
- quiescent state that performs no work.

Execution can:

1. detect changed or active regions;
2. compact them;
3. generate work only for those regions;
4. propagate deltas;
5. stop propagation when no meaningful change remains; and
6. fall back to a denser expression when the active set expands.

GPU mappings include ballots, scans, atomics, queues, indirect dispatch,
scatter/gather, and persistent work consumers.

## 38. Turning Values into Bits and Symbols

Signals and parameters need not be expanded into wide scalars before every
operation.

Forms include:

- binary values;
- ternary values;
- bitplanes;
- low-bit integers;
- packed categorical codes;
- symbolic feature sets;
- Bloom filters;
- hyperdimensional binary vectors;
- bit-sliced counters;
- sparse masks plus compact values; and
- mixed symbolic/numerical records.

Operations include:

- XNOR and population count;
- Boolean formulas;
- mask intersection and union;
- bit permutation;
- table lookup;
- Hamming similarity;
- majority;
- saturating update;
- exact integer comparison; and
- selected scalar corrections.

GPU mappings include integer lanes, bitfield instructions, subgroup ballots,
population counts, packed memory operations, and lookup tables.

## 39. Turning History into Evolving State

Instead of retaining an explicit record and re-addressing it on every activation,
history can modify a state transition.

Forms include:

- conventional recurrence;
- state-space filters;
- gated recurrence;
- reservoirs;
- attractor networks;
- fast weights;
- associative-memory slots;
- multi-timescale state;
- recurrent convolution;
- delay networks;
- oscillatory or phase state;
- hierarchical summaries;
- episodic exceptions plus bounded working state;
- learned compression and decompression of state; and
- periodic reconciliation with explicit history.

The state update can be:

- additive;
- multiplicative;
- gated;
- competitive;
- overwrite-based;
- event-triggered;
- sparse;
- reversible;
- probabilistic; or
- a composition of fast and slow updates.

GPU mappings include resident registers or buffers, scans for parallel prefill,
per-event recurrent steps, local memory, atomics, sampled fields, and
device-owned loops.

## 40. Turning Position into Dynamics

Order and distance can be represented without adding or rotating a dense
position vector.

Forms include:

- delay lines;
- oscillator phase;
- state-transition count;
- temporal convolution;
- relative graph edges;
- age fields on memories;
- decay;
- multirate clocks;
- hierarchical time buckets;
- spatial displacement;
- path length through a graph;
- causal event order; and
- learned state whose evolution itself encodes position.

GPU mappings include recurrence, ring buffers, texture coordinates, convolution,
integer counters, phase arithmetic, and multiresolution fields.

## 41. Turning Normalization into Representation Constraints

Dynamic-range control can be expressed through:

- explicit RMS or variance normalization;
- fixed-point scaling;
- block floating point;
- per-region exponents;
- bounded activations;
- unit-vector or spherical coordinates;
- rank or order statistics;
- logarithmic values;
- sign and magnitude separation;
- automatic gain control;
- homeostatic recurrent rules;
- quantile maps;
- normalized codebooks; and
- representations whose valid states already have bounded scale.

The normalization responsibility may then be fused into encoding, lookup,
interpolation, state update, or parameter selection rather than appearing as a
separate reduction.

## 42. Turning Residual Addition into General Signal Composition

Alternative composition operations include:

- weighted sum;
- gated sum;
- concatenation followed by selection;
- min or max;
- overwrite by priority;
- union or intersection of symbolic features;
- voting;
- mixture distributions;
- depth-priority selection;
- stencil-controlled updates;
- graph message accumulation;
- phase or waveform superposition;
- probabilistic product or mixture; and
- merge rules over state deltas.

GPU mappings include vector arithmetic, reductions, atomics, render-target
blending, depth/stencil, bitwise operations, and local workgroup accumulators.

## 43. Turning Vocabulary Scoring into Structured Decoding

A full vocabulary projection scores every token independently. Alternative
expressions include:

- hierarchical softmax;
- token tries;
- prefix trees;
- byte or subtoken composition;
- adaptive vocabularies;
- candidate retrieval followed by exact scoring;
- product codes whose parts are scored separately;
- graph walks through legal continuations;
- finite-state transducers;
- grammar-constrained generation;
- cached candidate sets by state region;
- nearest-prototype output;
- multi-stage coarse-to-fine scoring; and
- iterative refinement of a candidate symbol.

GPU mappings include tree traversal, search indexes, bitsets, texture lookup,
ray/BVH candidates, subgroup selection, sorting, and small final projections.

## 44. Turning One Token at a Time into Parallel Hypotheses

Autoregressive dependence does not require every internal calculation to follow
one committed scalar path.

Alternative expressions include:

- speculative branches;
- token trees;
- parallel candidate continuations;
- beam-like state sets;
- draft-and-verify;
- multi-token predictors;
- distributions propagated without immediate sampling;
- particle-like hypotheses;
- rollback and state snapshots; and
- branch merging when states become behaviorally equivalent.

GPU mappings include batched workgroups, tree-shaped state, copy-on-write state,
indirect dispatch, parallel sampling, prefix reuse, and device-side commit.

## 45. Turning Host Scheduling into Device Dataflow

The feedback process can be represented as a resident dataflow graph:

```text
external input queue
        |
        v
compiled state transition
        |
        +----> public output queue
        |
        +----> state commit
        |
        `----> continuation / stop / route
                     |
                     `----> next device work
```

Realizations include:

- persistent shaders;
- bounded resident feedback windows;
- indirect dispatch;
- execution graphs;
- device-visible queues;
- sampler-written continuation state;
- event-driven graph activation; and
- asynchronous producer-consumer stages.

This changes the expression of control and feedback even when the learned
component calculations remain unchanged.

## 46. Turning Stored Parameters into Reconstructed Parameters

A component does not need every parameter to exist as a resident independently
addressable scalar.

Forms include:

- codebook reconstruction;
- delta coding;
- low-rank generation;
- procedural blocks;
- shared prototypes with modifiers;
- compressed tiles expanded on demand;
- entropy-coded pages;
- media-codec surfaces;
- deterministic pseudorandom bases plus learned coefficients;
- layer-to-layer prediction; and
- generated parameters conditioned on current state.

The reconstruction can happen:

- once at compilation;
- once at model mounting;
- once per selected component;
- once per cache residency interval;
- on demand per tile; or
- inline while applying the parameter.

GPU mappings include vector decoding, texture formats, copy engines, media
decode, local codebooks, and asynchronous prefetch.

## 47. Turning Exact Global Work into Coarse-to-Fine Work

Many responsibilities can be decomposed into:

1. a cheap coarse representation;
2. candidate or region selection;
3. a more precise local calculation;
4. optional correction; and
5. fallback when the coarse stage is uncertain.

This pattern can apply to:

- attention;
- MoE routing;
- MLP feature activation;
- vocabulary scoring;
- state lookup;
- normalization;
- component execution;
- parameter residency; and
- stopping decisions.

Coarse and fine stages may use entirely different representations and GPU
processes.

## 48. Turning Approximation into Verified Correction

A candidate expression can produce:

- a result;
- an error estimate;
- a confidence;
- a set of unresolved coordinates;
- a candidate set;
- or a request for exact evaluation.

The correction path can then:

- repair selected outputs;
- execute the source operation on uncertain cases;
- expand the candidate set;
- fetch a missing state page;
- refine a field region;
- or record a behavioral counterexample for recompilation.

This makes approximation policy part of the circuit rather than a global
all-or-nothing decision.

# Part V: Mapping Responsibilities to Expressions

## 49. Cross-Map

The table below is a set of possible mappings, not an assignment of one
expression to each responsibility.

| LLM responsibility | Conventional expression | Other expression families | GPU processes that may realize them |
| --- | --- | --- | --- |
| Symbol transduction | Embedding row gather | Hashes, compositional codes, tries, codebooks, procedural embeddings, sampled fields | Buffer gather, texture fetch, bit operations, tree traversal |
| Position and order | Added position vector or RoPE | Delay, phase, recurrence, age, graph distance, temporal filters | ALU phase updates, ring buffers, convolution, texture coordinates |
| Feature transformation | Dense linear projection | Structured transforms, sparse graphs, lookup fields, programs, logic circuits, local experts | Vector/scalar ALU, subgroup shuffle, texture sampling, indirect dispatch |
| Contextual addressing | Dot-product attention | Trees, hashes, associative memory, graph search, BVHs, recurrent state, coarse-to-fine search | Reductions, bitsets, ray traversal, texture lookup, graph traversal |
| Memory write | KV append | Slot updates, fast weights, summaries, event logs, attractor state, sparse deltas | Buffer writes, atomics, queues, recurrent kernels, transfers |
| Memory read | Softmax-weighted value sum | Direct lookup, candidate retrieval, graph neighborhood, field sampling, recurrent readout | Gather, texture filter, ray query, reductions, local memory |
| Nonlinearity | SiLU/GELU/sigmoid | LUT, spline, mesh, decision tree, threshold logic, state machine | Texture sample, raster interpolation, compare/select, bit operations |
| Gating | Elementwise multiply | Masks, conditional routes, depth/stencil acceptance, event activation, symbolic conjunction | ALU masks, ballots, stencil, indirect work |
| Routing | Linear scores plus top-k | Tree, hash, BVH, trie, graph, geometry coverage, cached route | Search, sort, ray traversal, rasterization, indirect dispatch |
| Mixing | Residual or weighted sum | Blend, vote, min/max, overwrite, symbolic union, graph accumulation | ROP blend, reductions, atomics, bitwise operations |
| Dynamic-range control | RMSNorm/LayerNorm | Bounded codes, block exponents, rank, AGC, normalized fields, homeostatic state | Reductions, integer scaling, lookup, recurrent update |
| Output scoring | Full vocabulary projection | Hierarchy, trie, candidate search, product code, grammar, prototype retrieval | Tree/BVH traversal, bitsets, small projections, sorting |
| Sampling | Softmax and random draw | Hierarchical choice, parallel hypotheses, accept/reject, state-machine policy | Reductions, scans, PRNG shader code, queues |
| Feedback | Host-scheduled next token | Resident device loop, event graph, persistent circuit, indirect continuation | Command processor, indirect dispatch, execution graph, queues |
| Parameter supply | Resident tensors | Generated, decoded, codebook, procedural, prefetched, compressed tiles | Caches, copy engines, media decode, texture formats, ALU reconstruction |

## 50. GPU-Process-to-Responsibility Cross-Map

The inverse view exposes reuse of one hardware process across several model
responsibilities.

| GPU process | LLM responsibilities it could participate in |
| --- | --- |
| Subgroup reductions and scans | Normalization, softmax, top-k, compaction, sampling, state aggregation |
| Bitwise and integer execution | Quantized transforms, routing masks, symbolic state, Hamming search, finite-state control |
| Texture lookup and filtering | Embeddings, local nonlinear functions, field-based transforms, memory read, multiresolution state |
| Rasterization and interpolation | Piecewise functions, region routing, sparse expansion, local-coordinate evaluation |
| Depth/stencil | Threshold gates, priority selection, rejection, masks, bounded discrete state |
| Render-target blending | Residual mixture, expert mixture, sparse contribution accumulation, min/max competition |
| Ray/BVH traversal | Candidate retrieval, expert routing, associative memory, vocabulary shortlist, graph-like search |
| Registers and local memory | Recurrent state, accumulators, local dictionaries, active candidates, fused component state |
| Cache hierarchy | Reused parameters, active experts, field tiles, hot state, local programs |
| Atomics | Sparse state updates, queues, graph propagation, reference counts, histograms |
| Copy/transfer engines | Prefetch, migration, snapshot, state paging, multi-device transport |
| Media engines | Reconstruction of codec-encoded parameter or state surfaces |
| Indirect dispatch and execution graphs | Event-sparse work, dynamic routing, feedback, variable candidate processing |

# Part VI: Worked Expressions from the Design Space

## 51. Neural Texture Circuit

A component can be represented as several learned low-dimensional fields:

```text
input and state
      |
      v
coordinate encoders
      |
      +----> sampled field A ----+
      +----> sampled field B ----+--> composition --> output
      +----> sampled field C ----+               `-> next state
```

The fields may contain:

- output fragments;
- state deltas;
- local coefficients;
- expert identifiers;
- confidence;
- correction parameters; or
- coordinates into another field.

The coordinate encoders may be dense, sparse, structured, hashed, recurrent, or
symbolic. Several factorized fields can cover different projections of the
reachable domain. Mip levels can represent coarse and fine behavior. Sparse
residency can allocate only visited regions.

GPU processes include coordinate arithmetic, texture lookup, interpolation,
format conversion, texture caching, and result composition.

## 52. BVH-Routed Memory

High-dimensional keys can be projected into one or more geometric spaces:

```text
query
  |
  +--> projection A --> ray or point query --+
  +--> projection B --> ray or point query --+--> candidate union
  +--> projection C --> ray or point query --+          |
                                                        v
                                                  local scoring
```

Keys can be represented as boxes, triangles, instances, or other supported
geometry. Instance metadata points to memories, experts, state pages, or output
candidates. Traversal returns a shortlist; another expression resolves final
scores.

Mutable state may use:

- acceleration-structure updates;
- an immutable base plus a small mutable structure;
- periodic rebuilds;
- several age tiers; or
- a non-geometric exception table.

## 53. Rasterized Local Function

A projected reachable domain can be tessellated into local regions. Vertex
attributes hold function values or coefficients:

```text
signal coordinates
      |
      v
mesh/chart selection
      |
      v
raster coverage + barycentric interpolation
      |
      v
fragment-local transform or state update
```

Multiple render targets can carry different signal partitions. Depth and stencil
can select regions or priorities. Blending can combine contributions from
overlapping charts.

The representation may use many low-dimensional charts rather than one
high-dimensional mesh.

## 54. Packed Logic and Lookup Circuit

Signals and state can be represented by bitplanes or compact symbols:

```text
packed input/state
       |
       +--> Boolean/ternary transform
       +--> Hamming or mask-based routing
       +--> table lookup
       +--> packed recurrent update
       `--> selected numerical correction
```

This expression places learned behavior in masks, permutations, table contents,
logic, and sparse corrections rather than a dense floating-point matrix.

## 55. Structured Transform with Exceptions

A dense learned transformation can be expressed as:

```text
output =
    structured_base(input)
  + sparse_exceptions(input)
  + low_rank_correction(input)
```

The base can use diagonal, permutation, butterfly, convolutional, spectral, or
repeated-block structure. The exception path stores behavior that the base
cannot reproduce on the reachable domain.

This expression can use subgroup shuffles and shared-memory stages for the base,
sparse gathers for exceptions, and small matrix operations for a low-rank
correction.

## 56. Bounded Multi-Timescale Stream State

Growing attention state can be expressed as interacting state banks:

```text
event
  |
  +--> fast working state ------+
  +--> slow semantic state -----+--> output
  +--> associative exceptions --+
  `--> external references -----+
```

Each bank has its own write, decay, replacement, and read rules. Fast state can
update every event; slow state can update only when a learned condition fires.
Exceptions can retain details that the bounded summaries do not encode.

Prefill can use scans or batched state construction. Decode can use recurrent
steps with resident state.

## 57. Event-Sparse Graph

The model can execute as propagation of state changes:

```text
external event
      |
      v
changed nodes / active features
      |
      v
compact work queue
      |
      v
propagate deltas until quiescent or yielded
```

Nodes whose inputs and state do not change perform no transition. The graph can
use thresholds, exact change tests, confidence, or scheduled refresh to decide
activation.

GPU processes include ballots, scans, atomics, queues, sparse gather/scatter,
indirect dispatch, and persistent consumers.

## 58. Codec-Reconstructed Parameter Stream

Related parameter tiles or state snapshots can be represented as frames in a
supported codec:

```text
encoded parameter/state sequence
             |
             v
        media decoder
             |
             v
 reconstructed image surfaces
             |
             v
 sampled, copied, or shader-expanded circuit data
```

Frame prediction can represent deltas among related tiles or successive state
snapshots. The result may be consumed as an image, copied into a buffer, or
expanded by a shader. Codec constraints, reconstruction error, surface layout,
and decode latency are part of the expression's realization.

## 59. Heterogeneous Composite

The preceding forms can be composed:

```text
input/state
    |
    +--> bit-level change detector
    |
    +--> search or geometric router
    |
    +--> texture-field local circuit
    |
    +--> structured numerical correction
    |
    +--> blend or graph accumulation
    |
    `--> recurrent state + device continuation
```

No one mechanism defines the component. The compiler can preserve explicit
representation boundaries or absorb conversions into adjacent subcircuits.

# Part VII: Generating Further Expressions

## 60. Questions About the Source Behavior

- Which distinctions in the input actually affect future observable behavior?
- Which theoretically possible input vectors are unreachable in normal model
  operation?
- Which outputs are locally smooth over the reachable domain?
- Which behaviors are piecewise constant, piecewise affine, or clustered?
- Which features are sparse, stable, repeated, or mutually exclusive?
- Which state details affect only the near future?
- Which state details survive over long horizons?
- Which errors are corrected by later components?
- Which components implement repeated variants of the same local behavior?
- Which parameter regions are selected together?
- Which operations exist because of the source representation rather than the
  responsibility?
- Which internal states are behaviorally equivalent under all likely future
  inputs?

## 61. Questions About Representation

- Can arithmetic become addressing?
- Can addressing become geometry?
- Can parameters become topology?
- Can values become ranks, symbols, or bits?
- Can a dense vector become sparse events?
- Can explicit history become evolving state?
- Can a global function become local charts?
- Can a high-dimensional domain factor into several low-dimensional domains?
- Can an exact result become a candidate plus correction?
- Can repeated parameters become a generator and exceptions?
- Can position become phase, delay, age, or topology?
- Can normalization become an invariant of the representation?
- Can output symbols be constructed hierarchically instead of scored
  independently?
- Can two different internal states be merged because they have the same future
  behavior?

## 62. Questions About Hardware Mapping

- Which values can remain in registers or local memory across several steps?
- Which data can be arranged for texture or cache locality?
- Which decisions are wave-uniform and can use scalar control?
- Which irregular accesses can be sorted or compacted first?
- Which searches can use trees, bitsets, or acceleration structures?
- Which local functions can use fixed interpolation?
- Which mixtures fit blend, reduction, or atomic processes?
- Which work can be generated indirectly on the device?
- Which transfers can overlap independent computation?
- Which representations can remain packed while being processed?
- Which fixed-function unit is accessible through the selected API and format?
- Which units share a bottleneck even when their logical operations differ?
- Which conversion costs appear only at representation boundaries?

## 63. Questions About Composition

- Where should a candidate remain exact?
- Where can an approximation expose confidence?
- What is the correction path?
- Can several weak indexes produce a strong candidate set?
- Can coarse state coexist with exact episodic exceptions?
- Can one component emit the representation the next component consumes
  directly?
- Can a basis change be absorbed into adjacent parameterizations?
- Can prefill construct a state that decode evolves through another realization?
- Can active streams share permanent circuits while keeping independent
  transient state?
- Can device-owned control remove a host synchronization boundary?
- Can a graph become idle without destroying its state?

These questions are generative. New answers extend the design space without
requiring this document to enumerate every resulting circuit.

# Part VIII: From Expressions to Measured Experiments

## 64. Shared Behavioral Contract

A candidate expression is evaluated against the source behavior at more than one
level.

### Teacher-forced evaluation

The source and candidate receive the same external and feedback sequence.
Measurements can include:

- component output error;
- output-distribution divergence;
- top-k overlap and rank stability;
- state-transition consistency;
- route or memory recall;
- correction rate; and
- confidence calibration.

### Free-running evaluation

Each circuit consumes its own selected feedback. Measurements can include:

- long-horizon output-distribution divergence;
- stability;
- delayed-recall behavior;
- interruption and resumption;
- capability probes;
- state growth or boundedness;
- stopping behavior; and
- divergent histories collected as behavioral counterexamples.

Exact intermediate tensor equality is not required when the candidate uses a
different representation. Exact generated-text equality is not sufficient as
the only criterion because small score changes can select a different but
behaviorally valid continuation.

## 65. Shared Execution Measurements

Measurements can include:

- latency per event, component activation, and generated symbol;
- prefill and decode throughput;
- permanent-parameter bytes read;
- transient-state bytes read and written;
- representation-conversion traffic;
- cache and residency behavior;
- vector, scalar, matrix, texture, raster, ray, copy, and media utilization where
  observable;
- dispatches, draws, traces, transfers, and synchronization;
- host interactions;
- active-set size;
- candidate-set size;
- correction and fallback rates;
- resident and temporary memory;
- index, field, mesh, graph, or codec construction;
- update and rebuild costs;
- energy or board power under a controlled measurement; and
- time spent in each subcircuit and representation boundary.

Compilation, construction, warmup, steady-state execution, and teardown are
reported separately.

## 66. Experimental Scope Record

Every measured result should identify:

```text
source model and package:
source component or graph boundary:
source implementation:
behavioral trace domain:

candidate responsibility:
candidate expression:
signal representation:
parameter representation:
state representation:
topology:
correction or fallback:

device:
API and required features:
realization:
build or construction procedure:
steady-state procedure:

behavioral measurements:
execution measurements:
counterexamples:
artifacts:
observed result:
```

The observed result applies to the tested source behavior, device, realization,
and workload. It does not close the surrounding expression family or prevent a
different composition from being explored.

GPU-executing work must follow the sequential-test and device-residency rules in
[AGENTS.md](AGENTS.md).

# Part IX: Model Anatomy and Modular Compilation

## 67. Where Time Is Spent in Agentic Coding

An agentic coding session combines model inference with operations outside the
model:

```text
total agent time =
    prompt and context preparation
  + prefill
  + repeated decode transitions
  + sampling and feedback
  + tool execution and external waiting
  + orchestration, queuing, and synchronization
```

These terms must be measured separately. A slow build, test suite, network
request, or filesystem operation can dominate end-to-end wall time without being
an inference cost.

Within model inference, a generated output performs the model transition again
for every generated token. Long reasoning traces, code generation, and repeated
tool-call formulation therefore accumulate many decode transitions. Decode is
commonly dominated by repeatedly reading permanent parameters and executing the
layer sequence for a narrow activation batch. Very large or repeatedly
reprocessed contexts can instead make prefill and context-state construction
dominant.

The relevant measurements include:

- context tokens ingested;
- output tokens generated;
- prefill time and throughput;
- decode time and throughput;
- permanent parameter bytes read per generated token;
- transient state bytes read and written per generated token;
- model dispatch and synchronization count;
- time waiting for tools and external processes; and
- useful work versus speculative, cancelled, or discarded work.

This decomposition prevents an optimization to agent orchestration from being
reported as an inference optimization, or a decode optimization from being
hidden by unrelated tool latency.

## 68. General LLM Execution Anatomy

A conventional autoregressive language model can be described at the outermost
level as:

```text
tokens
  -> input representation
  -> layer 0
  -> layer 1
  -> ...
  -> layer N
  -> output representation
  -> candidate selection
  -> feedback token
```

The outer layer sequence is normally serial because the output state of one
layer is the input state of the next. It does not follow that all work within a
layer is serial. A layer is a directed computation graph containing independent
branches, joins, residual paths, parameterized transforms, and state
transitions.

Examples of internal parallelism and branching include:

- independent or grouped Q, K, and V projections;
- parallel attention heads;
- gate and value branches in a gated feed-forward network;
- routed expert selection and independent selected-expert evaluation;
- shared and routed expert paths;
- residual paths that bypass a transformation;
- recurrent or convolutional state update alongside feature computation; and
- reductions that join several branches.

Input transduction, final normalization, vocabulary projection, sampling, and
feedback are model components but are not ordinary repeated layers. Layer
boundaries describe the source architecture; they are not necessarily the most
efficient kernel, submission, tiling, or device boundaries.

## 69. The Top-Level Pedal Boundary

The modular-pedal representation keeps the logical source layer as the
top-level repeated pedal.

A layer supplies a useful top-level boundary because it has:

- a stable input and output activation contract;
- identifiable permanent parameters;
- identifiable transient state;
- an established position in the residual stream;
- repeated structural correspondence with other layers;
- existing support for removal, repetition, bypass, placement, and replacement;
  and
- a scope large enough for internal compiler optimization.

Layer clusters remain useful, but as composite editor groupings over several
layer pedals. A cluster can be repeated, collapsed, selected, or manipulated as
a unit without replacing the constituent layer identities.

The model input transducer, output transducer, sampler, and feedback connection
remain separate top-level components because they have different external
contracts and are not transformer-layer instances.

The top-level choice does not assert that a layer is indivisible. Selecting or
opening a layer reveals its internal semantic modules.

## 70. Layer Submodule Anatomy

A general layer view can begin with two major blocks:

```text
Layer
├── Token-mixer block
│   ├── input normalization
│   ├── token mixer
│   ├── mixer-owned state
│   ├── output transform
│   └── residual transport
└── Feature-transform block
    ├── input normalization
    ├── dense FFN or sparse MoE
    └── residual transport
```

The token mixer can expand differently according to the layer family.

An attention mixer can expose:

```text
Attention mixer
├── Q/K/V or combined projection
├── projection partitioning
├── per-head normalization
├── position transformation
├── KV state attachment
├── attention read
├── optional output gate
└── output projection
```

A recurrent, gated-delta, RG-LRU, state-space, or convolutional mixer can expose
its projections, gates, local filter, recurrence, state transition, and output
projection using the same recursive vocabulary.

A dense feature transform can expose gate, up, activation, multiplication, down,
and residual components. A sparse MoE transform can expose:

```text
Sparse MoE
├── router
├── top-k selection
├── routed expert bank
│   ├── expert 0
│   ├── expert 1
│   └── ...
├── shared expert
├── route-weighted reduction
└── residual
```

The expert bank is one module at the normal layer-inspection depth. Individual
experts are a deeper expansion rather than hundreds of peer top-level pedals.

KV is state owned by attention, not an operation in the outer serial layer
chain. The editor can render it as an inspectable state attachment. Recurrent
matrices, convolution history, and other persistent per-stream memories follow
the same ownership model.

## 71. Meaning of a Standalone Subcomponent

Two meanings of standalone must remain distinct.

A **semantic standalone component** has:

- a stable identity and responsibility;
- declared inputs and outputs;
- declared parameter references;
- declared state ownership and transitions;
- membership in a parent module;
- a replaceable region of the semantic execution graph; and
- a behavioral boundary against which a transformation can be checked.

A **physical standalone component** additionally forces an execution boundary:

- a separate kernel or kernel sequence;
- materialized intermediate storage;
- an independent scheduling unit;
- synchronization;
- possibly independent placement; and
- possibly independent lifetime and state-policy controls.

Pedalboard inspection and compiler transformation require semantic
standalone-ness. They do not require physical standalone-ness. A semantic module
may disappear into a fused kernel while remaining identifiable through compiler
provenance.

## 72. Why Models Are Lowered

Lowering translates an architectural description into operations precise enough
to validate, optimize, schedule, and execute.

For example:

```text
architectural description
attention block

        |
        v

lowered semantic graph
normalization
  -> Q/K/V projection
  -> head normalization
  -> position transformation
  -> KV state update
  -> attention read
  -> output projection
  -> residual

        |
        v

optimized physical graph
normalization
  -> fused projection regions
  -> fused normalization and position regions
  -> fused KV update and attention
  -> fused output projection and residual
```

The lowered representation makes explicit:

- data dependencies and topological order;
- exact intermediate signals and shapes;
- parameter use;
- state reads, writes, ownership, and update order;
- boundaries available for behavioral checking;
- regions available for fusion or replacement; and
- the operations for which a backend must select implementations.

Lowering does not require erasing architectural structure. It also does not
necessarily reduce every operation to a matrix multiplication or machine
instruction. A lowered intermediate representation can retain operations such
as attention, recurrence, sparse expert evaluation, scan, lookup, or structured
transform while making their contracts explicit.

## 73. Cost of Making Modules Hard Execution Boundaries

If every semantic subcomponent becomes a mandatory physical boundary, the
compiler can lose:

- fusion across adjacent semantic operations;
- elimination of intermediate values;
- register or local-memory producer-consumer locality;
- buffer reuse;
- reduced permanent and transient memory traffic;
- fewer kernel launches and synchronization points;
- legal reordering of independent operations;
- joint tiling across several operations; and
- backend freedom to replace a region with a device-specific implementation.

Examples include fusing:

- K and V projections;
- Q/K head normalization with position transformation;
- KV append with attention read;
- output projection with residual transport; and
- gate/up projections with activation and multiplication.

A physical operation can cover source nodes from more than one semantic module.
Per-module profiling must therefore support shared attribution rather than
pretending that every kernel belongs to exactly one module.

The modular design retains both views:

```text
semantic module tree          editor and transformation identity
lowered semantic graph        exact behavior and dataflow
optimized physical graph      backend execution
```

## 74. Modular Compilation Contract

A compiled layer can carry a versioned semantic tree alongside its flat lowered
or optimized node list:

```text
module:
    stable id
    kind and responsibility
    parent and children
    semantic source-node membership
    input and output contract
    parameter references
    owned state ports
```

The module tree references canonical semantic source nodes. An optimized node
records the source nodes from which it was compiled. A packaged kernel records
or inherits the same provenance.

Validation can require:

- unique module IDs;
- valid parent and child relationships;
- complete source-node coverage by leaf modules;
- unambiguous leaf ownership;
- child membership contained by parent membership;
- valid parameter and state references;
- root coverage of the complete layer;
- complete optimized-node coverage of semantic source nodes; and
- preservation of module metadata through packaging.

The runtime continues to schedule the optimized physical graph. The editor uses
the semantic tree and provenance to display module anatomy, implementation,
state, parameters, kernels, and measured cost.

## 75. Current NERVE Compiler Gap

The current transpiler already emits much of the source anatomy in each layer's
`reference_decomposition`. It records the topological normalization, mixer,
residual, feed-forward, and final residual components. Mixer descriptions also
record internal attention, convolutional, recurrent, and gated-delta
subcomponents.

For example,
[`compiled_models/qwen3_6_27b_fp8_bench/transpiled/layers/layer_03.json`](compiled_models/qwen3_6_27b_fp8_bench/transpiled/layers/layer_03.json)
describes an attention layer containing projections, per-head normalization,
RoPE, KV memory, attention read, output gating, and output projection.

Circuit lowering currently reconstructs these operations as a flat,
topologically ordered `nodes` list. The lowered circuit retains executable
semantics but no longer retains their parent-child module relationships. The
optimizer then fuses regions and records their semantic source nodes through
`compiled_from`.

The modular compiler proposal closes that gap by:

1. formalizing the source module tree;
2. constructing exact source-node membership during lowering;
3. preserving the tree through optimization and packaging;
4. mapping optimized nodes and kernels back through source provenance;
5. exposing the tree through the runtime editor schema; and
6. rendering an expandable layer pedal without changing physical execution
   boundaries.

## 76. Component-by-Component Transformation

The module tree provides a unit for changing one responsibility without fixing
the rest of the layer to the same representation:

```text
select semantic module
        |
        v
declare its current boundary behavior
        |
        v
replace its expression, representation, state, or realization
        |
        v
lower the replacement into an explicit semantic graph
        |
        v
verify local and whole-model behavior
        |
        v
optimize the complete physical graph
```

Possible transformations include replacing only:

- a dense projection with a structured transform and correction;
- attention addressing with search, recurrence, or another memory expression;
- KV history with a bounded or multiscale state;
- an expert with a generated, sparse, symbolic, or lookup representation;
- a gate with a piecewise or bit-level expression;
- normalization with a representation invariant;
- vocabulary scoring with hierarchical candidate construction; or
- a repeated group of equivalent modules with a shared generator and
  layer-specific exceptions.

The replacement boundary is semantic. After verification, the compiler remains
free to fuse the replacement with neighboring operations. This permits local
experimentation without imposing local execution boundaries.

# References

## GPU architecture and APIs

- [AMD Radeon AI PRO R9700 specifications](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)
- [AMD RDNA 4 Instruction Set Architecture reference](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
- [Vulkan shader execution and subgroup model](https://docs.vulkan.org/spec/latest/chapters/shaders.html)
- [Vulkan sampled-image operations](https://docs.vulkan.org/spec/latest/chapters/textures.html)
- [Vulkan rasterization](https://docs.vulkan.org/spec/latest/chapters/primsrast.html)
- [Vulkan depth and stencil operations](https://docs.vulkan.org/spec/latest/chapters/fragops.html)
- [Vulkan framebuffer blending](https://docs.vulkan.org/spec/latest/chapters/framebuffer.html)
- [Vulkan ray tracing](https://docs.vulkan.org/spec/latest/chapters/raytracing.html)
- [Vulkan acceleration structures](https://docs.vulkan.org/spec/latest/chapters/accelstructures.html)
- [Vulkan execution graphs](https://docs.vulkan.org/spec/latest/chapters/executiongraphs.html)
- [Vulkan command buffers](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html)
- [Vulkan copy commands](https://docs.vulkan.org/spec/latest/chapters/copies.html)

## Conventional and alternative sequence-model expressions

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://openreview.net/forum?id=H4DqfPSibmx)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
