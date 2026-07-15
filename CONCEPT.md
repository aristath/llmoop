# Continuous Stream Inference Engine

## Summary

This inference engine treats a language model as a continuously available, event-driven signal processor rather than a request/response service.

The engine is modeled after an audio signal chain consisting of a guitar, a pedalboard, an amplifier, and a feedback loop. The complete model is the pedalboard; its layers and other processing entities are individual pedals connected in series, in parallel, through mixers, and through delayed feedback paths. A user request is not a self-contained job that creates a temporary generation process. It is a signal injected into an existing stream. The board processes that signal, produces output, and feeds state back into itself so that the stream can continue.

The stream is logically always on. When it has no input and no internally scheduled continuation, it consumes no compute; its state simply remains suspended until the next event.

An existing model provides the behavioral reference, but its transformer implementation is not the required runtime structure. A behavioral compiler produces a stateful stream circuit and lowers it to vendor-neutral Vulkan/SPIR-V. While a stream is active, the GPU owns its sampling and feedback loop; the host injects external events, receives public output, and supplies control.

## Audio Model

The physical system is:

```text
                                      +--> pedal B --+
                                      |              v
guitar --> input mixer --> pedal A ---+-----------> mixer --> pedal D ---+--> amplifier
           ^                          |              ^                   |
           |                          +--> pedal C --+                   |
           |                                                             |
           +---------------- delayed insert/feedback <-------------------+
```

Its inference counterpart is:

| Audio component | Inference component |
| --- | --- |
| Guitar | User or another external input source |
| Sound | Tokens, embeddings, or another stream representation |
| Pedalboard | The complete model circuit graph |
| Pedal | A layer or another processing entity |
| Patch cable | A typed signal, state, or control connection |
| Splitter and mixer | Parallel paths, residual paths, and combination functions |
| Delayed feedback | State carried into a later activation |
| Input | New external information entering the stream |
| Insert-out | State produced for future processing |
| Insert-in | The point at which retained state re-enters the circuit |
| Output | Observable generated stream |
| Amplifier | The user or another consumer of the output |

The user injects a signal. The compiled board processes it and produces output. The insert loop carries the process forward, producing subsequent output until the circuit decides to stop. At that point the engine becomes idle without destroying the stream.

## The Stream Is the Primary Runtime Object

A conventional inference service usually models generation as a temporary job:

```text
request -> token -> token -> token -> end -> destroy job
```

This engine elevates the stream itself into the primary runtime object:

```text
                    +-------------------+----> public output
external input ---->| compiled model    |
                    | circuit           |
feedback state ---->|                   |----> next feedback state
                    +-------------------+
```

A request is an event within the lifetime of a stream, not the boundary of that lifetime. The stream may exist before, during, and after any individual interaction.

This makes several behaviors native to the architecture:

- New input can arrive while output is being produced.
- A running stream can be interrupted, redirected, or modulated.
- End-of-response is a control event, not necessarily the destruction of the process.
- Users, tools, models, sensors, and other systems can all act as live inputs.
- Stream identity and state can persist independently of any particular request.

## Event-Driven, Logically Always On

"Always on" does not initially mean that the GPU performs work during silence. It means that the circuit and its state continue to exist and remain addressable.

The lifecycle is:

1. An external event activates an idle stream.
2. The circuit processes the event and updates its internal state.
3. It may produce public output and schedule another activation through its feedback loop.
4. Feedback-driven activations continue for as long as the circuit keeps the loop open.
5. When the circuit closes the loop, computation stops and the retained state becomes idle.
6. A later external event resumes the same stream from that state.

There is no need for artificial clock tokens or continuous GPU activity during idle periods.

While generation is active, the device should own the steady-state loop. The host should not have to request or schedule every token:

```text
Host                                  GPU
----                                  ---
inject input -----------------------> input queue
                                      |
                                      v
                                model circuit
                                      |
                                      v
                                output selection
                                      |
                      +---------------+--------------+
                      |                              |
                      v                              v
                feedback input                 output queue
                      |                              |
                      +------> model circuit         +----> host
```

Logical persistence is separate from physical execution strategy. A backend may use a long-running shader, a sequence of bounded dispatches, or another mechanism while preserving the same device-owned stream semantics. The concept does not require one GPU program to spin forever during silence.

## The Model as a Compiled DSP Circuit

The source model defines a behavior that is compiled into a GPU-resident processing circuit. Its transformer architecture and weights are a reference implementation of that behavior, not a required description of the resulting circuit.

The compiled circuit does not need to reproduce the source model's attention heads, matrix multiplications, MLPs, layer topology, KV layout, or numerical path. It may use any internal representation and any sequence of calculations that produces sufficiently equivalent outputs for the same inputs and stream state.

The result is not treated as a function reconstructed around each prompt, but as an installed processor with stable input, output, and state ports.

A running instance consists of two parts:

```text
compiled stream processor
|-- permanent circuit
|   `-- immutable synthesized program and parameters
|
`-- transient circuit
    `-- mutable state belonging to this stream
```

The permanent circuit can be shared by many streams. Each stream has its own transient circuit and therefore its own continuity and identity.

The compiled form can define:

- the permanent circuit representation and parameter layout;
- the GPU operations and connections between them;
- the shape and placement of transient state;
- the ports through which external and feedback signals enter;
- the public output and feedback outputs; and
- the rules for updating, retaining, resetting, snapshotting, or forking state.

The compilation path is:

```text
source checkpoint and architecture
               |
               v
       reference execution
               |
               v
      behavioral compiler
               |
               v
       stream-circuit IR
               |
               v
       Vulkan/SPIR-V backend
               |
               v
     installed GPU processor
```

## Stream-Circuit Intermediate Representation

The central compiler artifact is a backend-neutral stream-circuit intermediate representation. It describes a graph of stateful signal-processing entities rather than a list of transformer operations or host-scheduled tensor kernels.

Each entity declares:

```text
entity
|-- signal input and output ports
|-- persistent parameters
|-- transient state
|-- state-transition behavior
|-- feedback and routing connections
|-- control input and event output ports
|-- numerical representation
`-- behavioral error contract
```

Its general transition is:

```text
(y_t, S_{t+1}, e_t) = C(x_t, S_t, c_t, r_t)
```

Where:

- `x_t` is an external or feedback signal;
- `S_t` is the current stream state;
- `c_t` is control such as interruption, gating, or routing;
- `r_t` is an explicit source of randomness;
- `y_t` is observable or internally routed output;
- `S_{t+1}` is updated state; and
- `e_t` contains control events produced by the circuit.

Randomness is explicit because individual reference layers may be deterministic while the complete generation loop includes stochastic output selection. Treating randomness as an input makes reference and compiled executions comparable under identical conditions.

Entity boundaries are logical. The backend may fuse entities, split them, or lower a connected region as one executable unit without changing its circuit-level contract.

## The Model as a Pedalboard Graph

The complete model is an editable graph of circuit entities. A transformer layer is one useful pedal-sized entity, but it is not the only possible granularity and is not an indivisible runtime primitive.

```text
input transducer
       |
       v
   layer pedal 0 <----> transient state 0
       |
       v
   layer pedal 1 <----> transient state 1
       |
       +-------------------+
       |                   |
       v                   v
   layer pedal 2       other entity
       |                   |
       +--------> mixer <--+
                    |
                    v
             output transducer
                    |
                    +----> delayed feedback
```

Each pedal instance declares:

```text
pedal instance
|-- typed input ports
|-- typed output ports
|-- immutable circuit and parameter references
|-- stream-owned transient state
|-- control ports
`-- transition behavior
```

The circuit graph makes topology first-class. Its basic editing operations include:

- insert, remove, replace, reorder, or bypass a compatible pedal;
- duplicate a pedal in series or in parallel;
- split and mix signals;
- add a delayed feedback connection;
- tap a signal for inspection;
- group several pedals into a rack entity; and
- expand a rack into its internal entities.

Because transformer layers commonly preserve the residual signal width, bypassing or removing one from a serial chain is often structurally simple:

```text
A --> B --> C

A --------> C
```

Port types and dimensions remain part of the contract. Where two entities are incompatible, the graph must contain an explicit adapter rather than silently reinterpret the signal.

### Layer pedals as device-placement boundaries

For the first practical architecture, each source model layer should be represented as a standalone pedal in the pedalboard schema. Smaller internal components may exist inside that pedal, and later compiler passes may fuse, split, or specialize implementation details, but the layer-level pedal boundary is important enough to preserve as a logical placement and routing boundary.

One reason is multi-device inference. If each layer is a self-contained pedal with typed input ports, output ports, permanent parameters, and stream-owned transient state, then different pedals can live on different execution devices:

```text
input
  |
  v
layer_00 pedal  @ GPU 0
  |
  v
layer_01 pedal  @ CPU
  |
  v
layer_02 pedal  @ GPU 1
  |
  v
layer_03 pedal  @ LAN device
  |
  v
output
```

The pedalboard schema does not need to change when placement changes. The circuit remains the same logical graph; only the cables become different kinds of transport. A short cable may be an in-device buffer reference. A longer cable may be a device-to-device copy, shared memory handoff, PCIe transfer, IPC channel, or LAN stream. In audio terms, the pedalboard still has the same pedals in the same order; the only thing that changes is cable length and cable type.

This makes placement a routing problem rather than a model-architecture problem:

- a pedal declares what it consumes and produces;
- a device backend declares what pedals it can host;
- a cable declares how signals move between hosted pedals;
- the scheduler decides when cross-device boundaries must synchronize; and
- the behavioral contract stays attached to the pedal, not to the device.

This is also why a layer should not be treated merely as a compile-time convenience. A layer pedal is a deployable unit. It can be hosted locally, remotely, duplicated, bypassed, inspected, replaced, or migrated, provided its port and state contracts remain compatible.

### Pedal duplication

Duplicating an entity does not necessarily duplicate its permanent parameters. Multiple instances can reference one immutable circuit while retaining independent stream state:

```text
instance B1 ----+
                +----> shared immutable parameters
instance B2 ----+

instance B1 --------> transient state B1
instance B2 --------> transient state B2
```

Serial duplication applies the entity twice:

```text
A --> B1 --> B2 --> C
```

Parallel duplication requires an explicit combination entity:

```text
      +--> B1 --+
A ----+         +--> mixer --> C
      +--> B2 --+
```

The mixer may sum, gate, select, concatenate, or otherwise combine compatible signals. Deterministic copies with identical parameters, input, and state produce identical output, so useful parallel copies require a meaningful difference in state, routing, control, representation, or parameters.

Duplicating a stateful pedal also requires an explicit state policy:

- **Fresh:** create empty state for the new instance.
- **Clone:** snapshot the original instance's current state.
- **Share:** let both instances address one state object with defined access ordering.
- **Derive:** compile new state from the original state.

State must never be accidentally copied or shared as a side effect of graph editing.

### Connection types and feedback

The graph distinguishes three kinds of connection:

1. A forward connection carries a signal to a later entity during the current activation.
2. A residual or parallel connection sends the signal through multiple same-activation paths before an explicit mixer.
3. A temporal feedback connection carries output through state into a later activation.

A true feedback edge must cross a delay or state boundary:

```text
entity output --> delay/state --> entity input on a later activation
```

A cycle without such a boundary is an algebraic loop: its output immediately depends on itself and requires a fixed-point solution. The graph validator must reject an instantaneous cycle unless an entity explicitly declares how to solve it.

After delayed feedback edges are separated at activation boundaries, the work within one activation remains schedulable as an acyclic graph.

### Hierarchical circuits

Circuit entities are hierarchical. A layer can appear as one opaque pedal while internally containing a smaller board:

```text
model board
`-- layer pedal
    |-- normalization
    |-- attention circuit
    |-- residual mixer
    |-- transformation circuit
    `-- residual mixer
```

The same circuit can therefore be viewed as a whole model, groups of layers, individual layers, internal operations, or lowered GPU instructions. Logical boundaries remain available for editing, instrumentation, and verification even when the backend fuses them during execution.

### Editing behavior versus compiling structure

Removing a pedal and compiling it away are different operations:

```text
remove pedal
    -> intentionally change the board's behavior

compile pedal away
    -> remove its executable boundary while synthesizing surrounding
       circuitry that preserves the board's behavior
```

Graph editing supports deliberate model surgery. Behavioral compilation may independently alter, fuse, or eliminate topology while remaining subject to the source board's behavioral contract.

## Vulkan/SPIR-V Baseline

Vulkan is the baseline execution target so that the engine does not require CUDA, ROCm, SYCL, or another vendor-specific compute stack. The intended hardware range includes AMD, Intel, NVIDIA, and other devices with suitable Vulkan compute support.

The stream-circuit IR remains independent of Vulkan, but its entities must be expressible using portable GPU resources and programs. A Vulkan lowering can use resident buffers for permanent parameters, transient stream state, signal frames, input and output queues, and control data, with SPIR-V programs implementing circuit transitions.

Portability is part of the purpose of the engine rather than an afterthought. Vendor-specific backends may eventually exist, but they must not define the architecture or its behavioral model.

## KV as a Transient Layer

In this architecture, KV is not best understood as a cache.

A cache is disposable: removing it makes a computation slower, but does not destroy the logical state of the process. The state retained by a running stream is authoritative. Removing it changes or destroys the stream itself.

For fixed key and value tensors, attention defines a temporary operator:

```text
A_S(q) = softmax(q K_S^T) V_S
```

In a reference transformer, the permanent weights generate queries, keys, and values. Once produced, the keys and values define a stream-specific function that transforms future queries. They can therefore be viewed as a dynamically synthesized layer rather than merely saved intermediate results.

The transient layer is ephemeral relative to the permanent weights, but persistent for the lifetime of its stream. It is part of the running circuit.

An activation can be described as:

```text
(public_output, next_stream_state, control) =
    process(external_input, current_stream_state)
```

Or more compactly:

```text
(y_t, S_{t+1}) = P_W(x_t, S_t)
```

Where:

- `W` is the compiled permanent circuit;
- `S_t` is the current transient circuit;
- `x_t` is an incoming external or feedback signal;
- `y_t` is observable output; and
- `S_{t+1}` is the updated transient circuit.

## A Logical Layer With Ports Throughout the Model

The transient layer need not be one physical layer placed at the beginning or end of the network. Attention state is associated with multiple model blocks, so the transient circuit may be one logical object with connections throughout the permanent circuit:

```text
permanent block 1 <-> transient bank 1
permanent block 2 <-> transient bank 2
permanent block 3 <-> transient bank 3
...
```

Together, these banks form the stream's transient layer. The engine owns its lifecycle as a single stateful object even when the compiled implementation distributes it across the model.

## Behavioral Compilation

The inner structure of a transformer is not part of the compiled circuit's behavioral contract. What matters is the transformation it performs.

For a stateless reference entity:

```text
y = F(x)
```

The compiler may synthesize any replacement `G` for which:

```text
G(x) ~= F(x)
```

over the input domain the model actually encounters.

The source and compiled paths may therefore be structurally unrelated:

```text
input ---> transformer calculations ---> reference output

input ---> synthesized GPU circuit ----> equivalent output
```

The replacement may use fused operations, fewer or differently shaped matrix operations, low-rank transformations, lookup structures, learned approximations, specialized numerical representations, or a completely different computational structure. The reference model is an executable specification that can be stimulated and measured; it is not a mandatory execution plan.

### Stateful equivalence

A running stream is stateful, so matching one isolated input and output is not sufficient. The reference behavior is:

```text
(y_t, S_{t+1}) = F(x_t, S_t)
```

The compiled circuit implements:

```text
(y_hat_t, S_hat_{t+1}) = G(x_t, S_hat_t)
```

`S_hat` does not need to resemble the source model's KV state. It may have a different shape, size, organization, and update mechanism. The requirement is that it preserve future behavior.

Two internal states are behaviorally equivalent when the same future input stream causes them to produce sufficiently similar future output streams. State equivalence is therefore defined by future observable behavior, not by tensor equality.

This allows the compiled transient circuit to be smaller, bounded, mutable, or otherwise structurally different from the source model's state while still representing essentially the same process.

### Layer boundaries are compilation tools

Individual transformer layers can be treated as black-box entities to divide, synthesize, and verify the circuit. Those boundaries do not need to survive compilation.

The compiler may replace:

- one reference layer with one circuit;
- one layer with several circuits;
- several layers with one circuit; or
- the complete transformer with a synthesized stream processor.

Layer boundaries are useful for decomposition, measurement, error localization, and verification. They are not requirements of the executable circuit.

### The reachable signal domain

A reference layer mathematically accepts a vast space of possible numeric vectors, but it encounters a much narrower distribution of signals during real model operation. A synthesized replacement needs its highest accuracy across this reachable signal domain rather than across every theoretically possible floating-point input.

This narrower domain gives the compiler freedom to construct a circuit specialized for the signals actually produced by the surrounding model.

### Behavioral closeness

Similarity between intermediate vectors is useful evidence, but it is not the final success criterion. Small internal differences may disappear in later processing or accumulate through the feedback loop until behavior changes.

Equivalence must therefore be evaluated using externally meaningful behavior, including:

- final output-distribution similarity;
- important output-ranking agreement;
- stability over long input and output streams;
- error accumulation through repeated feedback;
- response to interruptions and newly injected input; and
- preservation of the source model's externally observable capabilities.

The compiled circuit may produce very different intermediate signals while remaining behaviorally equivalent at its public ports.

### Behavioral compilation workflow

The source model acts as an executable oracle. Compilation can be driven by real trajectories through the source rather than uniformly sampled vectors from the full mathematical input space.

```text
reference model
      |
      v
capture reachable signal and state trajectories
      |
      v
synthesize a candidate circuit
      |
      v
compose it into the complete feedback loop
      |
      v
detect behavioral divergence
      |
      +----> add divergent cases and refine
```

A compiler can therefore operate counterexample by counterexample:

1. Run representative streams through the source model.
2. Capture actual entity inputs, outputs, and state transitions.
3. Synthesize a cheaper candidate implementation.
4. Test the candidate inside the composed model rather than only in isolation.
5. Capture inputs and histories that cause meaningful divergence.
6. Add those cases to the compilation set and refine the circuit.

This process does not require changing or retraining the source model. The model remains the behavioral authority while its compiled implementation is synthesized and verified separately.

### Closed-loop validation

Validation must operate at two levels.

Teacher-forced validation gives the source and compiled circuits the same input sequence and compares their output distributions and state behavior. This isolates local numerical and representational error.

Free-running validation lets each circuit consume its own selected output through the feedback loop. This reveals accumulated error, state instability, and behavioral divergence that isolated comparisons cannot detect.

A small difference in output scores can cross a selection boundary, producing a different token and therefore a different future input stream. Exact generated-text equality is consequently too brittle as the only criterion, while similarity between isolated intermediate vectors is too weak. Evaluation must include distributional similarity, long-horizon stability, and preservation of externally observable capabilities.

## Public Output and Feedback Are Separate Signals

In ordinary autoregressive generation, the visible token is generally also the signal fed into the next step. That remains a valid simple operating mode, but the four-jack architecture permits a more general design.

The public output and insert-loop output do not need to be identical:

- The public output carries language or other signals intended for an external consumer.
- The insert loop carries the private representation required for subsequent processing.

The insert signal may be richer than a visible token. It could contain tensors, sparse concepts, active intentions, unresolved predictions, routing information, control state, or dynamically generated parameters. Tokens are the audible signal, but they do not have to be the entire internal signal.

This separation allows the model to maintain continuity without exposing its complete recurrent representation or forcing all internal state through natural-language tokens.

## Transient State as DSP State

An audio processor does not need to retain the complete history of its input waveform. It retains the state its circuits require, such as delay lines, filter state, envelopes, oscillation phase, control parameters, and feedback energy.

A long reverb provides a concrete analogy. A direct implementation could revisit an arbitrarily long input history to calculate every output sample. A practical feedback network instead produces a long-lived response by evolving a much smaller internal state with nearly constant work per sample.

This suggests a central research hypothesis:

```text
explicit traversal of growing attention history
                    |
                    v
behaviorally equivalent evolving transient circuit
```

The hypothesis is not that KV can automatically be replaced by a small recurrent state. It is that attention over the reachable stream domain may admit another realization whose future behavior is sufficiently close while its state is bounded or cheaper to update.

The inference equivalent need not be an indefinitely growing token history. The transient circuit could instead contain purpose-built state such as:

- associative memory;
- short-lived activation;
- unresolved predictions;
- control and routing state;
- fast-changing working state;
- slow semantic state; and
- dynamically generated fast weights.

This state may eventually be:

- fixed in shape rather than append-only;
- mutable rather than an immutable record of history;
- divided into fast and slow timescales;
- selectively cleared, dampened, amplified, or gated;
- snapshotted and restored;
- forked into multiple streams;
- merged or cross-connected; and
- transported between compatible compiled circuits.

A fixed state shape would make the processor especially DSP-like: fixed GPU memory, stable circuit topology, and predictable per-event processing cost.

## Representation Research Directions

Audio samples, video frames, images, and three-dimensional fields are useful execution analogies, but none is assumed to be the correct semantic representation. Arbitrarily reshaping a dense activation vector into pixels does not create useful locality.

The deeper question is whether the behavioral compiler can discover a coordinate system in which reachable model signals become easier to process. Useful properties might include:

- sparsity;
- spatial or topological locality;
- separability;
- low-rank structure;
- stable clusters;
- repeated local transformations; or
- compact discrete or bit-level structure.

An invertible or behavior-preserving basis transformation could make a layer resemble a field effect even when its original embedding coordinates do not. Where possible, transformations between representations can be absorbed into adjacent entities rather than executed as separate conversions.

Packed numerical representations also form part of the compiler's search space. A low-bit model need not always be interpreted by expanding each packed value into a conventional wide scalar before computation. Candidate circuits may operate on packed state using bitwise mixing, lane permutations, lookup transformations, subgroup operations, bit-sliced computation, or other structures suited to the target GPU.

These are research hypotheses, not assumed optimizations. Their value depends on whether they reproduce source behavior with lower total cost on real hardware, including representation conversion, state movement, synchronization, and accumulated approximation error.

## Existing Models as Behavioral Sources

An existing transformer can be compiled while initially treating its per-block KV tensors as the transient circuit. This preserves the model's behavior while introducing the stream-oriented runtime, persistent state ownership, explicit ports, and event-driven lifecycle.

This form may retain append-only state and growing sequence cost, but it provides a practical way to exercise the engine architecture with existing weights.

Compatibility does not require permanent fidelity to the transformer's implementation. Exact lowering can provide a reference circuit, after which behavioral compilation can replace individual operations, layers, groups of layers, state representations, or the entire model while measuring equivalence against the source.

The project does not require training a new model architecture. The existing model and weights remain the behavioral authority. Any learned approximation, state reduction, or synthesized replacement belongs to the compilation process and must be validated against that source behavior.

## Core Definition

The engine's central abstraction is:

```text
running stream = compiled permanent circuit + mutable transient circuit
```

The permanent circuit supplies learned processing capability. It implements the behavior of the source model but need not preserve the source model's calculations or structure. The transient circuit is created and reshaped by the stream's experience and need not preserve the source model's state representation.

Together they form a persistent, interruptible, continuously addressable inference process.

```text
compiled model != optimized copy of the original calculations

compiled model = circuit implementing equivalent stream behavior
```

The complete system has four defining objects:

```text
source oracle
    + behavioral compiler
    + stream-circuit IR
    + Vulkan runtime
```

The source oracle defines the behavior. The compiler discovers another realization. The IR describes its stateful signal graph. The runtime installs and executes the closed loop on broadly available GPU hardware.

At the graph level:

```text
LLM = editable pedalboard graph

layer = reusable stateful pedal instance

compiled model = behaviorally equivalent realization
                 of the pedalboard graph
```
