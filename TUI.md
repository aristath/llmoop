# llmoop TUI

## Purpose

The model editor is the main feature and primary workspace of the llmoop TUI. It is not an auxiliary configuration page attached to a chat client. The TUI is the visible patch bay for the inference engine: it loads an existing compiled pedal kit or transpiles a source model into one, constructs a runtime pedalboard, places pedal instances on available devices, and lets the user inspect and edit that board.

Chat, runtime statistics, logs, and device information support this workspace. They must not displace the pedalboard editor as the application's central surface.

The TUI will be implemented in Rust and must be fully operable with either the keyboard or the mouse.

## Framework

Use [Ratatui](https://ratatui.rs/) for layout and rendering, with [Crossterm](https://docs.rs/crossterm/latest/crossterm/event/) as the terminal event backend.

This combination fits the editor because:

- Ratatui permits custom and stateful widgets instead of restricting the application to conventional forms and lists.
- The pedalboard can render each pedal and cable directly into the terminal buffer.
- Crossterm reports keyboard, mouse, paste, focus, and terminal-resize events.
- Crossterm supports explicit mouse capture.
- The application can record the terminal rectangle occupied by every rendered pedal and use those rectangles for precise mouse hit testing.

Ratatui has an [official interactive custom-widget example](https://ratatui.rs/examples/widgets/custom_widget/) that includes mouse handling, as well as documentation for its [custom widget model](https://ratatui.rs/recipes/widgets/custom/).

Ratatui is an immediate-mode rendering library rather than a complete retained GUI toolkit. llmoop will therefore own application state, focus, semantic actions, modal behavior, scrolling, and hit testing. This is appropriate for a specialized pedalboard editor because those behaviors are part of the product rather than generic form behavior.

## Launching the editor

Running `python -m llmoop` without an action opens the TUI. During source development the editor can also be started directly with:

```text
cargo run --manifest-path runtime-rs/Cargo.toml \
  --features vulkan,tokenizers,tui --bin llmoop-tui
```

`LLMOOP_TUI_BIN` may point the Python launcher at an installed TUI executable. `LLMOOP_WORKSPACE`, `LLMOOP_PYTHON`, and `LLMOOP_COMPILER_BIN` can override the compiler-client launch boundary without putting model-family knowledge in the TUI.

## Core terminology

The TUI must preserve the distinctions defined in `CONCEPT.md`:

- A **source pedal** is a reusable definition supplied by the compiled model package.
- A **pedal instance** is one mounted occurrence of a source pedal on the runtime board.
- The **pedalboard** is the editable runtime patch containing instances and cables.
- **Placement** assigns a pedal instance to a runtime-discovered physical device.
- A **cable** transports a signal between pedal instances. Device placement may change the cable's transport without changing the logical connection.
- The **transient state** belongs to a running stream and, by default, to an individual stateful pedal instance.

The compiled package provides the source pedal kit and canonical wiring. The TUI edits a runtime patch. It does not modify or recompile the source model merely to reorder, duplicate, bypass, or place pedals.

## Model loading and transpilation

The TUI must support both entry paths into the editor:

1. Load an existing llmoop compiled model package.
2. Select a Safetensors source model, transpile it for llmoop, and load the resulting package.

Model loading is part of the primary application workflow. It must be available from the empty application state and remain available from the model editor so the user can replace the current model.

```text
┌ Open model ──────────────────────────────────────────────────────────────┐
│                                                                         │
│ Source: [/home/user/models/model-name/                              ]    │
│                                                                         │
│ Detected: Safetensors source model                                      │
│ Config:   config.json                                                   │
│ Weights:  4 shards                                                      │
│ Tokenizer and chat template: available                                  │
│                                                                         │
│                [ Transpile and load ]  [ Cancel ]                       │
└─────────────────────────────────────────────────────────────────────────┘
```

The source selector must be usable as a path field and as a keyboard-and-mouse navigable filesystem browser. Pasting a path must work. The TUI should identify whether the selected path is an llmoop package or a source model and present the appropriate action explicitly: `Load model` or `Transpile and load`.

A Safetensors model is normally more than a weights file. The compiler may also require configuration, tokenizer assets, a chat template, weight-index metadata, or other source artifacts. The TUI passes the selected source to the compiler's discovery API and displays what the compiler found. It must not infer model architecture or required artifacts itself.

### Compiler boundary

The TUI is a compiler client, not a second implementation of the compiler. Model-family discovery, architecture recognition, circuit lowering, artifact generation, and validation remain compiler responsibilities. The TUI calls the same stable compiler interface used by other frontends such as the CLI.

No LFM-, Qwen-, Gemma-, Mamba-, transformer-, or repository-name checks belong in the TUI. The compiler returns structured discovery information, progress, diagnostics, and the completed package identity. The TUI renders that information generically.

The interface should not depend on scraping human-oriented compiler log text. The compiler boundary should expose structured events such as:

```text
DiscoveryStarted
SourceDiscovered
ValidationStarted
PedalLoweringStarted { current, total, pedal_id }
ArtifactWritingStarted
PackageValidationStarted
Completed { package }
Failed { diagnostics }
Cancelled
```

The exact event names are an implementation concern, but progress and errors must have machine-readable structure.

### Transpilation progress

Transpilation may take long enough that it cannot block the terminal event loop. The user must still be able to resize the terminal, inspect progress, scroll diagnostics, or cancel the operation.

```text
┌ Transpiling model ───────────────────────────────────────────────────────┐
│ Qwen example                                                            │
│                                                                         │
│ Discovering structure                         complete                  │
│ Validating source artifacts                   complete                  │
│ Lowering pedals                               17 / 32                   │
│ ███████████████████████░░░░░░░░░░░░░░░░░░░░                            │
│ Current: layer_16                                                      │
│                                                                         │
│                                                    [ Cancel ]            │
└─────────────────────────────────────────────────────────────────────────┘
```

Progress must report real compiler stages and completed work. The TUI must not fabricate a percentage when the compiler cannot calculate one.

Cancellation must leave no package that can be mistaken for a valid completed model. Package publication should be atomic: the result becomes loadable only after compiler validation succeeds.

On successful transpilation, the TUI immediately loads the completed package and opens its canonical pedalboard in the main editor. The user should not have to locate and reopen the generated package manually.

### Loading failures

Failure returns the user to the model-selection surface with the chosen path intact and diagnostics available. Errors should identify the actual missing or unsupported item, for example:

```text
Cannot transpile this source: tokenizer_config.json is missing.
Unsupported circuit operation in layer_12: selective_scan_v3.
Compiled package validation failed: pedal layer_07 has no output contract.
```

The TUI must not silently fall back to a different model, substitute a different architecture, or load a partially generated package.

## Primary workspace

The application opens into the model editor and gives the pedalboard most of the terminal area.

```text
┌ Model: LFM2.5-230M ───────────────────────────────────────────────────────┐
│ Layer order: [0,1,2,3,4,5,5,6,7,7,8,9,10,11,12,13]                     │
├ Pedalboard ───────────────────────────────────────────────────────────────┤
│                                                                          │
│  [0]──[1]──[2]──[3]──[4]──[5]──[5²]──[6]──[7]──[7²]──[8]──[9] ...     │
│  GPU0 GPU0 GPU0 GPU0 GPU0 GPU0 CPU  CPU  GPU1 GPU1 GPU1 GPU1             │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│ Enter: edit layer   Arrows: navigate   Click: edit   Tab: change region  │
└──────────────────────────────────────────────────────────────────────────┘
```

The exact layout must adapt to the terminal dimensions, but the priority remains stable:

1. Model and board identity.
2. The layer-order editor.
3. The live pedalboard visualization.
4. Contextual actions, validation, and status.

The editor must remain useful in a small terminal. A small viewport may scroll, collapse secondary metadata, or show a more compact pedal shape, but it must not remove the ability to edit the sequence, select a pedal, open its modal, or identify its device.

## Layer-order field

The layer-order field provides a direct, simple way to define the serial order of layer pedals:

```text
[0,1,2,3,4,5,5,6,7,7,8,9,10,11,12,13]
```

The field is a real text editor, not a sequence of single-character shortcuts. It must support normal cursor movement, insertion, deletion, selection where the terminal permits it, and bracketed paste.

The field and the visual pedalboard are two interfaces to the same board state:

- Editing a valid sequence immediately updates the visual board.
- A visual reorder, insertion, removal, or duplication updates the sequence field.
- Neither surface owns a second, divergent representation of the board.

The parser should accept insignificant whitespace around brackets, commas, and identifiers. More expressive syntax can be introduced when it has a concrete editing purpose; the literal list remains the clearest canonical form for a serial board.

### Editing and validation

Text is often temporarily invalid while a user is typing. The editor must therefore distinguish the text buffer from the last valid parsed board:

```text
text buffer
    |
    +-- valid ----> replace board draft ----> rerender visualization
    |
    `-- invalid --> retain last valid board -> show precise inline error
```

An incomplete bracket, missing comma, or unknown layer must not make the pedalboard disappear or partially mutate. The error must identify the problematic position and explain the correction, for example:

```text
Unknown layer `14` at column 14. Available layers: 0-13.
```

## Pedal instances and duplication

Every occurrence in the sequence creates a distinct pedal instance, even when several occurrences reference the same source pedal.

For example:

```text
[0,1,2,1,3]
```

contains two instances of source pedal `1`. Internally they require stable and distinct identities, such as:

```text
layer_01@1
layer_01@2
```

The display may use a compact occurrence marker such as `1²`, but an accessible textual label must remain available, for example `Layer 1, occurrence 2`.

Duplicated instances may reference the same immutable compiled circuit and parameters. They do not implicitly share transient state, placement, or controls. Each instance can have:

- its own physical-device assignment;
- its own enabled or bypassed state;
- its own runtime-editable properties; and
- its own declared transient-state policy.

State sharing or cloning must always be explicit. Duplicating a pedal must not accidentally duplicate or share state as an incidental UI side effect.

## Live pedalboard visualization

The visual area displays the instantiated signal path, one pedal at a time, in execution order. The default serial board uses a clear cable between adjacent pedals. Long boards are scrollable and may wrap only when continuity between rows remains unambiguous.

Each pedal should communicate, within the available space:

- source layer or pedal identifier;
- occurrence when the source pedal is duplicated;
- pedal kind when useful;
- assigned physical device;
- selected and focused state;
- enabled, bypassed, invalid, or unavailable state; and
- validation or compatibility warnings.

The board should spend its visual emphasis on the signal chain itself. Boxes, cables, and instance identity encode real topology; decoration that does not communicate state or structure should be avoided.

The signature visual element is the live cable path. It should make the execution order immediately legible and make placement boundaries visible without turning the screen into a generic dashboard. A cable crossing between devices may change its line treatment or include a transport marker, but color must never be its only distinguishing property.

### Navigation

There is exactly one logical board selection, regardless of input method.

- Left and right select the previous or next pedal in a serial chain.
- Directional navigation follows visible topology when the board contains branches.
- Enter opens the selected pedal's editor.
- Clicking a pedal selects it and opens its editor.
- Scrolling pans through a board larger than its viewport.
- Tab and Shift-Tab move between major focus regions.
- Escape closes the current modal or returns focus to the board.
- Home and End move to the first and last pedal where that meaning is unambiguous.

Keyboard selection and mouse selection must update the same selected-instance identifier. They must not have separate behavioral paths.

### Hit testing

During every render, the board widget records the `Rect` occupied by each visible pedal and interactive cable or control. Mouse events resolve their coordinates against this render map. The render map is regenerated after scrolling, resizing, topology changes, or density changes, so it cannot retain stale terminal coordinates.

## Pedal modal

Opening a pedal presents a modal for the selected pedal instance, not merely its shared source definition.

```text
┌ Layer 1 · occurrence 2 ──────────────────────┐
│ Source pedal: layer_01                       │
│ Type: transformer                            │
│                                              │
│ Device:       [ Vulkan GPU 1             ▼ ] │
│ Enabled:      [✓]                            │
│ State policy: [ Independent              ▼ ] │
│                                              │
│ Properties                                   │
│ Attention window: [ 4096                   ] │
│ Precision:        [ BF16                  ▼ ] │
│                                              │
│              [ Apply ]  [ Cancel ]            │
└──────────────────────────────────────────────┘
```

The modal may expose:

- source-pedal identity and instance occurrence;
- physical-device assignment;
- enabled or bypassed state;
- transient-state policy where applicable;
- an optional instance label;
- runtime-editable controls declared by the pedal; and
- compatibility, capacity, or remount warnings.

`Apply` changes the board draft and immediately updates the visualization. `Cancel` discards the modal's uncommitted changes. The interaction for applying a changed draft to an already mounted running stream remains a separate runtime-lifecycle decision; it must not be silently conflated with editing the draft.

When a source pedal has several instances, the modal edits only the selected instance by default. Any later operation that applies a change to every occurrence must say so explicitly.

## Schema-driven properties

The TUI must not hardcode controls for LFM, Qwen, Gemma, transformers, Mamba, or any other model family or circuit implementation.

The compiled pedal package declares the editable control schema for each source pedal. A property definition should provide enough metadata for a generic TUI to render and validate it, including:

- stable property identifier;
- user-facing name and description;
- value type;
- current and default values;
- valid range, step, enumeration, or other constraint;
- units where applicable;
- whether it is editable at runtime;
- whether changing it requires transient-state reset;
- whether changing it requires remounting or recompiling the pedal; and
- whether the property applies to an instance or to the shared source definition.

The TUI selects an appropriate generic editor from that schema: toggle, numeric input, slider, enumeration list, text input, or read-only value. Unknown property types must be displayed safely as unsupported metadata rather than ignored or guessed.

## Runtime device placement

Physical devices are discovered at runtime. They are not hardcoded in the TUI, compiler, compiled package, or a mandatory placement JSON file.

The device selector shows the devices currently reported by the runtime together with information needed to make a placement decision, such as:

- stable runtime identifier;
- human-readable device name;
- backend and device kind;
- supported pedal or kernel capabilities;
- relevant memory capacity and current availability; and
- availability or connection state.

CPU support and Vulkan devices participate through the same runtime device abstraction. Other device transports, including LAN devices, can be presented through the same selector when their runtime backends exist.

Changing an instance's placement must not change its source pedal or logical position. It changes where the instance is hosted and may change the transport used by its incoming and outgoing cables.

If a device cannot host the selected pedal, the TUI must explain the concrete incompatibility. It must not silently move the pedal elsewhere or present unavailable devices as valid choices.

## Unified interaction model

Keyboard and mouse events map to semantic application actions before they mutate editor state:

```text
keyboard event ----+
                   +--> semantic action --> editor state --> render
mouse event -------+
```

Representative actions include:

- `OpenModelSelector`
- `LoadCompiledModel`
- `StartModelTranspilation`
- `CancelModelTranspilation`
- `FocusNextRegion`
- `SelectPreviousPedal`
- `SelectNextPedal`
- `OpenSelectedPedal`
- `SetInstanceDevice`
- `SetInstanceProperty`
- `ApplyModalChanges`
- `CancelModalChanges`
- `ReplaceBoardSequence`
- `PanBoard`

This command layer prevents mouse and keyboard support from becoming two implementations of the editor. It also makes actions testable without a physical terminal.

Text entry remains context-sensitive: printable input edits the order field or the focused modal control instead of triggering global shortcuts.

## Accessibility requirements

Keyboard support is a first-class interaction method, not a fallback.

- Every mouse action has a keyboard equivalent.
- Focus is always visible.
- Selection and focus remain distinguishable.
- Meaning is never conveyed by color alone.
- Device placement includes a textual label even when devices also have colors.
- Duplicated pedals have textual occurrence labels.
- Modal focus is trapped inside the modal until it closes.
- Escape behavior is consistent and never discards committed edits.
- Help text reflects the currently focused region instead of displaying irrelevant global shortcuts.
- Mouse capture can be disabled so users and terminals that do not support it remain fully functional.
- Terminal resize preserves the selected pedal and scrolls it back into view where possible.

## State model

The TUI should maintain explicit state rather than deriving behavior from rendered labels:

```text
application
|-- model source selection
|-- compiler job and structured progress, if active
|-- loaded compiled package
|-- runtime-discovered devices
|-- board draft
|   |-- pedal instances with stable IDs
|   |-- cables
|   |-- per-instance placement
|   `-- per-instance control values
|-- mounted board identity and status, if any
|-- layer-order text buffer
|-- last valid parsed sequence
|-- focus and selected instance
|-- viewport and scroll positions
|-- open modal and uncommitted modal values
|-- current validation results
`-- current render hit map
```

The board draft is the authoritative editable runtime patch. The order field and visual chain are projections and editing surfaces over that state. Stable instance IDs must survive rerenders and should survive sequence edits where an occurrence can be matched unambiguously, so that placement and property edits are not needlessly lost.

## Validation and failure behavior

Validation occurs at the level where the problem exists:

- The sequence parser validates syntax and source-pedal references.
- The graph validator validates ports, dimensions, cycles, delays, and required adapters.
- The placement validator checks device capability and availability.
- The property validator checks schema constraints.
- The runtime validates whether a draft can be mounted or applied to a running stream.

Errors should say what failed and how to correct it. Examples:

```text
Layer 12 cannot run on GPU 1: the device lacks the required shader feature.
Connection layer_05@1 -> layer_08@1 is incompatible: expected width 2048, got 1024.
Layer 2 occurrence 2 needs a state policy before the duplicated board can be mounted.
```

The editor must not silently insert adapters, change placement, discard instance state, or repair topology unless the user invokes a clearly named operation that performs that change.

## Visual direction

The interface should feel like a technical signal-chain instrument rather than a generic administrative dashboard.

- Use the terminal's normal background as the quiet field.
- Give the live cable and current signal path the strongest contrast.
- Use restrained device accents, always paired with labels or line patterns.
- Use borders to express real boundaries: pedals, modal scope, focus region, and device transitions.
- Prefer compact engineering labels over decorative headings.
- Keep status and help text quiet until it is relevant.
- Avoid dense collections of unrelated panels around the board.

The distinctive element is not a decorative color scheme. It is the live, editable signal path whose pedals visibly carry identity, placement, and state.

## Settled public convention

The layer-order field uses zero-based numeric indices, for example `[0,1,2,1,3]`. The field does not repeat a ceremonial `layer_` prefix. Internal source-pedal and instance IDs remain stable model-package identities such as `layer_01` and `layer_01@2`; they are shown as metadata where that identity matters, but are not mixed into the numeric order field.

## Remaining runtime decisions

### Running-board edits

The runtime contract must define whether a board draft can be applied while a stream is active, which edits require a remount, and what happens to transient state. Until that contract exists, the TUI must distinguish editing a draft from changing an already mounted board.

### General graph editing

The serial layer-order field is the clearest editor for the common chain. The underlying board state must still support the graph operations defined in `CONCEPT.md`, including parallel paths, mixers, adapters, and delayed feedback. The visual interaction for creating and editing those connections requires its own deliberate design; it must extend the same board state rather than introduce a second graph representation.
