# Automata Lab architecture decision

- **Status:** Accepted direction; staged implementation
- **Date:** 2026-08-18
- **Scope:** Cellular-automata execution, experiment lineage, and result representation

## Implementation status

- Stage 1 is complete: the legacy image-conditioned evaluator is a UI-free numeric engine and its final state is retained independently from rendering.
- Stage 2A established strict versioned program documents with the real legacy and discrete-map models plus the histogram initializer. Lenia was added only when its real engine became available; no placeholder model was serialized.
- Discrete execution currently supports direct, aggregate rounded-mean/sum, ordered-neighbourhood, and explicitly reduced presence-triggered maps; compact `u8`/`u16`/`u32` buffers; checked dense tables; synchronous parallel stepping; fixed-point and period-two detection; and opt-in budgeted exact cycle detection.
- The canonical histogram quantizer and calibrated generation-zero bridge were pulled forward from Stage 3 because they are required to compile a discrete program without allowing its initializer and state cardinality to drift apart.
- Stage 2B is complete: `ContinuousLenia` is a real strict program variant with calibrated continuous initialization, a dense anchored kernel, explicit normalization and boundary policies, built-in growth functions, synchronous parallel evolution, clamped numeric results, and convergence diagnostics.
- Continuous execution is autonomous after generation zero and contains neither source-evidence gating nor an independent decay term. Exact fixed points may be safely fast-forwarded; approximate convergence is diagnostic only and never shortens the requested run.
- Stage 3 is complete at the engine boundary. Versioned projections derive normalized scalar fields, discrete-state membership, masks, and RGBA overlays from any validated final result without rerunning a model. The normal legacy overlay now uses this projection path with its original palette and square-root alpha transfer.
- Versioned run records bind an immutable prepared-plane identity, source identity, calibration, experiment/ROI lineage, complete program document, model-specific run specification, optional deterministic seed, and validated numeric result. Record creation revalidates the borrowed program under explicit compile limits and rejects result dimensions, state cardinality/domain, cycle counts, or diagnostics that disagree with the recorded program and run specification.
- The first Stage 4 interactive slice is complete for continuous Lenia. A dedicated Automata Lab view retains the selected image's calibrated prepared plane, initializes generation zero from that plane, renders the current numeric state as a disposable superimposed projection, and exposes Reset, background-threaded Step, timed Play/Pause, diagnostics, budgeted in-memory timeline checkpoints, and named state bookmarks.
- The consolidation pass removed the Lab's parallel model format and runner. A session now owns the same strict `AutomataProgramDocument` and compiled dispatcher used at the engine boundary; both discrete-map and continuous-Lenia programs run through that session, while the current UI remains a continuous-Lenia authoring facade. Editable Lenia controls are an unpersisted adapter that produces the shared document.
- Execution policy and capture policy are separate from model identity. Checkpoints are captured at exact generation multiples even when one UI step crosses several intervals; changing capture cadence does not reset or recompile the automaton. State buffers and dense compiled tables are shared across session clones where possible and history is bounded by an explicit byte budget.
- Lab computation is cooperatively cancellable at every generation and commits atomically. Resetting, changing settings, or changing the prepared source marks the single outstanding worker result for discard and signals cancellation; a second worker is not started until the first exits.
- Lab sessions are invalidated when image selection or preprocessing changes, so a state cannot silently outlive the prepared plane that initialized it. Editing transition settings pauses playback and requires an explicit apply-and-reset operation. Persisted Lab experiments, discrete rule authoring controls, temporal result projections, and ROI child tabs remain subsequent Stage 4/5 work.

## Context

VCEM currently exposes an image-conditioned cellular feature filter in the normal analysis workflow. The intended product also needs a separate environment for constructing, inspecting, and eventually searching discrete and continuous automata. Those two experiences should share an execution framework without forcing exploratory controls into the normal image-analysis path.

The source image is environmental input, not a permanent forcing term. It is mapped once into the automaton's state space to create generation zero. A run then evolves autonomously, except when reproducing the existing evidence-gated model.

## Decision

### Workflow boundary

The normal workflow will offer a compact discrete-CA feature suitable for routine image processing and will continue to render its result as an image overlay. The **Automata Lab** will be a separate workflow for authoring rules and kernels, stepping and replaying simulations, inspecting numeric results and diagnostics, and branching experiments from regions of interest (ROIs).

Both workflows use the same state initialization, automata engine, result, and projection interfaces. The normal workflow is a constrained facade over those interfaces; it is not a second CA implementation.

One experiment has exactly one transition model and one compatible state representation. Discrete, continuous, and legacy modalities are not mixed during a run. A later composition workflow may combine the immutable outputs of independently validated models.

### Common execution model

Every run has the following stages:

1. `Initializer` maps calibrated source intensities to generation-zero state.
2. `SensingModel` samples the current state with an explicitly ordered kernel and boundary policy.
3. One `TransitionModel` produces the next state.
4. `AutomataEngine` repeats the transition for the configured cycle count and records requested checkpoints and metrics.
5. One or more `OutputProjection`s derive masks, scalar fields, or renderable overlays from the numeric result.

The input image is unavailable to steps 2-4 after initialization unless the selected model is the explicitly named legacy compatibility model. Reset is deterministic for a fixed source, calibration, initializer, program, and random seed.

### State initialization

An initializer is serialized independently from the transition model. At minimum it records the intensity calibration and one of:

- discrete histogram quantization, where the configured histogram bins are the CA states;
- another explicit discrete quantizer; or
- continuous normalization into the configured continuous interval, initially `[0, 1]`.

Histogram quantization lets a modality operate over `q = bin_count` symbols instead of the raw intensity domain. Calibration is inherited by ROI children so that the same intensity maps to the same state in parent and child experiments. A child may deliberately fork its calibration, but that change must be recorded in its lineage.

The implemented calibration currently describes the prepared integer plane. Float-TIFF ROIs must therefore crop or replay the parent's prepared plane. Persisting the exact raw-float normalization window, resampling, and preprocessing transform is required before independently decoding child crops.

Discrete state identifiers use a validated `u32` logical representation. Implementations may store a field or rule table as `u8`, `u16`, or `u32` according to `q`, including histogram counts well above eight; no API assumes a tiny state count. Conversion to a narrower representation must be checked and lossless.

### Transition models

`TransitionModel` is a versioned tagged variant, initially containing:

- `DiscreteMap`: an autonomous finite-state automaton;
- `ContinuousLenia`: an autonomous continuous-state model; and
- `LegacyEvidenceGrowth`: the current image-conditioned behaviour for saved-configuration and result compatibility.

The variant tag and version are part of the program hash. Model-specific parameters do not leak into the other variants.

#### Discrete maps

Discrete rules support these first-class rule encodings:

- **Direct state map:** sensing reduces the neighbourhood to one symbol, then a dense `q`-entry table implements `M: S -> S`.
- **Aggregate lookup:** position-independent statistics such as the rounded mean or sum select a dense transition table. This is the standard totalistic family and matches the `totalistic` branch of the older prototype, which rounded the neighbourhood mean before lookup.
- **Ordered full-neighbourhood lookup:** the `k` sampled symbols are encoded in a fixed kernel order as a base-`q` index, `i = sum(s[j] * q^j)`, into a dense `q^k`-entry table. This matches the non-totalistic branch of the older prototype and can express positional and directional bias.
- **Presence-triggered lookup:** sample position is discarded and the rule responds when mapped states are present anywhere in the sensed cells. The serialized rule must define how simultaneous matches are reduced; it may not depend on incidental scan order.

These are explicit enum-like modes, not a `totalistic: bool`. The proposed UX terminology uses “totalistic” for a rich/position-sensitive option in places, while standard CA terminology and the old implementation use “totalistic” for an aggregate that removes position. The intended behaviours are retained, but their final user-facing names require an explicit contract. Until then, UI and persistence should use the unambiguous names **Aggregate**, **Ordered neighbourhood**, and **Presence triggered**.

State fields, transition tables, and scratch buffers are dense one-dimensional arrays. Images use a documented row-major index; kernel sample order is stable and serialized. An ordered lookup is accepted when `q^k` can be computed without integer overflow and its exact byte size fits the configured allocation budget and addressable memory. Validation uses checked exponentiation, multiplication, and indexing. There is no arbitrary small-`q` or small-k prohibition: feasibility is determined from the chosen symbol width and actual resource limits. Compilation occurs before the frame loop, and a step performs no per-cell heap allocation. These properties preserve the performance strategy demonstrated by the prototype and leave 60 fps as a benchmark target, not an unconditional guarantee for every image, kernel, and machine.

#### Continuous Lenia

The initial continuous model is autonomous Lenia-style evolution:

`a(t + dt) = clamp(a(t) + dt * G(K * a(t)))`.

The image supplies `a(0)` only. The model has an explicit kernel, kernel normalization policy, boundary policy, growth response `G`, time step, clamp/domain policy, and cycle count. It has no independent linear decay term: suppression or extinction must emerge from the growth response and repeated transitions.

The implemented v1 kernel is a dense row-major rectangle with an explicit anchor. It can be left unnormalized, normalized by its signed sum, or normalized by its L1 norm. Clamp, reflect, wrap, and constant boundary policies are supported. V1 growth families are Gaussian, band, and linear; the canonical Lenia Gaussian is represented as a parameterization of the Gaussian family rather than a separate evaluator.

Initial authoring uses built-in, serialized response families and parameters. A later authoring stage may add piecewise curves and a whitelisted mathematical expression language. Expressions must compile to a validated, deterministic representation with bounded evaluation and no I/O, loops, allocation, or arbitrary host-code execution.

#### Legacy compatibility

`LegacyEvidenceGrowth` preserves the current positive-response evidence gate and the interpretation of already saved settings. It is the only transition model allowed to consult the immutable source-derived evidence after generation zero. If exact reproduction requires the existing decay parameter, that parameter remains private to this versioned compatibility variant; it is not exposed by new discrete or Lenia programs. New work should not use this variant as the basis of the Automata Lab model design.

### Numeric results and rendering

The canonical result is numeric rather than an RGBA texture. A `SimulationResult` records:

- final state and optional requested checkpoints;
- dimensions, state/domain metadata, cycle count, and dynamics metrics;
- source, calibration, initializer, program, and random-seed identities; and
- enough provenance to reproduce the run.

Rendering is a downstream projection. An `OutputProjection` can reproduce the current superimposed overlay, but changing its palette, threshold, opacity, or selected states does not rerun or mutate the automaton. Numeric projections also provide the future inputs for validated-model composition, including masks, final activation, temporal maxima, occupancy, and change frequency.

Projection v1 implements normalized final-state fields for all three model families, discrete-state membership, threshold masks, and configurable single-colour RGBA overlays. Transfer functions are identity, power, and linear windowing. Timeline checkpoints now exist in the interactive Lab, but converting captured histories into persisted temporal projections remains deferred.

### Experiments and ROI lineage

An `Experiment` owns one source reference, initializer, automata program, run specification, annotations, and result revisions. An ROI action creates a child experiment rather than an unrelated crop. The child records:

- parent experiment and result revision;
- source-image identity and parent-coordinate bounds;
- inherited calibration and preprocessing provenance;
- any context halo and the inner scoring bounds; and
- subsequent program, annotation, and result revisions.

This lineage makes manual and automated selection experiments auditable and allows successful rules to be evaluated against other regions or images without losing their origin.

The implemented run-provenance record stores this lineage together with stable source and prepared-plane identifiers. The prepared-plane identifier is the reproducibility boundary for current float-TIFF inputs: independently decoding a crop remains invalid until the raw normalization and preprocessing transform are structured rather than merely identified.

### Evolution and composition

Quality-diversity and evolutionary search are later clients of the same serialized programs and pure engine. A search genome may vary supported discrete tables, kernels, or continuous response parameters; fitness consumes explicit preserve, suppress, and ignore annotations plus separately reported behavioural metrics. Search returns reproducible candidate programs and numeric results, not a special runtime model.

Model composition is also deferred. It operates on projections from validated results and never injects one modality into another modality's in-progress state. Composition provenance records every source result and operation.

## Proposed core interfaces

The names are architectural roles rather than a commitment to a particular Rust module layout.

```text
Experiment
  source: SourceRef
  lineage: ExperimentLineage
  initializer: Initializer
  program: AutomataProgram
  run: RunSpec

AutomataProgram
  state_space: Discrete(q, storage) | Continuous(domain)
  sensing: SensingModel
  transition: DiscreteMap | ContinuousLenia | LegacyEvidenceGrowth
  boundary: BoundaryPolicy
  version: ProgramVersion

AutomataEngine
  initialize(source, calibration, initializer) -> StateBuffer
  compile(program, limits) -> CompiledAutomaton
  run(compiled, initial_state, run_spec) -> SimulationResult

OutputProjection
  project(result, projection_spec) -> NumericField | Mask | Overlay
```

The compiled transition API conceptually reads `current` and writes `next`; the buffers never alias. Parallel execution must therefore produce the same state as serial execution for deterministic models.

## Invariants

- The source image maps into state exactly once, except in the named legacy model.
- A run contains exactly one transition-model variant and a compatible state space.
- State cardinality, symbol width, table length, byte size, and every derived index are validated before execution.
- Boundary policy, kernel sample order, normalization, random seed, and transition version are explicit and reproducible.
- Simulation state and results remain numeric; overlays are disposable projections.
- ROI children retain parent coordinates, calibration, and revision lineage.
- Modality composition consumes completed, validated results and is not part of an experiment's transition loop.
- New autonomous models contain no hidden evidence gate or independent decay.

## Delivery sequence

1. Extract a pure CA engine that returns numeric state independently of egui and texture rendering.
2. Add versioned `LegacyEvidenceGrowth`, `DiscreteMap`, and `ContinuousLenia` variants.
3. Add initializer/calibration, histogram-bin states, boundary policy, deterministic seeds, and output projections.
4. Add Reset/Step/Play, timeline checkpoints, diagnostics, and named state bookmarks in Automata Lab.
5. Add child ROI experiment tabs, context halos, lineage, and result-layer composition foundations.
6. Add curve sketching and then the whitelisted expression language.
7. Add preserve/suppress/ignore rewards, exhaustive scans where tractable, and quality-diversity/evolutionary search.

## Consequences

The normal workflow remains approachable while Automata Lab can grow without duplicating simulation logic. Legacy outputs remain reproducible, and new discrete and continuous models receive autonomous semantics. Numeric results and explicit lineage add storage and migration work, while dense `q^k` tables require careful resource validation; those costs are necessary for auditable comparison, high-performance execution, ROI branching, later composition, and automated search.
