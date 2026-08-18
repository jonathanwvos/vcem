use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{fmt, sync::atomic::AtomicBool};

use crate::continuous_automata::{
    CompiledContinuousLenia, CompiledContinuousStateInitializer, ContinuousAutomataError,
    ContinuousCompileLimits, ContinuousDomain, ContinuousLeniaProgram, ContinuousRunConfig,
    ContinuousSimulationResult, ContinuousStateInitializer, compile_continuous_lenia,
    run_continuous_lenia, run_continuous_lenia_with_cancellation,
    validate_continuous_lenia_program,
};
use crate::discrete_automata::{
    CompactState, CompileLimits, CompiledDiscreteAutomaton, DiscreteAutomataError, DiscreteProgram,
    DiscreteRunConfig, DiscreteSimulationResult, compile_discrete_program, run_discrete_automaton,
    run_discrete_automaton_with_cancellation, validate_discrete_program,
};
use crate::quantization::{
    CompiledHistogramStateInitializer, HistogramStateInitializer, IntensityCalibration,
};

pub const AUTOMATA_PROGRAM_SCHEMA_VERSION: u32 = 1;
pub const LEGACY_EVIDENCE_GROWTH_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellularAutomataSeedMode {
    Bright,
    Dark,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellularAutomataResponse {
    Gaussian,
    Band,
    Threshold,
    LowThreshold,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellularAutomataSettings {
    pub kernel_size: usize,
    pub kernel: Vec<f32>,
    pub seed_mode: CellularAutomataSeedMode,
    #[serde(default)]
    pub omni_directional: bool,
    pub response: CellularAutomataResponse,
    pub response_center: f32,
    pub response_width: f32,
    pub growth_rate: f32,
    pub decay_rate: f32,
    pub time_step: f32,
    pub iterations: usize,
}

impl Default for CellularAutomataSettings {
    fn default() -> Self {
        let kernel_size = 7;
        let radius = kernel_size as f32 / 2.0;
        let center = (kernel_size / 2) as f32;
        let kernel = (0..kernel_size * kernel_size)
            .map(|index| {
                let x = (index % kernel_size) as f32 - center;
                let y = (index / kernel_size) as f32 - center;
                if x * x + y * y <= radius * radius {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        Self {
            kernel_size,
            kernel,
            seed_mode: CellularAutomataSeedMode::Dark,
            omni_directional: false,
            response: CellularAutomataResponse::Gaussian,
            response_center: 0.65,
            response_width: 0.12,
            growth_rate: 1.0,
            decay_rate: 0.10,
            time_step: 0.15,
            iterations: 24,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CellularAutomataSummary {
    pub iterations_run: usize,
    pub converged: bool,
    pub mean_activation: f32,
    pub active_fraction: f32,
}

/// The numeric output of a completed cellular automata simulation.
///
/// `state` is row-major and has exactly `size[0] * size[1]` entries.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellularAutomataResult {
    pub size: [usize; 2],
    pub state: Vec<f32>,
    pub summary: CellularAutomataSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyEvidenceGrowthProgram {
    pub version: u32,
    pub settings: CellularAutomataSettings,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscreteMapProgram {
    pub initializer: HistogramStateInitializer,
    pub program: DiscreteProgram,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousLeniaProgramDefinition {
    pub initializer: ContinuousStateInitializer,
    pub program: ContinuousLeniaProgram,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "model",
    content = "definition",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum AutomataModelDefinition {
    LegacyEvidenceGrowth(LegacyEvidenceGrowthProgram),
    DiscreteMap(DiscreteMapProgram),
    ContinuousLenia(ContinuousLeniaProgramDefinition),
}

/// A standalone program document for Automata Lab. This is intentionally
/// separate from the existing `ExperimentConfiguration` schema-v1 document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutomataProgramDocument {
    pub schema_version: u32,
    pub transition_model: AutomataModelDefinition,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutomataCompileLimits {
    pub discrete: CompileLimits,
    pub continuous: ContinuousCompileLimits,
}

#[derive(Clone, Debug)]
pub enum CompiledAutomataProgram {
    LegacyEvidenceGrowthV1(CellularAutomataSettings),
    DiscreteMapV1(CompiledDiscreteMapProgram),
    ContinuousLeniaV1(CompiledContinuousLeniaProgram),
}

#[derive(Clone, Debug)]
pub struct CompiledDiscreteMapProgram {
    initializer: CompiledHistogramStateInitializer,
    automaton: CompiledDiscreteAutomaton,
}

#[derive(Clone, Debug)]
pub struct CompiledContinuousLeniaProgram {
    initializer: CompiledContinuousStateInitializer,
    automaton: CompiledContinuousLenia,
}

impl CompiledDiscreteMapProgram {
    pub fn initializer(&self) -> &CompiledHistogramStateInitializer {
        &self.initializer
    }

    pub fn automaton(&self) -> &CompiledDiscreteAutomaton {
        &self.automaton
    }

    pub fn compact_initial_state_from_bins(
        &self,
        calibration: IntensityCalibration,
        bin_count: u32,
        bin_ids: Vec<u32>,
        max_packed_state_bytes: u64,
    ) -> Result<CompactState, AutomataFrameworkError> {
        if calibration != self.initializer.calibration() {
            return Err(AutomataFrameworkError::CalibrationMismatch {
                expected: self.initializer.calibration(),
                actual: calibration,
            });
        }
        if bin_count != self.initializer.bin_count() {
            return Err(AutomataFrameworkError::BinCountMismatch {
                expected: self.initializer.bin_count(),
                actual: bin_count,
            });
        }
        self.initializer
            .compact_states_from_owned_bins(bin_ids, max_packed_state_bytes)
            .map_err(AutomataFrameworkError::Initializer)
    }

    pub fn compact_initial_state_from_intensities(
        &self,
        calibration: IntensityCalibration,
        intensities: &[u32],
        max_packed_state_bytes: u64,
    ) -> Result<CompactState, AutomataFrameworkError> {
        if calibration != self.initializer.calibration() {
            return Err(AutomataFrameworkError::CalibrationMismatch {
                expected: self.initializer.calibration(),
                actual: calibration,
            });
        }
        self.initializer
            .compact_states(intensities, max_packed_state_bytes)
            .map_err(AutomataFrameworkError::Initializer)
    }
}

impl CompiledContinuousLeniaProgram {
    pub fn initializer(&self) -> CompiledContinuousStateInitializer {
        self.initializer
    }

    pub fn automaton(&self) -> &CompiledContinuousLenia {
        &self.automaton
    }

    pub fn initial_state_from_intensities(
        &self,
        calibration: IntensityCalibration,
        intensities: &[u32],
        max_state_bytes: u64,
    ) -> Result<Vec<f32>, AutomataFrameworkError> {
        if calibration != self.initializer.calibration() {
            return Err(AutomataFrameworkError::CalibrationMismatch {
                expected: self.initializer.calibration(),
                actual: calibration,
            });
        }
        self.initializer
            .initialize(intensities, max_state_bytes)
            .map_err(AutomataFrameworkError::Continuous)
    }
}

pub enum AutomataRunInput<'a> {
    LegacyImage(&'a [u32]),
    DiscreteState {
        state: &'a CompactState,
        config: &'a DiscreteRunConfig,
    },
    ContinuousState {
        state: &'a [f32],
        config: &'a ContinuousRunConfig,
    },
}

impl AutomataRunInput<'_> {
    fn label(&self) -> &'static str {
        match self {
            Self::LegacyImage(_) => "legacy-image",
            Self::DiscreteState { .. } => "discrete-state",
            Self::ContinuousState { .. } => "continuous-state",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "model", content = "result", rename_all = "snake_case")]
pub enum AutomataSimulationResult {
    LegacyEvidenceGrowthV1(CellularAutomataResult),
    DiscreteMapV1(DiscreteSimulationResult),
    ContinuousLeniaV1(ContinuousSimulationResult),
}

#[derive(Debug)]
pub enum AutomataFrameworkError {
    UnsupportedDocumentVersion {
        found: u32,
    },
    UnsupportedLegacyModelVersion {
        found: u32,
    },
    InitializerProgramStateCountMismatch {
        initializer: u32,
        program: u32,
    },
    InitializerProgramDomainMismatch {
        initializer: ContinuousDomain,
        program: ContinuousDomain,
    },
    CalibrationMismatch {
        expected: IntensityCalibration,
        actual: IntensityCalibration,
    },
    BinCountMismatch {
        expected: u32,
        actual: u32,
    },
    IncompatibleInput {
        model: &'static str,
        input: &'static str,
    },
    Legacy(String),
    Initializer(String),
    Discrete(DiscreteAutomataError),
    Continuous(ContinuousAutomataError),
}

impl fmt::Display for AutomataFrameworkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDocumentVersion { found } => {
                write!(
                    formatter,
                    "Unsupported automata program schema version {found}."
                )
            }
            Self::UnsupportedLegacyModelVersion { found } => {
                write!(
                    formatter,
                    "Unsupported legacy automata model version {found}."
                )
            }
            Self::InitializerProgramStateCountMismatch {
                initializer,
                program,
            } => write!(
                formatter,
                "The initializer produces {initializer} states, but the discrete program expects {program}."
            ),
            Self::InitializerProgramDomainMismatch {
                initializer,
                program,
            } => write!(
                formatter,
                "The continuous initializer maps into {initializer:?}, but the Lenia program uses {program:?}."
            ),
            Self::CalibrationMismatch { expected, actual } => write!(
                formatter,
                "Initial-state calibration {actual:?} does not match the compiled calibration {expected:?}."
            ),
            Self::BinCountMismatch { expected, actual } => write!(
                formatter,
                "Initial-state bin count {actual} does not match the compiled bin count {expected}."
            ),
            Self::IncompatibleInput { model, input } => {
                write!(
                    formatter,
                    "The {model} model cannot run with {input} input."
                )
            }
            Self::Legacy(error) => formatter.write_str(error),
            Self::Initializer(error) => formatter.write_str(error),
            Self::Discrete(error) => error.fmt(formatter),
            Self::Continuous(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AutomataFrameworkError {}

impl From<DiscreteAutomataError> for AutomataFrameworkError {
    fn from(error: DiscreteAutomataError) -> Self {
        Self::Discrete(error)
    }
}

impl From<ContinuousAutomataError> for AutomataFrameworkError {
    fn from(error: ContinuousAutomataError) -> Self {
        Self::Continuous(error)
    }
}

pub fn compile_automata_program(
    document: AutomataProgramDocument,
    limits: AutomataCompileLimits,
) -> Result<CompiledAutomataProgram, AutomataFrameworkError> {
    if document.schema_version != AUTOMATA_PROGRAM_SCHEMA_VERSION {
        return Err(AutomataFrameworkError::UnsupportedDocumentVersion {
            found: document.schema_version,
        });
    }
    match document.transition_model {
        AutomataModelDefinition::LegacyEvidenceGrowth(program) => {
            if program.version != LEGACY_EVIDENCE_GROWTH_VERSION {
                return Err(AutomataFrameworkError::UnsupportedLegacyModelVersion {
                    found: program.version,
                });
            }
            validate_legacy_settings(&program.settings).map_err(AutomataFrameworkError::Legacy)?;
            Ok(CompiledAutomataProgram::LegacyEvidenceGrowthV1(
                program.settings,
            ))
        }
        AutomataModelDefinition::DiscreteMap(definition) => {
            let initializer = definition
                .initializer
                .compile()
                .map_err(AutomataFrameworkError::Initializer)?;
            if initializer.state_count() != definition.program.state_count {
                return Err(
                    AutomataFrameworkError::InitializerProgramStateCountMismatch {
                        initializer: initializer.state_count(),
                        program: definition.program.state_count,
                    },
                );
            }
            let automaton = compile_discrete_program(definition.program, limits.discrete)?;
            Ok(CompiledAutomataProgram::DiscreteMapV1(
                CompiledDiscreteMapProgram {
                    initializer,
                    automaton,
                },
            ))
        }
        AutomataModelDefinition::ContinuousLenia(definition) => {
            let initializer = definition.initializer.compile()?;
            if initializer.domain() != definition.program.domain {
                return Err(AutomataFrameworkError::InitializerProgramDomainMismatch {
                    initializer: initializer.domain(),
                    program: definition.program.domain,
                });
            }
            let automaton = compile_continuous_lenia(definition.program, limits.continuous)?;
            Ok(CompiledAutomataProgram::ContinuousLeniaV1(
                CompiledContinuousLeniaProgram {
                    initializer,
                    automaton,
                },
            ))
        }
    }
}

/// Validates a program document without consuming or cloning its potentially
/// large dense tables. Run-record binding uses this path before accepting
/// provenance supplied independently from execution.
pub fn validate_automata_program_document(
    document: &AutomataProgramDocument,
    limits: AutomataCompileLimits,
) -> Result<(), AutomataFrameworkError> {
    if document.schema_version != AUTOMATA_PROGRAM_SCHEMA_VERSION {
        return Err(AutomataFrameworkError::UnsupportedDocumentVersion {
            found: document.schema_version,
        });
    }
    match &document.transition_model {
        AutomataModelDefinition::LegacyEvidenceGrowth(program) => {
            if program.version != LEGACY_EVIDENCE_GROWTH_VERSION {
                return Err(AutomataFrameworkError::UnsupportedLegacyModelVersion {
                    found: program.version,
                });
            }
            validate_legacy_settings(&program.settings).map_err(AutomataFrameworkError::Legacy)
        }
        AutomataModelDefinition::DiscreteMap(definition) => {
            definition
                .initializer
                .validate()
                .map_err(AutomataFrameworkError::Initializer)?;
            if definition.initializer.state_count != definition.program.state_count {
                return Err(
                    AutomataFrameworkError::InitializerProgramStateCountMismatch {
                        initializer: definition.initializer.state_count,
                        program: definition.program.state_count,
                    },
                );
            }
            validate_discrete_program(&definition.program, limits.discrete)?;
            Ok(())
        }
        AutomataModelDefinition::ContinuousLenia(definition) => {
            let initializer = definition.initializer.compile()?;
            if initializer.domain() != definition.program.domain {
                return Err(AutomataFrameworkError::InitializerProgramDomainMismatch {
                    initializer: initializer.domain(),
                    program: definition.program.domain,
                });
            }
            validate_continuous_lenia_program(&definition.program, limits.continuous)?;
            Ok(())
        }
    }
}

/// Dispatches once on the model and state domain, outside every transition
/// loop. A run cannot mix discrete and continuous/legacy state.
pub fn run_compiled_automata(
    program: &CompiledAutomataProgram,
    input: AutomataRunInput<'_>,
    width: usize,
    height: usize,
) -> Result<AutomataSimulationResult, AutomataFrameworkError> {
    match (program, input) {
        (
            CompiledAutomataProgram::LegacyEvidenceGrowthV1(settings),
            AutomataRunInput::LegacyImage(values),
        ) => Ok(AutomataSimulationResult::LegacyEvidenceGrowthV1(
            simulate_legacy(values, width, height, settings)
                .map_err(AutomataFrameworkError::Legacy)?,
        )),
        (
            CompiledAutomataProgram::DiscreteMapV1(program),
            AutomataRunInput::DiscreteState { state, config },
        ) => Ok(AutomataSimulationResult::DiscreteMapV1(
            run_discrete_automaton(program.automaton(), state, width, height, config)?,
        )),
        (
            CompiledAutomataProgram::ContinuousLeniaV1(program),
            AutomataRunInput::ContinuousState { state, config },
        ) => Ok(AutomataSimulationResult::ContinuousLeniaV1(
            run_continuous_lenia(program.automaton(), state, width, height, config)?,
        )),
        (CompiledAutomataProgram::LegacyEvidenceGrowthV1(_), input) => {
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "legacy evidence-growth",
                input: input.label(),
            })
        }
        (CompiledAutomataProgram::DiscreteMapV1(_), input) => {
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "discrete-map",
                input: input.label(),
            })
        }
        (CompiledAutomataProgram::ContinuousLeniaV1(_), input) => {
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "continuous-lenia",
                input: input.label(),
            })
        }
    }
}

/// Runtime dispatcher with cooperative cancellation at generation
/// boundaries. The token is execution control and is intentionally absent
/// from persisted run specifications.
pub fn run_compiled_automata_with_cancellation(
    program: &CompiledAutomataProgram,
    input: AutomataRunInput<'_>,
    width: usize,
    height: usize,
    cancellation: &AtomicBool,
) -> Result<AutomataSimulationResult, AutomataFrameworkError> {
    match (program, input) {
        (
            CompiledAutomataProgram::DiscreteMapV1(program),
            AutomataRunInput::DiscreteState { state, config },
        ) => Ok(AutomataSimulationResult::DiscreteMapV1(
            run_discrete_automaton_with_cancellation(
                program.automaton(),
                state,
                width,
                height,
                config,
                cancellation,
            )?,
        )),
        (
            CompiledAutomataProgram::ContinuousLeniaV1(program),
            AutomataRunInput::ContinuousState { state, config },
        ) => Ok(AutomataSimulationResult::ContinuousLeniaV1(
            run_continuous_lenia_with_cancellation(
                program.automaton(),
                state,
                width,
                height,
                config,
                cancellation,
            )?,
        )),
        (CompiledAutomataProgram::LegacyEvidenceGrowthV1(_), AutomataRunInput::LegacyImage(_)) => {
            Err(AutomataFrameworkError::Legacy(
                "The legacy compatibility model does not support cooperative cancellation."
                    .to_owned(),
            ))
        }
        (CompiledAutomataProgram::LegacyEvidenceGrowthV1(_), input) => {
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "legacy evidence-growth",
                input: input.label(),
            })
        }
        (CompiledAutomataProgram::DiscreteMapV1(_), input) => {
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "discrete-map",
                input: input.label(),
            })
        }
        (CompiledAutomataProgram::ContinuousLeniaV1(_), input) => {
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "continuous-lenia",
                input: input.label(),
            })
        }
    }
}

/// Runs the image-conditioned continuous cellular automata model used by the
/// original analysis workflow.
///
/// The source intensities are normalized to form both the initial state and
/// the immutable evidence field. This intentionally retains the legacy
/// evidence gating and decay behavior until that model is versioned.
pub fn simulate_legacy(
    values: &[u32],
    width: usize,
    height: usize,
    settings: &CellularAutomataSettings,
) -> Result<CellularAutomataResult, String> {
    let initial_state = initialize_legacy_state(values, width, height, settings.seed_mode)?;
    evolve_legacy_state(&initial_state, width, height, settings)
}

/// Maps integer image intensities into the legacy model's continuous state
/// space without performing an evolution step.
pub fn initialize_legacy_state(
    values: &[u32],
    width: usize,
    height: usize,
    seed_mode: CellularAutomataSeedMode,
) -> Result<Vec<f32>, String> {
    validate_rectangular_len(values.len(), width, height)?;

    let min_value = values.iter().copied().min().unwrap_or(0);
    let max_value = values.iter().copied().max().unwrap_or(min_value);
    let observed_range = max_value.saturating_sub(min_value) as f32;
    let mut state = if observed_range <= f32::EPSILON {
        vec![0.0; values.len()]
    } else {
        values
            .iter()
            .map(|value| (value.saturating_sub(min_value) as f32 / observed_range).clamp(0.0, 1.0))
            .collect::<Vec<_>>()
    };
    if observed_range > f32::EPSILON && seed_mode == CellularAutomataSeedMode::Dark {
        for value in &mut state {
            *value = 1.0 - *value;
        }
    }

    Ok(state)
}

/// Evolves an already-mapped continuous state using the legacy evidence-gated
/// transition model. The supplied initial state is also the immutable evidence
/// field, matching the original workflow.
pub fn evolve_legacy_state(
    initial_state: &[f32],
    width: usize,
    height: usize,
    settings: &CellularAutomataSettings,
) -> Result<CellularAutomataResult, String> {
    validate_rectangular_len(initial_state.len(), width, height)?;
    if initial_state
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(
            "Cellular automata initial state must contain finite values from 0 to 1.".to_owned(),
        );
    }
    validate_legacy_scalars(settings)?;
    let kernels = compile_cellular_kernels(settings)?;

    let evidence = initial_state;
    let mut state = initial_state.to_vec();
    let mut next_state = vec![0.0_f32; state.len()];
    let time_step = settings.time_step.clamp(0.001, 1.0);
    let growth_rate = settings.growth_rate.clamp(0.0, 4.0);
    let decay_rate = settings.decay_rate.clamp(0.0, 4.0);
    let iterations = settings.iterations.clamp(1, 512);
    let mut iterations_run = 0;
    let mut converged = false;

    for iteration in 0..iterations {
        let max_change = next_state
            .par_chunks_mut(width)
            .enumerate()
            .map(|(y, row)| {
                let mut row_max_change = 0.0_f32;
                for (x, output) in row.iter_mut().enumerate() {
                    let index = y * width + x;
                    let mut response = f32::NEG_INFINITY;

                    for kernel in &kernels {
                        let mut sensed = 0.0_f32;
                        for sample in kernel {
                            let sample_x = reflect_coordinate(x as isize + sample.offset_x, width);
                            let sample_y = reflect_coordinate(y as isize + sample.offset_y, height);
                            sensed += state[sample_y * width + sample_x] * sample.weight;
                        }
                        response = response.max(cellular_growth_response(sensed, settings));
                    }

                    let evidence_gated_response = if response > 0.0 {
                        response * evidence[index]
                    } else {
                        response
                    };
                    let change = time_step
                        * (growth_rate * evidence_gated_response - decay_rate * state[index]);
                    let updated = (state[index] + change).clamp(0.0, 1.0);
                    row_max_change = row_max_change.max((updated - state[index]).abs());
                    *output = updated;
                }
                row_max_change
            })
            .reduce(|| 0.0, f32::max);

        std::mem::swap(&mut state, &mut next_state);
        iterations_run = iteration + 1;
        if max_change < 0.0001 {
            converged = true;
            break;
        }
    }

    let mean_activation = state.iter().sum::<f32>() / state.len() as f32;
    let active_fraction = state
        .iter()
        .filter(|activation| **activation >= 0.1)
        .count() as f32
        / state.len() as f32;

    Ok(CellularAutomataResult {
        size: [width, height],
        state,
        summary: CellularAutomataSummary {
            iterations_run,
            converged,
            mean_activation,
            active_fraction,
        },
    })
}

fn validate_rectangular_len(len: usize, width: usize, height: usize) -> Result<(), String> {
    let expected_len = width.checked_mul(height);
    if width == 0 || height == 0 || expected_len != Some(len) {
        return Err("Cellular automata requires a non-empty rectangular image.".to_owned());
    }
    Ok(())
}

fn validate_legacy_scalars(settings: &CellularAutomataSettings) -> Result<(), String> {
    if [
        settings.response_center,
        settings.response_width,
        settings.growth_rate,
        settings.decay_rate,
        settings.time_step,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("Cellular automata response and evolution values must be finite.".to_owned());
    }
    Ok(())
}

fn validate_legacy_settings(settings: &CellularAutomataSettings) -> Result<(), String> {
    validate_legacy_scalars(settings)?;
    compile_cellular_kernels(settings).map(|_| ())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CellularKernelSample {
    offset_x: isize,
    offset_y: isize,
    weight: f32,
}

fn compile_cellular_kernels(
    settings: &CellularAutomataSettings,
) -> Result<Vec<Vec<CellularKernelSample>>, String> {
    let kernel_size = settings.kernel_size;
    let expected_len = kernel_size.checked_mul(kernel_size);
    if kernel_size < 3
        || kernel_size.is_multiple_of(2)
        || expected_len != Some(settings.kernel.len())
    {
        return Err("Cellular automata kernel must be an odd square of at least 3 x 3.".to_owned());
    }
    if settings.kernel.iter().any(|weight| !weight.is_finite()) {
        return Err("Cellular automata kernel weights must be finite.".to_owned());
    }

    let kernel_radius = (kernel_size / 2) as isize;
    let kernel_scale: f32 = settings.kernel.iter().map(|weight| weight.abs()).sum();
    if !kernel_scale.is_finite() {
        return Err("Cellular automata kernel weights must have a finite total scale.".to_owned());
    }
    if kernel_scale <= f32::EPSILON {
        return Err("Cellular automata kernel must contain at least one non-zero cell.".to_owned());
    }

    let base_samples = settings
        .kernel
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| weight.abs() > f32::EPSILON)
        .map(|(index, weight)| {
            let x = (index % kernel_size) as isize - kernel_radius;
            let y = (index / kernel_size) as isize - kernel_radius;
            (x, y, weight / kernel_scale)
        })
        .collect::<Vec<_>>();
    let turns = if settings.omni_directional { 4 } else { 1 };
    let mut kernels = Vec::with_capacity(turns);

    for turn in 0..turns {
        let mut rotated = base_samples
            .iter()
            .map(|(offset_x, offset_y, weight)| {
                let (offset_x, offset_y) = rotate_kernel_offset(*offset_x, *offset_y, turn);
                CellularKernelSample {
                    offset_x,
                    offset_y,
                    weight: *weight,
                }
            })
            .collect::<Vec<_>>();
        rotated.sort_by_key(|sample| (sample.offset_y, sample.offset_x));
        if !kernels.contains(&rotated) {
            kernels.push(rotated);
        }
    }

    Ok(kernels)
}

fn rotate_kernel_offset(x: isize, y: isize, turn: usize) -> (isize, isize) {
    match turn % 4 {
        0 => (x, y),
        1 => (-y, x),
        2 => (-x, -y),
        _ => (y, -x),
    }
}

fn cellular_growth_response(sensed: f32, settings: &CellularAutomataSettings) -> f32 {
    let center = settings.response_center.clamp(-1.0, 1.0);
    let width = settings.response_width.clamp(0.001, 2.0);

    match settings.response {
        CellularAutomataResponse::Gaussian => {
            let distance = (sensed - center) / width;
            2.0 * (-0.5 * distance * distance).exp() - 1.0
        }
        CellularAutomataResponse::Band => {
            let edge_distance = width - (sensed - center).abs();
            (edge_distance / (width * 0.15).max(0.005)).tanh()
        }
        CellularAutomataResponse::Threshold => ((sensed - center) / width.max(0.005)).tanh(),
        CellularAutomataResponse::LowThreshold => ((center - sensed) / width.max(0.005)).tanh(),
    }
}

fn reflect_coordinate(coordinate: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }

    let period = 2 * (length as isize - 1);
    let wrapped = coordinate.rem_euclid(period);
    if wrapped < length as isize {
        wrapped as usize
    } else {
        (period - wrapped) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous_automata::{
        CONTINUOUS_INITIALIZER_VERSION, CONTINUOUS_LENIA_PROGRAM_VERSION,
        CONTINUOUS_RUN_CONFIG_VERSION, ContinuousBoundaryPolicy, ContinuousGrowthFunction,
        DenseContinuousKernel, KernelNormalization,
    };
    use crate::discrete_automata::{
        BoundaryPolicy, CompactState, DISCRETE_PROGRAM_VERSION, DISCRETE_RUN_CONFIG_VERSION,
        DiscreteRule, StateStorage,
    };

    fn continuous_lenia_document(calibration: IntensityCalibration) -> AutomataProgramDocument {
        AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::ContinuousLenia(
                ContinuousLeniaProgramDefinition {
                    initializer: ContinuousStateInitializer {
                        version: CONTINUOUS_INITIALIZER_VERSION,
                        calibration,
                        domain: ContinuousDomain::UNIT,
                    },
                    program: ContinuousLeniaProgram {
                        version: CONTINUOUS_LENIA_PROGRAM_VERSION,
                        domain: ContinuousDomain::UNIT,
                        kernel: DenseContinuousKernel {
                            width: 1,
                            height: 1,
                            anchor_x: 0,
                            anchor_y: 0,
                            weights: vec![0.0],
                        },
                        kernel_normalization: KernelNormalization::None,
                        boundary: ContinuousBoundaryPolicy::Wrap,
                        growth: ContinuousGrowthFunction::Linear {
                            slope: 0.0,
                            intercept: 1.0,
                        },
                        time_step: 0.25,
                    },
                },
            ),
        }
    }

    #[test]
    fn legacy_simulation_matches_pre_extraction_fixture() {
        let values = vec![0, 20, 90, 255, 230, 160, 80, 10, 40, 120, 200, 250];
        let settings = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0, -0.5, 0.0, 1.0, 0.25, 0.0, 0.0, 0.5, 0.0],
            seed_mode: CellularAutomataSeedMode::Dark,
            omni_directional: true,
            response: CellularAutomataResponse::Gaussian,
            response_center: 0.15,
            response_width: 0.35,
            growth_rate: 1.3,
            decay_rate: 0.2,
            time_step: 0.17,
            iterations: 3,
        };

        let result = simulate_legacy(&values, 4, 3, &settings).unwrap();
        let expected = [
            1.0,
            1.0,
            0.9894689,
            0.0,
            0.1495924,
            0.5431361,
            1.0,
            1.0,
            1.0,
            0.8152603,
            0.32267764,
            0.030097008,
        ];

        assert_eq!(result.size, [4, 3]);
        for (actual, expected) in result.state.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
        }
        assert_eq!(result.summary.iterations_run, 3);
        assert!(!result.summary.converged);
        assert!((result.summary.mean_activation - 0.65418607).abs() < 1.0e-6);
        assert!((result.summary.active_fraction - 0.8333333).abs() < 1.0e-6);
    }

    #[test]
    fn compatibility_adapter_matches_separate_initialization_and_evolution() {
        let values = [0, 64, 128, 255];
        let settings = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0, 1.0, 0.0, -0.25, 0.5, 0.0, 0.0, 0.0, 0.0],
            seed_mode: CellularAutomataSeedMode::Dark,
            iterations: 2,
            ..CellularAutomataSettings::default()
        };

        let initial = initialize_legacy_state(&values, 2, 2, settings.seed_mode).unwrap();
        let direct = evolve_legacy_state(&initial, 2, 2, &settings).unwrap();
        let compatible = simulate_legacy(&values, 2, 2, &settings).unwrap();

        assert_eq!(direct, compatible);
        assert_eq!(
            initial,
            vec![1.0, 1.0 - 64.0 / 255.0, 1.0 - 128.0 / 255.0, 0.0]
        );
    }

    #[test]
    fn extinguishes_unsupported_activation() {
        let values = vec![0, 32, 64, 96, 128, 160, 192, 224, 255];
        let settings = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            seed_mode: CellularAutomataSeedMode::Bright,
            response: CellularAutomataResponse::Gaussian,
            response_center: -1.0,
            response_width: 0.01,
            decay_rate: 0.25,
            time_step: 0.2,
            iterations: 5,
            ..CellularAutomataSettings::default()
        };

        let result = simulate_legacy(&values, 3, 3, &settings).unwrap();

        assert!(result.state.iter().sum::<f32>() / (result.state.len() as f32) < 0.1);
        assert_eq!(result.summary.iterations_run, 5);
        assert!(result.summary.active_fraction < 0.2);
    }

    #[test]
    fn preserves_supported_activation() {
        let values = vec![0, 0, 0, 0, 255, 0, 0, 0, 0];
        let settings = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            seed_mode: CellularAutomataSeedMode::Bright,
            response: CellularAutomataResponse::Threshold,
            response_center: 0.2,
            response_width: 0.01,
            decay_rate: 0.0,
            time_step: 0.1,
            iterations: 4,
            ..CellularAutomataSettings::default()
        };

        let result = simulate_legacy(&values, 3, 3, &settings).unwrap();

        assert!(result.state[4] > 0.9);
        assert!(
            result
                .state
                .iter()
                .enumerate()
                .all(|(index, value)| index == 4 || *value < 0.01)
        );
    }

    #[test]
    fn low_pass_threshold_inverts_high_pass_threshold_response() {
        let high_pass = CellularAutomataSettings {
            response: CellularAutomataResponse::Threshold,
            response_center: 0.2,
            response_width: 0.05,
            ..CellularAutomataSettings::default()
        };
        let low_pass = CellularAutomataSettings {
            response: CellularAutomataResponse::LowThreshold,
            ..high_pass.clone()
        };

        assert!(cellular_growth_response(0.0, &high_pass) < 0.0);
        assert!(cellular_growth_response(0.0, &low_pass) > 0.0);
        assert!(cellular_growth_response(0.4, &high_pass) > 0.0);
        assert!(cellular_growth_response(0.4, &low_pass) < 0.0);
    }

    #[test]
    fn omni_directional_kernel_passes_collapse_by_cardinal_symmetry() {
        let mut disk = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            omni_directional: true,
            ..CellularAutomataSettings::default()
        };
        assert_eq!(compile_cellular_kernels(&disk).unwrap().len(), 1);

        disk.kernel = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(compile_cellular_kernels(&disk).unwrap().len(), 2);

        disk.kernel = vec![0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(compile_cellular_kernels(&disk).unwrap().len(), 4);
    }

    #[test]
    fn rejects_an_empty_kernel() {
        let values = vec![0; 9];
        let settings = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0; 9],
            ..CellularAutomataSettings::default()
        };

        let error = simulate_legacy(&values, 3, 3, &settings).unwrap_err();

        assert!(error.contains("non-zero cell"));
    }

    #[test]
    fn rejects_non_rectangular_numeric_input() {
        let error =
            simulate_legacy(&[0, 1, 2], 2, 2, &CellularAutomataSettings::default()).unwrap_err();

        assert_eq!(
            error,
            "Cellular automata requires a non-empty rectangular image."
        );
    }

    #[test]
    fn rejects_overflowing_dimensions() {
        let error = initialize_legacy_state(&[], usize::MAX, 2, CellularAutomataSeedMode::Bright)
            .unwrap_err();

        assert_eq!(
            error,
            "Cellular automata requires a non-empty rectangular image."
        );
    }

    #[test]
    fn rejects_non_finite_engine_inputs() {
        let mut settings = CellularAutomataSettings::default();
        settings.kernel[0] = f32::NAN;
        let kernel_error = evolve_legacy_state(&[0.0], 1, 1, &settings).unwrap_err();
        assert_eq!(
            kernel_error,
            "Cellular automata kernel weights must be finite."
        );

        settings = CellularAutomataSettings::default();
        settings.time_step = f32::INFINITY;
        let scalar_error = evolve_legacy_state(&[0.0], 1, 1, &settings).unwrap_err();
        assert_eq!(
            scalar_error,
            "Cellular automata response and evolution values must be finite."
        );

        let state_error =
            evolve_legacy_state(&[f32::NAN], 1, 1, &CellularAutomataSettings::default())
                .unwrap_err();
        assert_eq!(
            state_error,
            "Cellular automata initial state must contain finite values from 0 to 1."
        );
    }

    #[test]
    fn settings_keep_their_serialized_shape() {
        let settings = CellularAutomataSettings {
            kernel_size: 3,
            kernel: vec![0.0, 1.0, 0.0, -0.5, 1.0, 0.5, 0.0, 0.0, 0.0],
            seed_mode: CellularAutomataSeedMode::Bright,
            omni_directional: true,
            response: CellularAutomataResponse::LowThreshold,
            response_center: -0.25,
            response_width: 0.5,
            growth_rate: 1.25,
            decay_rate: 0.75,
            time_step: 0.125,
            iterations: 7,
        };

        let json = serde_json::to_string(&settings).unwrap();
        let decoded: CellularAutomataSettings = serde_json::from_str(&json).unwrap();

        assert_eq!(
            json,
            r#"{"kernel_size":3,"kernel":[0.0,1.0,0.0,-0.5,1.0,0.5,0.0,0.0,0.0],"seed_mode":"bright","omni_directional":true,"response":"low_threshold","response_center":-0.25,"response_width":0.5,"growth_rate":1.25,"decay_rate":0.75,"time_step":0.125,"iterations":7}"#
        );
        assert_eq!(decoded, settings);
    }

    #[test]
    fn missing_omni_directional_setting_keeps_legacy_default() {
        let json = r#"{
            "kernel_size": 3,
            "kernel": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            "seed_mode": "dark",
            "response": "gaussian",
            "response_center": 0.65,
            "response_width": 0.12,
            "growth_rate": 1.0,
            "decay_rate": 0.1,
            "time_step": 0.15,
            "iterations": 24
        }"#;

        let decoded: CellularAutomataSettings = serde_json::from_str(json).unwrap();

        assert!(!decoded.omni_directional);
    }

    #[test]
    fn versioned_legacy_dispatch_matches_the_compatibility_entry_point() {
        let settings = CellularAutomataSettings {
            iterations: 2,
            ..CellularAutomataSettings::default()
        };
        let values = vec![0, 64, 128, 255];
        let expected = simulate_legacy(&values, 2, 2, &settings).unwrap();
        let compiled = compile_automata_program(
            AutomataProgramDocument {
                schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
                transition_model: AutomataModelDefinition::LegacyEvidenceGrowth(
                    LegacyEvidenceGrowthProgram {
                        version: LEGACY_EVIDENCE_GROWTH_VERSION,
                        settings,
                    },
                ),
            },
            AutomataCompileLimits::default(),
        )
        .unwrap();

        let result =
            run_compiled_automata(&compiled, AutomataRunInput::LegacyImage(&values), 2, 2).unwrap();

        assert_eq!(
            result,
            AutomataSimulationResult::LegacyEvidenceGrowthV1(expected)
        );
    }

    #[test]
    fn versioned_discrete_dispatch_returns_the_discrete_result_variant() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let compiled = compile_automata_program(
            AutomataProgramDocument {
                schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
                transition_model: AutomataModelDefinition::DiscreteMap(DiscreteMapProgram {
                    initializer: HistogramStateInitializer::identity(calibration, 2),
                    program: DiscreteProgram {
                        version: DISCRETE_PROGRAM_VERSION,
                        state_count: 2,
                        storage: StateStorage::U8,
                        kernel: vec![],
                        boundary: BoundaryPolicy::Wrap,
                        rule: DiscreteRule::Direct {
                            table: CompactState::U8(vec![1, 0]),
                        },
                    },
                }),
            },
            AutomataCompileLimits::default(),
        )
        .unwrap();
        let CompiledAutomataProgram::DiscreteMapV1(discrete) = &compiled else {
            panic!("expected compiled discrete program")
        };
        let initial = discrete
            .compact_initial_state_from_bins(calibration, 2, vec![0, 1], 2)
            .unwrap();
        let config = DiscreteRunConfig {
            version: DISCRETE_RUN_CONFIG_VERSION,
            cycles: 1,
            ..DiscreteRunConfig::default()
        };

        let result = run_compiled_automata(
            &compiled,
            AutomataRunInput::DiscreteState {
                state: &initial,
                config: &config,
            },
            2,
            1,
        )
        .unwrap();

        let AutomataSimulationResult::DiscreteMapV1(result) = result else {
            panic!("expected a discrete result")
        };
        assert_eq!(result.final_state.as_u8(), Some([1, 0].as_slice()));
    }

    #[test]
    fn versioned_continuous_dispatch_initializes_and_runs_autonomously() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let compiled = compile_automata_program(
            continuous_lenia_document(calibration),
            AutomataCompileLimits::default(),
        )
        .unwrap();
        let CompiledAutomataProgram::ContinuousLeniaV1(continuous) = &compiled else {
            panic!("expected compiled continuous Lenia program")
        };
        let initial = continuous
            .initial_state_from_intensities(calibration, &[0, 255], 8)
            .unwrap();
        assert_eq!(initial, [0.0, 1.0]);
        let config = ContinuousRunConfig {
            version: CONTINUOUS_RUN_CONFIG_VERSION,
            cycles: 2,
            ..ContinuousRunConfig::default()
        };

        let result = run_compiled_automata(
            &compiled,
            AutomataRunInput::ContinuousState {
                state: &initial,
                config: &config,
            },
            2,
            1,
        )
        .unwrap();

        let AutomataSimulationResult::ContinuousLeniaV1(result) = result else {
            panic!("expected a continuous Lenia result")
        };
        assert_eq!(result.final_state, [0.5, 1.0]);
        assert_eq!(result.cycles_completed, 2);
    }

    #[test]
    fn cancellable_dispatch_stops_before_the_next_generation() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let compiled = compile_automata_program(
            continuous_lenia_document(calibration),
            AutomataCompileLimits::default(),
        )
        .unwrap();
        let CompiledAutomataProgram::ContinuousLeniaV1(continuous) = &compiled else {
            panic!("expected compiled continuous Lenia program")
        };
        let initial = continuous
            .initial_state_from_intensities(calibration, &[0, 255], 8)
            .unwrap();
        let config = ContinuousRunConfig {
            version: CONTINUOUS_RUN_CONFIG_VERSION,
            cycles: 10,
            ..ContinuousRunConfig::default()
        };
        let cancellation = AtomicBool::new(true);

        assert!(matches!(
            run_compiled_automata_with_cancellation(
                &compiled,
                AutomataRunInput::ContinuousState {
                    state: &initial,
                    config: &config,
                },
                2,
                1,
                &cancellation,
            ),
            Err(AutomataFrameworkError::Continuous(
                ContinuousAutomataError::Cancelled
            ))
        ));
    }

    #[test]
    fn continuous_bundle_rejects_initializer_and_provenance_mismatches() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let mut mismatched = continuous_lenia_document(calibration);
        let AutomataModelDefinition::ContinuousLenia(definition) = &mut mismatched.transition_model
        else {
            unreachable!()
        };
        definition.initializer.domain = ContinuousDomain {
            minimum: -1.0,
            maximum: 1.0,
        };
        assert!(matches!(
            compile_automata_program(mismatched, AutomataCompileLimits::default()),
            Err(AutomataFrameworkError::InitializerProgramDomainMismatch { .. })
        ));

        let compiled = compile_automata_program(
            continuous_lenia_document(calibration),
            AutomataCompileLimits::default(),
        )
        .unwrap();
        let CompiledAutomataProgram::ContinuousLeniaV1(continuous) = compiled else {
            panic!("expected compiled continuous Lenia program")
        };
        assert!(matches!(
            continuous.initial_state_from_intensities(
                IntensityCalibration::full_scale(65_536).unwrap(),
                &[0],
                4,
            ),
            Err(AutomataFrameworkError::CalibrationMismatch { .. })
        ));
    }

    #[test]
    fn compiled_discrete_bundle_applies_mapped_bins_with_canonical_u16_storage() {
        let calibration = IntensityCalibration::full_scale(4096).unwrap();
        let mapping = (0..4096).map(|bin| bin % 257).collect::<Vec<_>>();
        let document = AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::DiscreteMap(DiscreteMapProgram {
                initializer: HistogramStateInitializer {
                    version: crate::quantization::HISTOGRAM_INITIALIZER_VERSION,
                    calibration,
                    bin_count: 4096,
                    state_count: 257,
                    bin_to_state: Some(mapping),
                },
                program: DiscreteProgram {
                    version: DISCRETE_PROGRAM_VERSION,
                    state_count: 257,
                    storage: StateStorage::U16,
                    kernel: vec![],
                    boundary: BoundaryPolicy::Wrap,
                    rule: DiscreteRule::Direct {
                        table: CompactState::pack((0..257).collect(), 257).unwrap(),
                    },
                },
            }),
        };
        let json = serde_json::to_string(&document).unwrap();
        let decoded: AutomataProgramDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, document);
        let compiled =
            compile_automata_program(document, AutomataCompileLimits::default()).unwrap();
        let CompiledAutomataProgram::DiscreteMapV1(discrete) = compiled else {
            panic!("expected compiled discrete program")
        };

        let state = discrete
            .compact_initial_state_from_bins(calibration, 4096, vec![0, 256, 257, 4095], 8)
            .unwrap();

        assert_eq!(state.storage(), StateStorage::U16);
        assert_eq!(state.as_u16(), Some([0, 256, 0, 240].as_slice()));
    }

    #[test]
    fn discrete_bundle_rejects_initializer_and_provenance_mismatches() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let mismatched = AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::DiscreteMap(DiscreteMapProgram {
                initializer: HistogramStateInitializer::identity(calibration, 3),
                program: DiscreteProgram {
                    version: DISCRETE_PROGRAM_VERSION,
                    state_count: 2,
                    storage: StateStorage::U8,
                    kernel: vec![],
                    boundary: BoundaryPolicy::Wrap,
                    rule: DiscreteRule::Direct {
                        table: CompactState::U8(vec![0, 1]),
                    },
                },
            }),
        };
        assert!(matches!(
            compile_automata_program(mismatched, AutomataCompileLimits::default()),
            Err(
                AutomataFrameworkError::InitializerProgramStateCountMismatch {
                    initializer: 3,
                    program: 2,
                }
            )
        ));

        let matching = AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::DiscreteMap(DiscreteMapProgram {
                initializer: HistogramStateInitializer::identity(calibration, 2),
                program: DiscreteProgram {
                    version: DISCRETE_PROGRAM_VERSION,
                    state_count: 2,
                    storage: StateStorage::U8,
                    kernel: vec![],
                    boundary: BoundaryPolicy::Wrap,
                    rule: DiscreteRule::Direct {
                        table: CompactState::U8(vec![0, 1]),
                    },
                },
            }),
        };
        let compiled =
            compile_automata_program(matching, AutomataCompileLimits::default()).unwrap();
        let CompiledAutomataProgram::DiscreteMapV1(discrete) = compiled else {
            panic!("expected compiled discrete program")
        };
        assert!(matches!(
            discrete.compact_initial_state_from_bins(
                IntensityCalibration::full_scale(65_536).unwrap(),
                2,
                vec![0],
                1,
            ),
            Err(AutomataFrameworkError::CalibrationMismatch { .. })
        ));
        assert!(matches!(
            discrete.compact_initial_state_from_bins(calibration, 3, vec![0], 1),
            Err(AutomataFrameworkError::BinCountMismatch {
                expected: 2,
                actual: 3,
            })
        ));
    }

    #[test]
    fn dispatcher_rejects_a_state_domain_mismatch() {
        let compiled =
            CompiledAutomataProgram::LegacyEvidenceGrowthV1(CellularAutomataSettings::default());
        let initial = CompactState::U8(vec![0]);
        let config = DiscreteRunConfig::new(1);

        assert!(matches!(
            run_compiled_automata(
                &compiled,
                AutomataRunInput::DiscreteState {
                    state: &initial,
                    config: &config,
                },
                1,
                1,
            ),
            Err(AutomataFrameworkError::IncompatibleInput { .. })
        ));

        let continuous = compile_automata_program(
            continuous_lenia_document(IntensityCalibration::full_scale(256).unwrap()),
            AutomataCompileLimits::default(),
        )
        .unwrap();
        assert!(matches!(
            run_compiled_automata(&continuous, AutomataRunInput::LegacyImage(&[0]), 1, 1,),
            Err(AutomataFrameworkError::IncompatibleInput {
                model: "continuous-lenia",
                input: "legacy-image",
            })
        ));
    }

    #[test]
    fn automata_program_documents_are_tagged_strict_and_separate_from_v1_configs() {
        let document = AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::LegacyEvidenceGrowth(
                LegacyEvidenceGrowthProgram {
                    version: LEGACY_EVIDENCE_GROWTH_VERSION,
                    settings: CellularAutomataSettings::default(),
                },
            ),
        };
        let json = serde_json::to_string(&document).unwrap();
        let decoded: AutomataProgramDocument = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, document);
        assert!(json.contains("\"model\":\"legacy_evidence_growth\""));
        assert!(!json.contains("continuous_lenia"));

        let continuous = continuous_lenia_document(IntensityCalibration::full_scale(256).unwrap());
        let continuous_json = serde_json::to_string(&continuous).unwrap();
        let decoded_continuous: AutomataProgramDocument =
            serde_json::from_str(&continuous_json).unwrap();
        assert_eq!(decoded_continuous, continuous);
        assert!(continuous_json.contains("\"model\":\"continuous_lenia\""));

        let unknown_model = r#"{
            "schema_version":1,
            "transition_model":{"model":"growing_neural_ca","definition":{}}
        }"#;
        assert!(serde_json::from_str::<AutomataProgramDocument>(unknown_model).is_err());

        let mut value = serde_json::to_value(document).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unknown".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<AutomataProgramDocument>(value).is_err());
    }

    #[test]
    fn automata_program_compilation_rejects_unknown_document_and_model_versions() {
        let unknown_document = AutomataProgramDocument {
            schema_version: 99,
            transition_model: AutomataModelDefinition::LegacyEvidenceGrowth(
                LegacyEvidenceGrowthProgram {
                    version: LEGACY_EVIDENCE_GROWTH_VERSION,
                    settings: CellularAutomataSettings::default(),
                },
            ),
        };
        assert!(matches!(
            compile_automata_program(unknown_document, AutomataCompileLimits::default()),
            Err(AutomataFrameworkError::UnsupportedDocumentVersion { found: 99 })
        ));

        let unknown_model = AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::LegacyEvidenceGrowth(
                LegacyEvidenceGrowthProgram {
                    version: 99,
                    settings: CellularAutomataSettings::default(),
                },
            ),
        };
        assert!(matches!(
            compile_automata_program(unknown_model, AutomataCompileLimits::default()),
            Err(AutomataFrameworkError::UnsupportedLegacyModelVersion { found: 99 })
        ));
    }
}
