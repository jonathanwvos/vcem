use std::{
    collections::BTreeSet,
    fmt,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    automata::{
        AUTOMATA_PROGRAM_SCHEMA_VERSION, AutomataCompileLimits, AutomataFrameworkError,
        AutomataModelDefinition, AutomataProgramDocument, AutomataRunInput,
        AutomataSimulationResult, CompiledAutomataProgram, ContinuousLeniaProgramDefinition,
        compile_automata_program, run_compiled_automata_with_cancellation,
    },
    continuous_automata::{
        CONTINUOUS_INITIALIZER_VERSION, CONTINUOUS_LENIA_PROGRAM_VERSION,
        CONTINUOUS_RUN_CONFIG_VERSION, ContinuousAutomataError, ContinuousBoundaryPolicy,
        ContinuousDiagnostics, ContinuousDomain, ContinuousGrowthFunction, ContinuousLeniaProgram,
        ContinuousRunConfig, ContinuousStateInitializer, DenseContinuousKernel,
        KernelNormalization,
    },
    discrete_automata::{
        CompactState, DISCRETE_RUN_CONFIG_VERSION, DiscreteAutomataError, DiscreteDiagnostics,
        DiscreteRunConfig,
    },
    quantization::IntensityCalibration,
};

const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Editable Lenia controls. This is an authoring adapter rather than a second
/// persisted program format: the Lab compiles the resulting shared program
/// document and retains that document as the session identity.
#[derive(Clone, Debug, PartialEq)]
pub struct LeniaLabSettings {
    pub kernel_size: u32,
    pub kernel_ring_center: f32,
    pub kernel_ring_width: f32,
    pub growth_center: f32,
    pub growth_width: f32,
    pub time_step: f32,
    pub boundary: ContinuousBoundaryPolicy,
}

impl Default for LeniaLabSettings {
    fn default() -> Self {
        Self {
            kernel_size: 5,
            kernel_ring_center: 0.55,
            kernel_ring_width: 0.18,
            growth_center: 0.35,
            growth_width: 0.08,
            time_step: 0.1,
            boundary: ContinuousBoundaryPolicy::Wrap,
        }
    }
}

impl LeniaLabSettings {
    pub fn validate(&self) -> Result<(), AutomataLabError> {
        if self.kernel_size == 0 || self.kernel_size > 31 || self.kernel_size.is_multiple_of(2) {
            return Err(AutomataLabError::InvalidSettings(
                "Lenia Lab kernel_size must be odd and between 1 and 31.".to_owned(),
            ));
        }
        if !self.kernel_ring_center.is_finite()
            || !(0.0..=1.5).contains(&self.kernel_ring_center)
            || !self.kernel_ring_width.is_finite()
            || self.kernel_ring_width <= 0.0
        {
            return Err(AutomataLabError::InvalidSettings(
                "Kernel ring parameters must be finite, with centre in [0, 1.5] and width greater than zero."
                    .to_owned(),
            ));
        }
        if !self.growth_center.is_finite()
            || !self.growth_width.is_finite()
            || self.growth_width <= 0.0
            || !self.time_step.is_finite()
            || self.time_step <= 0.0
        {
            return Err(AutomataLabError::InvalidSettings(
                "Growth parameters and time_step must be finite, with widths and time_step greater than zero."
                    .to_owned(),
            ));
        }
        Ok(())
    }

    pub fn program_document(
        &self,
        calibration: IntensityCalibration,
    ) -> Result<AutomataProgramDocument, AutomataLabError> {
        self.validate()?;
        Ok(AutomataProgramDocument {
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
                        kernel: radial_ring_kernel(self)?,
                        kernel_normalization: KernelNormalization::SumToOne,
                        boundary: self.boundary,
                        growth: ContinuousGrowthFunction::canonical_gaussian(
                            self.growth_center,
                            self.growth_width,
                        ),
                        time_step: self.time_step,
                    },
                },
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabExecutionPolicy {
    pub max_working_memory_bytes: u64,
    pub exact_discrete_cycle_detection: bool,
    pub max_diagnostic_memory_bytes: u64,
    pub continuous_convergence_tolerance: f32,
}

impl Default for LabExecutionPolicy {
    fn default() -> Self {
        Self {
            max_working_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            exact_discrete_cycle_detection: false,
            max_diagnostic_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            continuous_convergence_tolerance: 1.0e-6,
        }
    }
}

impl LabExecutionPolicy {
    fn validate(self) -> Result<(), AutomataLabError> {
        if !self.continuous_convergence_tolerance.is_finite()
            || self.continuous_convergence_tolerance < 0.0
        {
            return Err(AutomataLabError::InvalidExecutionPolicy);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabCapturePolicy {
    pub checkpoint_interval: u32,
    pub max_history_bytes: u64,
}

impl Default for LabCapturePolicy {
    fn default() -> Self {
        Self {
            checkpoint_interval: 1,
            max_history_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

impl LabCapturePolicy {
    fn validate(self) -> Result<(), AutomataLabError> {
        if self.checkpoint_interval == 0 {
            return Err(AutomataLabError::InvalidCapturePolicy(
                "Checkpoint interval must be at least one generation.".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LabDiagnostics {
    Discrete(DiscreteDiagnostics),
    Continuous(ContinuousDiagnostics),
}

#[derive(Clone, Debug)]
enum LabState {
    Discrete(Arc<CompactState>),
    Continuous(Arc<Vec<f32>>),
}

impl LabState {
    fn byte_len(&self) -> Result<u64, AutomataLabError> {
        let bytes = match self {
            Self::Discrete(state) => state.byte_len().ok_or(AutomataLabError::SizeOverflow)?,
            Self::Continuous(state) => state
                .len()
                .checked_mul(size_of::<f32>())
                .ok_or(AutomataLabError::SizeOverflow)?,
        };
        u64::try_from(bytes).map_err(|_| AutomataLabError::SizeOverflow)
    }

    fn identity(&self) -> (u8, usize) {
        match self {
            Self::Discrete(state) => (0, Arc::as_ptr(state) as usize),
            Self::Continuous(state) => (1, Arc::as_ptr(state) as usize),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LabCheckpoint {
    pub generation: u32,
    pub diagnostics: Option<LabDiagnostics>,
    state: LabState,
}

#[derive(Clone, Debug)]
pub struct LabStateBookmark {
    pub name: String,
    pub generation: u32,
    pub diagnostics: Option<LabDiagnostics>,
    state: LabState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LabStepOutcome {
    pub checkpoints_created: u32,
    pub checkpoints_skipped_for_budget: u32,
}

#[derive(Clone, Debug)]
pub struct AutomataLabSession {
    size: [usize; 2],
    calibration: IntensityCalibration,
    program_document: AutomataProgramDocument,
    program: CompiledAutomataProgram,
    execution_policy: LabExecutionPolicy,
    capture_policy: LabCapturePolicy,
    initial_state: LabState,
    current_state: LabState,
    generation: u32,
    last_diagnostics: Option<LabDiagnostics>,
    checkpoints: Vec<LabCheckpoint>,
    bookmarks: Vec<LabStateBookmark>,
}

impl AutomataLabSession {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        prepared_values: &[u32],
        calibration: IntensityCalibration,
        size: [usize; 2],
        program_document: AutomataProgramDocument,
        compile_limits: AutomataCompileLimits,
        execution_policy: LabExecutionPolicy,
        capture_policy: LabCapturePolicy,
    ) -> Result<Self, AutomataLabError> {
        execution_policy.validate()?;
        capture_policy.validate()?;
        let cell_count = size[0]
            .checked_mul(size[1])
            .filter(|count| *count > 0)
            .ok_or(AutomataLabError::InvalidDimensions { size })?;
        if prepared_values.len() != cell_count {
            return Err(AutomataLabError::SourceLengthMismatch {
                expected: cell_count,
                actual: prepared_values.len(),
            });
        }

        let program = compile_automata_program(program_document.clone(), compile_limits)
            .map_err(AutomataLabError::Framework)?;
        let initial_state = match &program {
            CompiledAutomataProgram::DiscreteMapV1(program) => LabState::Discrete(Arc::new(
                program
                    .compact_initial_state_from_intensities(
                        calibration,
                        prepared_values,
                        execution_policy.max_working_memory_bytes,
                    )
                    .map_err(AutomataLabError::Framework)?,
            )),
            CompiledAutomataProgram::ContinuousLeniaV1(program) => LabState::Continuous(Arc::new(
                program
                    .initial_state_from_intensities(
                        calibration,
                        prepared_values,
                        execution_policy.max_working_memory_bytes,
                    )
                    .map_err(AutomataLabError::Framework)?,
            )),
            CompiledAutomataProgram::LegacyEvidenceGrowthV1(_) => {
                return Err(AutomataLabError::LegacyModelNotSupported);
            }
        };
        let state_bytes = initial_state.byte_len()?;
        if state_bytes > capture_policy.max_history_bytes {
            return Err(AutomataLabError::HistoryBudgetExceeded {
                required_bytes: state_bytes,
                budget_bytes: capture_policy.max_history_bytes,
            });
        }
        let checkpoints = vec![LabCheckpoint {
            generation: 0,
            diagnostics: None,
            state: initial_state.clone(),
        }];
        Ok(Self {
            size,
            calibration,
            program_document,
            program,
            execution_policy,
            capture_policy,
            initial_state: initial_state.clone(),
            current_state: initial_state,
            generation: 0,
            last_diagnostics: None,
            checkpoints,
            bookmarks: Vec::new(),
        })
    }

    pub fn size(&self) -> [usize; 2] {
        self.size
    }

    pub fn calibration(&self) -> IntensityCalibration {
        self.calibration
    }

    pub fn program_document(&self) -> &AutomataProgramDocument {
        &self.program_document
    }

    pub fn capture_policy(&self) -> LabCapturePolicy {
        self.capture_policy
    }

    pub fn set_capture_policy(
        &mut self,
        capture_policy: LabCapturePolicy,
    ) -> Result<(), AutomataLabError> {
        capture_policy.validate()?;
        let retained = self.retained_history_bytes();
        if retained > capture_policy.max_history_bytes {
            return Err(AutomataLabError::HistoryBudgetExceeded {
                required_bytes: retained,
                budget_bytes: capture_policy.max_history_bytes,
            });
        }
        self.capture_policy = capture_policy;
        Ok(())
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn continuous_state(&self) -> Option<&[f32]> {
        match &self.current_state {
            LabState::Continuous(state) => Some(state.as_slice()),
            LabState::Discrete(_) => None,
        }
    }

    pub fn discrete_state(&self) -> Option<&CompactState> {
        match &self.current_state {
            LabState::Discrete(state) => Some(state.as_ref()),
            LabState::Continuous(_) => None,
        }
    }

    pub fn last_diagnostics(&self) -> Option<LabDiagnostics> {
        self.last_diagnostics
    }

    pub fn checkpoints(&self) -> &[LabCheckpoint] {
        &self.checkpoints
    }

    pub fn bookmarks(&self) -> &[LabStateBookmark] {
        &self.bookmarks
    }

    pub fn retained_history_bytes(&self) -> u64 {
        retained_state_bytes(
            &self.initial_state,
            self.checkpoints.iter().map(|checkpoint| &checkpoint.state),
            self.bookmarks.iter().map(|bookmark| &bookmark.state),
        )
    }

    pub fn step(&mut self, cycles: u32) -> Result<LabStepOutcome, AutomataLabError> {
        self.step_with_cancellation(cycles, &AtomicBool::new(false))
    }

    pub fn step_with_cancellation(
        &mut self,
        cycles: u32,
        cancellation: &AtomicBool,
    ) -> Result<LabStepOutcome, AutomataLabError> {
        if cycles == 0 {
            return Err(AutomataLabError::InvalidStepCycles);
        }
        let final_generation = self
            .generation
            .checked_add(cycles)
            .ok_or(AutomataLabError::GenerationOverflow)?;

        let mut generation = self.generation;
        let mut current_state = self.current_state.clone();
        let mut diagnostics = self.last_diagnostics;
        let mut checkpoints = self.checkpoints.clone();
        let mut outcome = LabStepOutcome::default();

        while generation < final_generation {
            if cancellation.load(Ordering::Relaxed) {
                return Err(AutomataLabError::Cancelled);
            }
            let interval = self.capture_policy.checkpoint_interval;
            let until_checkpoint = interval - generation % interval;
            let chunk_cycles = until_checkpoint.min(final_generation - generation);
            let chunk_start = generation;
            let (next_state, mut chunk_diagnostics) =
                self.run_chunk(&current_state, chunk_cycles, cancellation)?;
            offset_diagnostics(&mut chunk_diagnostics, chunk_start)?;
            generation = generation
                .checked_add(chunk_cycles)
                .ok_or(AutomataLabError::GenerationOverflow)?;
            current_state = next_state;
            diagnostics = Some(chunk_diagnostics);

            if generation.is_multiple_of(interval) {
                checkpoints.push(LabCheckpoint {
                    generation,
                    diagnostics,
                    state: current_state.clone(),
                });
                let required = retained_state_bytes(
                    &self.initial_state,
                    checkpoints.iter().map(|checkpoint| &checkpoint.state),
                    self.bookmarks.iter().map(|bookmark| &bookmark.state),
                );
                if required > self.capture_policy.max_history_bytes {
                    checkpoints.pop();
                    outcome.checkpoints_skipped_for_budget += 1;
                } else {
                    outcome.checkpoints_created += 1;
                }
            }
        }

        self.current_state = current_state;
        self.generation = generation;
        self.last_diagnostics = diagnostics;
        self.checkpoints = checkpoints;
        Ok(outcome)
    }

    fn run_chunk(
        &self,
        current_state: &LabState,
        cycles: u32,
        cancellation: &AtomicBool,
    ) -> Result<(LabState, LabDiagnostics), AutomataLabError> {
        let result = match (&self.program, current_state) {
            (CompiledAutomataProgram::DiscreteMapV1(_), LabState::Discrete(state)) => {
                let config = DiscreteRunConfig {
                    version: DISCRETE_RUN_CONFIG_VERSION,
                    cycles,
                    max_working_memory_bytes: self.execution_policy.max_working_memory_bytes,
                    exact_cycle_detection: self.execution_policy.exact_discrete_cycle_detection,
                    max_diagnostic_memory_bytes: self.execution_policy.max_diagnostic_memory_bytes,
                };
                run_compiled_automata_with_cancellation(
                    &self.program,
                    AutomataRunInput::DiscreteState {
                        state,
                        config: &config,
                    },
                    self.size[0],
                    self.size[1],
                    cancellation,
                )
            }
            (CompiledAutomataProgram::ContinuousLeniaV1(_), LabState::Continuous(state)) => {
                let config = ContinuousRunConfig {
                    version: CONTINUOUS_RUN_CONFIG_VERSION,
                    cycles,
                    max_working_memory_bytes: self.execution_policy.max_working_memory_bytes,
                    convergence_tolerance: self.execution_policy.continuous_convergence_tolerance,
                };
                run_compiled_automata_with_cancellation(
                    &self.program,
                    AutomataRunInput::ContinuousState {
                        state,
                        config: &config,
                    },
                    self.size[0],
                    self.size[1],
                    cancellation,
                )
            }
            _ => return Err(AutomataLabError::StateModelMismatch),
        }
        .map_err(|error| match error {
            AutomataFrameworkError::Continuous(ContinuousAutomataError::Cancelled)
            | AutomataFrameworkError::Discrete(DiscreteAutomataError::Cancelled) => {
                AutomataLabError::Cancelled
            }
            error => AutomataLabError::Framework(error),
        })?;

        match result {
            AutomataSimulationResult::DiscreteMapV1(result) => Ok((
                LabState::Discrete(Arc::new(result.final_state)),
                LabDiagnostics::Discrete(result.diagnostics),
            )),
            AutomataSimulationResult::ContinuousLeniaV1(result) => Ok((
                LabState::Continuous(Arc::new(result.final_state)),
                LabDiagnostics::Continuous(result.diagnostics),
            )),
            AutomataSimulationResult::LegacyEvidenceGrowthV1(_) => {
                Err(AutomataLabError::StateModelMismatch)
            }
        }
    }

    pub fn reset(&mut self) {
        self.current_state = self.initial_state.clone();
        self.generation = 0;
        self.last_diagnostics = None;
        self.checkpoints.clear();
        self.checkpoints.push(LabCheckpoint {
            generation: 0,
            diagnostics: None,
            state: self.initial_state.clone(),
        });
    }

    pub fn restore_checkpoint(&mut self, index: usize) -> Result<(), AutomataLabError> {
        let checkpoint = self
            .checkpoints
            .get(index)
            .ok_or(AutomataLabError::CheckpointNotFound { index })?
            .clone();
        self.current_state = checkpoint.state;
        self.generation = checkpoint.generation;
        self.last_diagnostics = checkpoint.diagnostics;
        self.checkpoints.truncate(index + 1);
        Ok(())
    }

    pub fn save_bookmark(&mut self, name: &str) -> Result<(), AutomataLabError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(AutomataLabError::EmptyBookmarkName);
        }
        if self.bookmarks.iter().any(|bookmark| bookmark.name == name) {
            return Err(AutomataLabError::DuplicateBookmarkName {
                name: name.to_owned(),
            });
        }
        self.bookmarks.push(LabStateBookmark {
            name: name.to_owned(),
            generation: self.generation,
            diagnostics: self.last_diagnostics,
            state: self.current_state.clone(),
        });
        let required_bytes = self.retained_history_bytes();
        if required_bytes > self.capture_policy.max_history_bytes {
            self.bookmarks.pop();
            return Err(AutomataLabError::HistoryBudgetExceeded {
                required_bytes,
                budget_bytes: self.capture_policy.max_history_bytes,
            });
        }
        Ok(())
    }

    pub fn restore_bookmark(&mut self, index: usize) -> Result<(), AutomataLabError> {
        let bookmark = self
            .bookmarks
            .get(index)
            .ok_or(AutomataLabError::BookmarkNotFound { index })?
            .clone();
        self.current_state = bookmark.state.clone();
        self.generation = bookmark.generation;
        self.last_diagnostics = bookmark.diagnostics;
        self.checkpoints.clear();
        self.checkpoints.push(LabCheckpoint {
            generation: bookmark.generation,
            diagnostics: bookmark.diagnostics,
            state: bookmark.state,
        });
        Ok(())
    }
}

#[derive(Debug)]
pub enum AutomataLabError {
    InvalidSettings(String),
    InvalidExecutionPolicy,
    InvalidCapturePolicy(String),
    InvalidDimensions {
        size: [usize; 2],
    },
    SourceLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidStepCycles,
    GenerationOverflow,
    CheckpointNotFound {
        index: usize,
    },
    BookmarkNotFound {
        index: usize,
    },
    EmptyBookmarkName,
    DuplicateBookmarkName {
        name: String,
    },
    HistoryBudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
    },
    SizeOverflow,
    LegacyModelNotSupported,
    StateModelMismatch,
    Cancelled,
    Framework(AutomataFrameworkError),
}

impl fmt::Display for AutomataLabError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSettings(error) | Self::InvalidCapturePolicy(error) => {
                formatter.write_str(error)
            }
            Self::InvalidExecutionPolicy => formatter
                .write_str("Continuous convergence tolerance must be finite and non-negative."),
            Self::InvalidDimensions { size } => write!(
                formatter,
                "Automata Lab requires non-zero addressable dimensions; found {} x {}.",
                size[0], size[1]
            ),
            Self::SourceLengthMismatch { expected, actual } => write!(
                formatter,
                "Prepared plane has {actual} values; expected {expected}."
            ),
            Self::InvalidStepCycles => {
                formatter.write_str("Automata Lab steps require at least one cycle.")
            }
            Self::GenerationOverflow => {
                formatter.write_str("Automata Lab generation overflowed u32.")
            }
            Self::CheckpointNotFound { index } => {
                write!(formatter, "Automata Lab checkpoint {index} does not exist.")
            }
            Self::BookmarkNotFound { index } => {
                write!(
                    formatter,
                    "Automata Lab state bookmark {index} does not exist."
                )
            }
            Self::EmptyBookmarkName => {
                formatter.write_str("State bookmarks require a non-empty name.")
            }
            Self::DuplicateBookmarkName { name } => {
                write!(formatter, "A state bookmark named {name:?} already exists.")
            }
            Self::HistoryBudgetExceeded {
                required_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "Automata Lab history requires {required_bytes} bytes, exceeding the {budget_bytes}-byte budget."
            ),
            Self::SizeOverflow => formatter.write_str("Automata Lab state size overflowed."),
            Self::LegacyModelNotSupported => formatter.write_str(
                "The autonomous Automata Lab does not run the legacy evidence-gated model.",
            ),
            Self::StateModelMismatch => {
                formatter.write_str("Automata Lab state does not match its compiled model.")
            }
            Self::Cancelled => formatter.write_str("Automata Lab computation was cancelled."),
            Self::Framework(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AutomataLabError {}

fn offset_diagnostics(
    diagnostics: &mut LabDiagnostics,
    generation_offset: u32,
) -> Result<(), AutomataLabError> {
    let offset = |generation: u32| {
        generation_offset
            .checked_add(generation)
            .ok_or(AutomataLabError::GenerationOverflow)
    };
    match diagnostics {
        LabDiagnostics::Continuous(diagnostics) => {
            diagnostics.exact_fixed_point_generation = diagnostics
                .exact_fixed_point_generation
                .map(offset)
                .transpose()?;
        }
        LabDiagnostics::Discrete(diagnostics) => {
            diagnostics.fixed_point_generation =
                diagnostics.fixed_point_generation.map(offset).transpose()?;
            if let Some(cycle) = &mut diagnostics.cycle {
                cycle.start_generation = offset(cycle.start_generation)?;
                cycle.first_repeat_generation = offset(cycle.first_repeat_generation)?;
            }
        }
    }
    Ok(())
}

fn radial_ring_kernel(
    settings: &LeniaLabSettings,
) -> Result<DenseContinuousKernel, AutomataLabError> {
    let size = usize::try_from(settings.kernel_size).map_err(|_| AutomataLabError::SizeOverflow)?;
    let len = size
        .checked_mul(size)
        .ok_or(AutomataLabError::SizeOverflow)?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(len)
        .map_err(|_| AutomataLabError::SizeOverflow)?;
    let center = (size / 2) as f64;
    let radius = center.max(1.0);
    for y in 0..size {
        for x in 0..size {
            let dx = x as f64 - center;
            let dy = y as f64 - center;
            let normalized_radius = (dx * dx + dy * dy).sqrt() / radius;
            let offset = (normalized_radius - settings.kernel_ring_center as f64)
                / settings.kernel_ring_width as f64;
            weights.push((-0.5 * offset * offset).exp() as f32);
        }
    }
    Ok(DenseContinuousKernel {
        width: settings.kernel_size,
        height: settings.kernel_size,
        anchor_x: settings.kernel_size / 2,
        anchor_y: settings.kernel_size / 2,
        weights,
    })
}

fn retained_state_bytes<'a>(
    initial_state: &'a LabState,
    checkpoints: impl Iterator<Item = &'a LabState>,
    bookmarks: impl Iterator<Item = &'a LabState>,
) -> u64 {
    let mut identities = BTreeSet::new();
    let mut bytes = 0_u64;
    for state in std::iter::once(initial_state)
        .chain(checkpoints)
        .chain(bookmarks)
    {
        if identities.insert(state.identity()) {
            bytes = bytes.saturating_add(state.byte_len().unwrap_or(u64::MAX));
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        automata::{AutomataModelDefinition, DiscreteMapProgram},
        discrete_automata::{
            BoundaryPolicy, CompactState, DISCRETE_PROGRAM_VERSION, DiscreteProgram, DiscreteRule,
            StateStorage,
        },
        quantization::{HISTOGRAM_INITIALIZER_VERSION, HistogramStateInitializer},
    };

    fn continuous_session(
        capture_policy: LabCapturePolicy,
    ) -> Result<AutomataLabSession, AutomataLabError> {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        AutomataLabSession::new(
            &[0, 64, 128, 255],
            calibration,
            [2, 2],
            LeniaLabSettings::default()
                .program_document(calibration)
                .unwrap(),
            AutomataCompileLimits::default(),
            LabExecutionPolicy::default(),
            capture_policy,
        )
    }

    fn discrete_identity_document(calibration: IntensityCalibration) -> AutomataProgramDocument {
        AutomataProgramDocument {
            schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
            transition_model: AutomataModelDefinition::DiscreteMap(DiscreteMapProgram {
                initializer: HistogramStateInitializer {
                    version: HISTOGRAM_INITIALIZER_VERSION,
                    calibration,
                    bin_count: 2,
                    state_count: 2,
                    bin_to_state: None,
                },
                program: DiscreteProgram {
                    version: DISCRETE_PROGRAM_VERSION,
                    state_count: 2,
                    storage: StateStorage::U8,
                    kernel: Vec::new(),
                    boundary: BoundaryPolicy::Wrap,
                    rule: DiscreteRule::Direct {
                        table: CompactState::U8(vec![0, 1]),
                    },
                },
            }),
        }
    }

    #[test]
    fn lenia_authoring_controls_compile_into_the_shared_program_document() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let document = LeniaLabSettings::default()
            .program_document(calibration)
            .unwrap();
        assert!(matches!(
            document.transition_model,
            AutomataModelDefinition::ContinuousLenia(_)
        ));
        compile_automata_program(document, AutomataCompileLimits::default()).unwrap();
    }

    #[test]
    fn continuous_generation_zero_is_calibrated_and_reset_is_exact() {
        let mut session = continuous_session(LabCapturePolicy::default()).unwrap();
        let initial = session.continuous_state().unwrap().to_vec();
        assert_eq!(initial[0], 0.0);
        assert_eq!(initial[3], 1.0);
        session.step(2).unwrap();
        session.reset();
        assert_eq!(session.generation(), 0);
        assert_eq!(session.continuous_state().unwrap(), initial);
    }

    #[test]
    fn discrete_programs_use_the_same_lab_session_and_dispatcher() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let mut session = AutomataLabSession::new(
            &[0, 255],
            calibration,
            [2, 1],
            discrete_identity_document(calibration),
            AutomataCompileLimits::default(),
            LabExecutionPolicy::default(),
            LabCapturePolicy::default(),
        )
        .unwrap();
        session.step(3).unwrap();
        assert_eq!(session.generation(), 3);
        assert_eq!(
            session.discrete_state().unwrap(),
            &CompactState::U8(vec![0, 1])
        );
        assert!(matches!(
            session.last_diagnostics(),
            Some(LabDiagnostics::Discrete(_))
        ));
    }

    #[test]
    fn multi_cycle_steps_capture_every_exact_interval() {
        let mut session = continuous_session(LabCapturePolicy {
            checkpoint_interval: 2,
            ..LabCapturePolicy::default()
        })
        .unwrap();
        let outcome = session.step(7).unwrap();
        assert_eq!(outcome.checkpoints_created, 3);
        assert_eq!(
            session
                .checkpoints()
                .iter()
                .map(|checkpoint| checkpoint.generation)
                .collect::<Vec<_>>(),
            [0, 2, 4, 6]
        );
        assert_eq!(session.generation(), 7);
    }

    #[test]
    fn capture_policy_changes_do_not_reset_or_recompile_the_session() {
        let mut session = continuous_session(LabCapturePolicy::default()).unwrap();
        session.step(1).unwrap();
        let document = session.program_document().clone();
        session
            .set_capture_policy(LabCapturePolicy {
                checkpoint_interval: 5,
                ..LabCapturePolicy::default()
            })
            .unwrap();
        assert_eq!(session.generation(), 1);
        assert_eq!(session.program_document(), &document);
        assert_eq!(session.capture_policy().checkpoint_interval, 5);
    }

    #[test]
    fn cancellation_is_atomic_at_checkpoint_chunk_boundaries() {
        let mut session = continuous_session(LabCapturePolicy::default()).unwrap();
        let before = session.continuous_state().unwrap().to_vec();
        let cancellation = AtomicBool::new(true);
        assert!(matches!(
            session.step_with_cancellation(5, &cancellation),
            Err(AutomataLabError::Cancelled)
        ));
        assert_eq!(session.generation(), 0);
        assert_eq!(session.continuous_state().unwrap(), before);
    }

    #[test]
    fn state_bookmarks_share_states_and_restore_without_losing_generation_zero() {
        let mut session = continuous_session(LabCapturePolicy::default()).unwrap();
        session.step(1).unwrap();
        let bookmarked = session.continuous_state().unwrap().to_vec();
        session.save_bookmark("first").unwrap();
        session.step(1).unwrap();
        session.restore_bookmark(0).unwrap();
        assert_eq!(session.generation(), 1);
        assert_eq!(session.continuous_state().unwrap(), bookmarked);
        assert_eq!(
            session.retained_history_bytes(),
            2 * 4 * size_of::<f32>() as u64
        );
    }

    #[test]
    fn invalid_settings_and_history_budgets_fail_without_partial_sessions() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let invalid = LeniaLabSettings {
            kernel_size: 4,
            ..LeniaLabSettings::default()
        };
        assert!(matches!(
            invalid.program_document(calibration),
            Err(AutomataLabError::InvalidSettings(_))
        ));
        assert!(matches!(
            AutomataLabSession::new(
                &[0],
                calibration,
                [1, 1],
                LeniaLabSettings::default()
                    .program_document(calibration)
                    .unwrap(),
                AutomataCompileLimits::default(),
                LabExecutionPolicy::default(),
                LabCapturePolicy {
                    max_history_bytes: 3,
                    ..LabCapturePolicy::default()
                },
            ),
            Err(AutomataLabError::HistoryBudgetExceeded { .. })
        ));
    }
}
