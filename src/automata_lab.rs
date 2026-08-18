use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt, mem::size_of, sync::Arc};

use crate::{
    continuous_automata::{
        CONTINUOUS_INITIALIZER_VERSION, CONTINUOUS_LENIA_PROGRAM_VERSION, CompiledContinuousLenia,
        ContinuousAutomataError, ContinuousBoundaryPolicy, ContinuousCompileLimits,
        ContinuousDiagnostics, ContinuousDomain, ContinuousGrowthFunction, ContinuousLeniaProgram,
        ContinuousRunConfig, ContinuousStateInitializer, DenseContinuousKernel,
        KernelNormalization, compile_continuous_lenia, run_continuous_lenia,
    },
    quantization::IntensityCalibration,
};

pub const AUTOMATA_LAB_SETTINGS_VERSION: u32 = 1;

const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeniaLabSettings {
    pub version: u32,
    pub kernel_size: u32,
    pub kernel_ring_center: f32,
    pub kernel_ring_width: f32,
    pub growth_center: f32,
    pub growth_width: f32,
    pub time_step: f32,
    pub boundary: ContinuousBoundaryPolicy,
    pub checkpoint_interval: u32,
    pub max_state_bytes: u64,
    pub max_history_bytes: u64,
}

impl Default for LeniaLabSettings {
    fn default() -> Self {
        Self {
            version: AUTOMATA_LAB_SETTINGS_VERSION,
            kernel_size: 5,
            kernel_ring_center: 0.55,
            kernel_ring_width: 0.18,
            growth_center: 0.35,
            growth_width: 0.08,
            time_step: 0.1,
            boundary: ContinuousBoundaryPolicy::Wrap,
            checkpoint_interval: 1,
            max_state_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            max_history_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

impl LeniaLabSettings {
    pub fn validate(&self) -> Result<(), AutomataLabError> {
        if self.version != AUTOMATA_LAB_SETTINGS_VERSION {
            return Err(AutomataLabError::UnsupportedSettingsVersion {
                found: self.version,
            });
        }
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
        if self.checkpoint_interval == 0 {
            return Err(AutomataLabError::InvalidSettings(
                "Checkpoint interval must be at least one generation.".to_owned(),
            ));
        }
        Ok(())
    }

    fn compile_program(&self) -> Result<CompiledContinuousLenia, AutomataLabError> {
        self.validate()?;
        let kernel = radial_ring_kernel(self)?;
        compile_continuous_lenia(
            ContinuousLeniaProgram {
                version: CONTINUOUS_LENIA_PROGRAM_VERSION,
                domain: ContinuousDomain::UNIT,
                kernel,
                kernel_normalization: KernelNormalization::SumToOne,
                boundary: self.boundary,
                growth: ContinuousGrowthFunction::canonical_gaussian(
                    self.growth_center,
                    self.growth_width,
                ),
                time_step: self.time_step,
            },
            ContinuousCompileLimits::default(),
        )
        .map_err(AutomataLabError::Continuous)
    }
}

#[derive(Clone, Debug)]
pub struct LabCheckpoint {
    pub generation: u32,
    pub diagnostics: Option<ContinuousDiagnostics>,
    state: Arc<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct LabCandidateSnapshot {
    pub name: String,
    pub generation: u32,
    pub diagnostics: Option<ContinuousDiagnostics>,
    state: Arc<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabStepOutcome {
    pub checkpoint_created: bool,
    pub checkpoint_skipped_for_budget: bool,
}

#[derive(Clone, Debug)]
pub struct AutomataLabSession {
    size: [usize; 2],
    calibration: IntensityCalibration,
    settings: LeniaLabSettings,
    program: CompiledContinuousLenia,
    initial_state: Arc<Vec<f32>>,
    current_state: Arc<Vec<f32>>,
    generation: u32,
    last_diagnostics: Option<ContinuousDiagnostics>,
    checkpoints: Vec<LabCheckpoint>,
    candidates: Vec<LabCandidateSnapshot>,
}

impl AutomataLabSession {
    pub fn new(
        prepared_values: &[u32],
        calibration: IntensityCalibration,
        size: [usize; 2],
        settings: LeniaLabSettings,
    ) -> Result<Self, AutomataLabError> {
        settings.validate()?;
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
        let initializer = ContinuousStateInitializer {
            version: CONTINUOUS_INITIALIZER_VERSION,
            calibration,
            domain: ContinuousDomain::UNIT,
        }
        .compile()
        .map_err(AutomataLabError::Continuous)?;
        let initial_state = Arc::new(
            initializer
                .initialize(prepared_values, settings.max_state_bytes)
                .map_err(AutomataLabError::Continuous)?,
        );
        let state_bytes = state_bytes(initial_state.len())?;
        if state_bytes > settings.max_history_bytes {
            return Err(AutomataLabError::HistoryBudgetExceeded {
                required_bytes: state_bytes,
                budget_bytes: settings.max_history_bytes,
            });
        }
        let program = settings.compile_program()?;
        let checkpoints = vec![LabCheckpoint {
            generation: 0,
            diagnostics: None,
            state: Arc::clone(&initial_state),
        }];
        Ok(Self {
            size,
            calibration,
            settings,
            program,
            initial_state: Arc::clone(&initial_state),
            current_state: initial_state,
            generation: 0,
            last_diagnostics: None,
            checkpoints,
            candidates: Vec::new(),
        })
    }

    pub fn size(&self) -> [usize; 2] {
        self.size
    }

    pub fn calibration(&self) -> IntensityCalibration {
        self.calibration
    }

    pub fn settings(&self) -> &LeniaLabSettings {
        &self.settings
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

    pub fn current_state(&self) -> &[f32] {
        self.current_state.as_slice()
    }

    pub fn last_diagnostics(&self) -> Option<ContinuousDiagnostics> {
        self.last_diagnostics
    }

    pub fn checkpoints(&self) -> &[LabCheckpoint] {
        &self.checkpoints
    }

    pub fn candidates(&self) -> &[LabCandidateSnapshot] {
        &self.candidates
    }

    pub fn retained_history_bytes(&self) -> u64 {
        retained_state_bytes(
            &self.initial_state,
            self.checkpoints.iter().map(|checkpoint| &checkpoint.state),
            self.candidates.iter().map(|candidate| &candidate.state),
        )
    }

    pub fn step(&mut self, cycles: u32) -> Result<LabStepOutcome, AutomataLabError> {
        if cycles == 0 {
            return Err(AutomataLabError::InvalidStepCycles);
        }
        let next_generation = self
            .generation
            .checked_add(cycles)
            .ok_or(AutomataLabError::GenerationOverflow)?;
        let result = run_continuous_lenia(
            &self.program,
            self.current_state.as_slice(),
            self.size[0],
            self.size[1],
            &ContinuousRunConfig {
                max_working_memory_bytes: self.settings.max_state_bytes,
                ..ContinuousRunConfig::new(cycles)
            },
        )
        .map_err(AutomataLabError::Continuous)?;
        let previous_generation = self.generation;
        let mut diagnostics = result.diagnostics;
        diagnostics.exact_fixed_point_generation = diagnostics
            .exact_fixed_point_generation
            .map(|generation| {
                previous_generation
                    .checked_add(generation)
                    .ok_or(AutomataLabError::GenerationOverflow)
            })
            .transpose()?;
        self.current_state = Arc::new(result.final_state);
        self.generation = next_generation;
        self.last_diagnostics = Some(diagnostics);

        let crossed_checkpoint = previous_generation / self.settings.checkpoint_interval
            != self.generation / self.settings.checkpoint_interval;
        let mut outcome = LabStepOutcome {
            checkpoint_created: false,
            checkpoint_skipped_for_budget: false,
        };
        if crossed_checkpoint {
            self.checkpoints.push(LabCheckpoint {
                generation: self.generation,
                diagnostics: self.last_diagnostics,
                state: Arc::clone(&self.current_state),
            });
            if self.retained_history_bytes() > self.settings.max_history_bytes {
                self.checkpoints.pop();
                outcome.checkpoint_skipped_for_budget = true;
            } else {
                outcome.checkpoint_created = true;
            }
        }
        Ok(outcome)
    }

    pub fn reset(&mut self) {
        self.current_state = Arc::clone(&self.initial_state);
        self.generation = 0;
        self.last_diagnostics = None;
        self.checkpoints.clear();
        self.checkpoints.push(LabCheckpoint {
            generation: 0,
            diagnostics: None,
            state: Arc::clone(&self.initial_state),
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

    pub fn save_candidate(&mut self, name: &str) -> Result<(), AutomataLabError> {
        let name = name.trim();
        if name.is_empty() {
            return Err(AutomataLabError::EmptyCandidateName);
        }
        if self
            .candidates
            .iter()
            .any(|candidate| candidate.name == name)
        {
            return Err(AutomataLabError::DuplicateCandidateName {
                name: name.to_owned(),
            });
        }
        self.candidates.push(LabCandidateSnapshot {
            name: name.to_owned(),
            generation: self.generation,
            diagnostics: self.last_diagnostics,
            state: Arc::clone(&self.current_state),
        });
        let required_bytes = self.retained_history_bytes();
        if required_bytes > self.settings.max_history_bytes {
            self.candidates.pop();
            return Err(AutomataLabError::HistoryBudgetExceeded {
                required_bytes,
                budget_bytes: self.settings.max_history_bytes,
            });
        }
        Ok(())
    }

    pub fn restore_candidate(&mut self, index: usize) -> Result<(), AutomataLabError> {
        let candidate = self
            .candidates
            .get(index)
            .ok_or(AutomataLabError::CandidateNotFound { index })?
            .clone();
        self.current_state = Arc::clone(&candidate.state);
        self.generation = candidate.generation;
        self.last_diagnostics = candidate.diagnostics;
        self.checkpoints.clear();
        self.checkpoints.push(LabCheckpoint {
            generation: candidate.generation,
            diagnostics: candidate.diagnostics,
            state: candidate.state,
        });
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AutomataLabError {
    UnsupportedSettingsVersion {
        found: u32,
    },
    InvalidSettings(String),
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
    CandidateNotFound {
        index: usize,
    },
    EmptyCandidateName,
    DuplicateCandidateName {
        name: String,
    },
    HistoryBudgetExceeded {
        required_bytes: u64,
        budget_bytes: u64,
    },
    SizeOverflow,
    Continuous(ContinuousAutomataError),
}

impl fmt::Display for AutomataLabError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSettingsVersion { found } => {
                write!(
                    formatter,
                    "Unsupported Automata Lab settings version {found}."
                )
            }
            Self::InvalidSettings(error) => formatter.write_str(error),
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
            Self::CandidateNotFound { index } => {
                write!(formatter, "Automata Lab candidate {index} does not exist.")
            }
            Self::EmptyCandidateName => {
                formatter.write_str("Candidate snapshots require a non-empty name.")
            }
            Self::DuplicateCandidateName { name } => {
                write!(formatter, "A candidate named {name:?} already exists.")
            }
            Self::HistoryBudgetExceeded {
                required_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "Automata Lab history requires {required_bytes} bytes, exceeding the {budget_bytes}-byte budget."
            ),
            Self::SizeOverflow => formatter.write_str("Automata Lab state size overflowed."),
            Self::Continuous(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AutomataLabError {}

impl From<ContinuousAutomataError> for AutomataLabError {
    fn from(error: ContinuousAutomataError) -> Self {
        Self::Continuous(error)
    }
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

fn state_bytes(len: usize) -> Result<u64, AutomataLabError> {
    len.checked_mul(size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(AutomataLabError::SizeOverflow)
}

fn retained_state_bytes<'a>(
    initial_state: &'a Arc<Vec<f32>>,
    checkpoints: impl Iterator<Item = &'a Arc<Vec<f32>>>,
    candidates: impl Iterator<Item = &'a Arc<Vec<f32>>>,
) -> u64 {
    let mut pointers = BTreeSet::new();
    let mut bytes = 0_u64;
    for state in std::iter::once(initial_state)
        .chain(checkpoints)
        .chain(candidates)
    {
        let pointer = Arc::as_ptr(state) as usize;
        if pointers.insert(pointer) {
            bytes = bytes.saturating_add(
                u64::try_from(state.len().saturating_mul(size_of::<f32>())).unwrap_or(u64::MAX),
            );
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(settings: LeniaLabSettings) -> AutomataLabSession {
        AutomataLabSession::new(
            &[0, 64, 128, 255],
            IntensityCalibration::full_scale(256).unwrap(),
            [2, 2],
            settings,
        )
        .unwrap()
    }

    #[test]
    fn generation_zero_is_calibrated_and_reset_is_exact() {
        let mut session = session(LeniaLabSettings::default());
        let initial = session.current_state().to_vec();
        assert_eq!(initial[0], 0.0);
        assert_eq!(initial[3], 1.0);
        assert_eq!(session.checkpoints().len(), 1);

        session.step(2).unwrap();
        assert_eq!(session.generation(), 2);
        session.reset();

        assert_eq!(session.generation(), 0);
        assert_eq!(session.current_state(), initial);
        assert_eq!(session.checkpoints().len(), 1);
        assert_eq!(session.checkpoints()[0].generation, 0);
    }

    #[test]
    fn diagnostics_use_the_session_generation_across_separate_steps() {
        let mut session = AutomataLabSession::new(
            &[0],
            IntensityCalibration::full_scale(256).unwrap(),
            [1, 1],
            LeniaLabSettings::default(),
        )
        .unwrap();

        session.step(1).unwrap();
        assert_eq!(
            session
                .last_diagnostics()
                .unwrap()
                .exact_fixed_point_generation,
            Some(0)
        );
        session.step(2).unwrap();
        assert_eq!(
            session
                .last_diagnostics()
                .unwrap()
                .exact_fixed_point_generation,
            Some(1)
        );
    }

    #[test]
    fn stepping_creates_timeline_checkpoints_and_restore_branches_history() {
        let settings = LeniaLabSettings {
            checkpoint_interval: 2,
            ..LeniaLabSettings::default()
        };
        let mut session = session(settings);

        assert!(!session.step(1).unwrap().checkpoint_created);
        assert!(session.step(1).unwrap().checkpoint_created);
        assert!(session.step(2).unwrap().checkpoint_created);
        assert_eq!(
            session
                .checkpoints()
                .iter()
                .map(|checkpoint| checkpoint.generation)
                .collect::<Vec<_>>(),
            [0, 2, 4]
        );

        session.restore_checkpoint(1).unwrap();
        assert_eq!(session.generation(), 2);
        assert_eq!(session.checkpoints().len(), 2);
    }

    #[test]
    fn named_candidates_share_states_and_can_be_restored() {
        let mut session = session(LeniaLabSettings::default());
        session.step(1).unwrap();
        let candidate_state = session.current_state().to_vec();
        session.save_candidate("first").unwrap();
        let retained = session.retained_history_bytes();
        session.step(1).unwrap();
        assert!(session.retained_history_bytes() >= retained);

        session.restore_candidate(0).unwrap();
        assert_eq!(session.generation(), 1);
        assert_eq!(session.current_state(), candidate_state);
        assert_eq!(
            session.retained_history_bytes(),
            2 * 4 * size_of::<f32>() as u64
        );
        assert!(matches!(
            session.save_candidate("first"),
            Err(AutomataLabError::DuplicateCandidateName { .. })
        ));
    }

    #[test]
    fn invalid_settings_and_history_budgets_fail_without_partial_sessions() {
        let mut settings = LeniaLabSettings {
            kernel_size: 4,
            ..LeniaLabSettings::default()
        };
        assert!(matches!(
            AutomataLabSession::new(
                &[0],
                IntensityCalibration::full_scale(256).unwrap(),
                [1, 1],
                settings.clone(),
            ),
            Err(AutomataLabError::InvalidSettings(_))
        ));

        settings.kernel_size = 1;
        settings.max_history_bytes = 3;
        assert!(matches!(
            AutomataLabSession::new(
                &[0],
                IntensityCalibration::full_scale(256).unwrap(),
                [1, 1],
                settings,
            ),
            Err(AutomataLabError::HistoryBudgetExceeded {
                required_bytes: 4,
                budget_bytes: 3,
            })
        ));
    }
}
