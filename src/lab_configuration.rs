//! Reproducible Lab experiments and bounded discrete authoring helpers.
use crate::{
    automata::{
        AutomataCompileLimits, AutomataModelDefinition, AutomataProgramDocument,
        AutomataSimulationResult, compile_automata_program,
    },
    automata_lab::{
        AutomataLabSession, DiscreteLabSettings, LabCapturePolicy, LabExecutionPolicy,
        LeniaLabSettings,
    },
    continuous_automata::{
        CONTINUOUS_RESULT_VERSION, ContinuousDiagnostics, ContinuousDomain,
        ContinuousSimulationResult,
    },
    discrete_automata::{
        DISCRETE_RESULT_VERSION, DiscreteDiagnostics, DiscreteRuleKind, DiscreteSimulationResult,
        KernelOffset, PresenceTransition, SamplePolarity,
    },
    quantization::IntensityCalibration,
    types::{ExperimentConfiguration, validate_experiment_configuration},
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;

pub const MAX_AUTHORING_TABLE_LEN: usize = 65_536;
pub const MAX_LAB_CONFIG_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "model",
    content = "settings",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum LabModelSettings {
    Discrete(DiscreteLabSettings),
    Continuous(LeniaLabSettings),
}
impl LabModelSettings {
    pub fn program_document(
        &self,
        calibration: IntensityCalibration,
    ) -> Result<AutomataProgramDocument, String> {
        match self {
            Self::Discrete(settings) => settings.program_document(calibration),
            Self::Continuous(settings) => settings.program_document(calibration),
        }
        .map_err(|e| e.to_string())
    }
    pub fn validate(&self) -> Result<(), String> {
        if let Self::Discrete(s) = self {
            if !(2..=64).contains(&s.state_count)
                || s.kernel.len() > 225
                || s.kernel
                    .iter()
                    .any(|k| k.dx.abs_diff(0) > 7 || k.dy.abs_diff(0) > 7)
                || [
                    &s.direct_table,
                    &s.aggregate_table,
                    &s.ordered_table,
                    &s.presence_fallback,
                ]
                .iter()
                .any(|t| t.len() > MAX_AUTHORING_TABLE_LEN)
                || s.presence_transitions.len() > 64
            {
                return Err(
                    "Lab configuration exceeds the state, kernel, or rule-table authoring limits."
                        .into(),
                );
            }
        }
        let document = self
            .program_document(IntensityCalibration::full_scale(256).map_err(|e| e.to_string())?)?;
        compile_automata_program(document, AutomataCompileLimits::default())
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabExperimentConfiguration {
    pub schema_version: u32,
    pub preprocessing: ExperimentConfiguration,
    pub automaton: LabModelSettings,
    pub generations: u32,
}
impl LabExperimentConfiguration {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("Unsupported Lab configuration version.".into());
        }
        validate_experiment_configuration(&self.preprocessing)?;
        if self.preprocessing.cellular_automata.is_some() {
            return Err("Lab preparation must not include an Inspect CA run.".into());
        }
        self.automaton.validate()
    }
    pub fn initialize(
        &self,
        values: &[u32],
        calibration: IntensityCalibration,
        size: [usize; 2],
    ) -> Result<AutomataLabSession, String> {
        self.validate()?;
        AutomataLabSession::new(
            values,
            calibration,
            size,
            self.automaton.program_document(calibration)?,
            AutomataCompileLimits::default(),
            LabExecutionPolicy::default(),
            LabCapturePolicy {
                checkpoint_interval: self.generations.max(1),
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())
    }
    pub fn run(
        &self,
        values: &[u32],
        calibration: IntensityCalibration,
        size: [usize; 2],
        cancelled: &AtomicBool,
    ) -> Result<AutomataSimulationResult, String> {
        let mut session = self.initialize(values, calibration, size)?;
        if self.generations > 0 {
            session
                .step_with_cancellation(self.generations, cancelled)
                .map_err(|e| e.to_string())?;
        }
        snapshot_result(&session)
    }
}

/// A numeric projection snapshot, including generation zero. Diagnostic fields
/// describe the snapshot; no invented convergence or cycle claims are attached.
pub fn snapshot_result(session: &AutomataLabSession) -> Result<AutomataSimulationResult, String> {
    let size = session.size();
    let cycles_completed = session.generation();
    match &session.program_document().transition_model {
        AutomataModelDefinition::DiscreteMap(definition) => {
            let state = session
                .discrete_state()
                .ok_or("Missing discrete state")?
                .clone();
            let nonzero = (0..state.len())
                .filter(|i| state.get(*i) != Some(0))
                .count();
            Ok(AutomataSimulationResult::DiscreteMapV1(
                DiscreteSimulationResult {
                    version: DISCRETE_RESULT_VERSION,
                    size,
                    state_count: definition.program.state_count,
                    storage: state.storage(),
                    cycles_completed,
                    final_state: state,
                    diagnostics: DiscreteDiagnostics {
                        transition_steps_executed: 0,
                        distinct_states_observed: None,
                        fixed_point_generation: None,
                        cycle: None,
                        final_nonzero_cells: nonzero,
                        last_transition_changed_cells: 0,
                    },
                },
            ))
        }
        AutomataModelDefinition::ContinuousLenia(_) => {
            let state = session
                .continuous_state()
                .ok_or("Missing continuous state")?;
            let mass: f64 = state.iter().map(|v| *v as f64).sum();
            Ok(AutomataSimulationResult::ContinuousLeniaV1(
                ContinuousSimulationResult {
                    version: CONTINUOUS_RESULT_VERSION,
                    size,
                    domain: ContinuousDomain::UNIT,
                    cycles_completed,
                    final_state: state.to_vec(),
                    diagnostics: ContinuousDiagnostics {
                        transition_steps_executed: 0,
                        exact_fixed_point_generation: None,
                        last_transition_max_delta: None,
                        last_transition_within_tolerance: None,
                        final_minimum: state.iter().copied().fold(f32::INFINITY, f32::min),
                        final_maximum: state.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                        final_mean: (mass / state.len() as f64) as f32,
                        final_mass: mass,
                    },
                },
            ))
        }
        _ => Err("Only Lab models can be captured here.".into()),
    }
}

/// SplitMix64: deterministic exploratory configurations, not cryptographic randomness.
struct Generator(u64);
impl Generator {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}
pub fn random_kernel(
    side: u32,
    seed: u64,
    density: f32,
    negative_fraction: f32,
    max_samples: usize,
) -> Vec<KernelOffset> {
    let side = side.clamp(3, 15) | 1;
    let mut rng = Generator(seed);
    let half = (side / 2) as i32;
    let mut cells: Vec<_> = (0..side * side).map(|i| (rng.next(), i)).collect();
    cells.sort_unstable();
    let count = ((side * side) as f32 * density.clamp(0.0, 1.0)).round() as usize;
    let mut kernel: Vec<_> = cells
        .into_iter()
        .take(count.max(1).min(max_samples.max(1)))
        .map(|(_, i)| KernelOffset {
            dx: (i % side) as i32 - half,
            dy: (i / side) as i32 - half,
            polarity: if (rng.next() as f64 / u64::MAX as f64) < negative_fraction as f64 {
                SamplePolarity::Negative
            } else {
                SamplePolarity::Positive
            },
        })
        .collect();
    kernel.sort_by_key(|k| (k.dy, k.dx));
    kernel
}
pub fn randomize_rule(settings: &mut DiscreteLabSettings, seed: u64) -> Result<(), String> {
    if !(2..=64).contains(&settings.state_count) {
        return Err("State count must be between 2 and 64.".into());
    }
    let len = settings.expected_table_len().map_err(|e| e.to_string())?;
    if len > MAX_AUTHORING_TABLE_LEN {
        return Err("Rule table exceeds 65,536 entries. Reduce states or kernel samples.".into());
    }
    let mut rng = Generator(seed);
    let mut table = || {
        (0..len)
            .map(|_| (rng.next() % settings.state_count as u64) as u32)
            .collect::<Vec<_>>()
    };
    match settings.rule_kind {
        DiscreteRuleKind::Direct => settings.direct_table = table(),
        DiscreteRuleKind::Aggregate => settings.aggregate_table = table(),
        DiscreteRuleKind::OrderedNeighborhood => settings.ordered_table = table(),
        DiscreteRuleKind::PresenceTriggered => {
            settings.presence_fallback = table();
            settings.presence_transitions = (0..settings.state_count)
                .filter_map(|trigger_state| {
                    (rng.next() % 2 == 0).then(|| PresenceTransition {
                        trigger_state,
                        output_state: (rng.next() % settings.state_count as u64) as u32,
                    })
                })
                .collect();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        automata_projection::{
            MaskProjectionSpec, NumericProjectionSpec, OUTPUT_PROJECTION_VERSION, ProjectionLimits,
            project_mask,
        },
        discrete_automata::{AggregateReducer, BoundaryPolicy},
    };
    fn models() -> Vec<LabModelSettings> {
        vec![
            LabModelSettings::Continuous(LeniaLabSettings::default()),
            LabModelSettings::Discrete(DiscreteLabSettings {
                state_count: 3,
                rule_kind: DiscreteRuleKind::Aggregate,
                aggregate_reducer: AggregateReducer::Sum,
                kernel: vec![
                    KernelOffset::new(-1, 0),
                    KernelOffset {
                        dx: 1,
                        dy: 0,
                        polarity: SamplePolarity::Negative,
                    },
                ],
                aggregate_table: vec![0, 2, 1, 0, 2],
                boundary: BoundaryPolicy::Wrap,
                ..Default::default()
            }),
        ]
    }
    #[test]
    fn serialized_experiments_replay_live_states_including_generation_zero() {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        let input = [0, 32, 64, 128, 160, 192, 220, 255];
        for model in models() {
            for generations in [0, 3] {
                let configuration = LabExperimentConfiguration {
                    schema_version: 1,
                    preprocessing: ExperimentConfiguration::default(),
                    automaton: model.clone(),
                    generations,
                };
                let json = serde_json::to_string(&configuration).unwrap();
                let loaded: LabExperimentConfiguration = serde_json::from_str(&json).unwrap();
                let mut live = configuration
                    .initialize(&input, calibration, [4, 2])
                    .unwrap();
                for _ in 0..generations {
                    live.step(1).unwrap();
                }
                let captured = snapshot_result(&live).unwrap();
                let rebuilt = loaded
                    .run(&input, calibration, [4, 2], &AtomicBool::new(false))
                    .unwrap();
                assert_eq!(captured, rebuilt);
                let mask = project_mask(
                    &captured,
                    &MaskProjectionSpec {
                        version: OUTPUT_PROJECTION_VERSION,
                        field: NumericProjectionSpec::normalized_final_state(),
                        minimum: 0.5,
                        maximum: 1.0,
                    },
                    ProjectionLimits::default(),
                )
                .unwrap();
                assert_eq!(mask.values.len(), 8);
            }
        }
    }
    #[test]
    fn random_generators_are_repeatable_bounded_and_compile_for_every_rule() {
        for kind in [
            DiscreteRuleKind::Direct,
            DiscreteRuleKind::Aggregate,
            DiscreteRuleKind::OrderedNeighborhood,
            DiscreteRuleKind::PresenceTriggered,
        ] {
            let kernel = random_kernel(15, 42, 1.0, 0.5, 4);
            assert_eq!(kernel, random_kernel(15, 42, 1.0, 0.5, 4));
            assert_ne!(kernel, random_kernel(15, 43, 1.0, 0.5, 4));
            assert_eq!(kernel.len(), 4);
            let mut s = DiscreteLabSettings {
                state_count: 3,
                kernel,
                rule_kind: kind,
                ..Default::default()
            };
            randomize_rule(&mut s, 42).unwrap();
            let mut again = s.clone();
            randomize_rule(&mut again, 42).unwrap();
            assert_eq!(s, again);
            LabModelSettings::Discrete(s).validate().unwrap();
        }
        assert!(
            random_kernel(3, 1, 0.0, 0.0, 1)
                .iter()
                .all(|k| k.polarity == SamplePolarity::Positive)
        );
        assert!(
            random_kernel(3, 1, 1.0, 1.0, 9)
                .iter()
                .all(|k| k.polarity == SamplePolarity::Negative)
        );
        let mut too_large = DiscreteLabSettings {
            rule_kind: DiscreteRuleKind::OrderedNeighborhood,
            state_count: 64,
            kernel: random_kernel(15, 1, 1.0, 0.5, 225),
            ..Default::default()
        };
        let previous = too_large.clone();
        assert!(randomize_rule(&mut too_large, 1).is_err());
        assert_eq!(previous, too_large);
    }
    #[test]
    fn configuration_rejects_invalid_versions_values_and_cancelled_replay() {
        let mut configuration = LabExperimentConfiguration {
            schema_version: 2,
            preprocessing: ExperimentConfiguration::default(),
            automaton: models().remove(0),
            generations: 10,
        };
        assert!(configuration.validate().is_err());
        configuration.schema_version = 1;
        assert!(
            configuration
                .run(
                    &[0, 128, 255, 32],
                    IntensityCalibration::full_scale(256).unwrap(),
                    [2, 2],
                    &AtomicBool::new(true)
                )
                .is_err()
        );
        configuration.preprocessing.cellular_automata = Some(Default::default());
        assert!(configuration.validate().is_err());
    }
}
