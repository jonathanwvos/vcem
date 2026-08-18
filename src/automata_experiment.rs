use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{
    automata::{
        AUTOMATA_PROGRAM_SCHEMA_VERSION, AutomataCompileLimits, AutomataModelDefinition,
        AutomataProgramDocument, AutomataSimulationResult, LEGACY_EVIDENCE_GROWTH_VERSION,
        validate_automata_program_document,
    },
    automata_projection::validate_simulation_result,
    continuous_automata::{
        CONTINUOUS_INITIALIZER_VERSION, CONTINUOUS_LENIA_PROGRAM_VERSION,
        CONTINUOUS_RUN_CONFIG_VERSION, ContinuousRunConfig,
    },
    discrete_automata::{DISCRETE_PROGRAM_VERSION, DISCRETE_RUN_CONFIG_VERSION, DiscreteRunConfig},
    quantization::{HISTOGRAM_INITIALIZER_VERSION, IntensityCalibration},
};

pub const AUTOMATA_RUN_PROVENANCE_VERSION: u32 = 1;
pub const AUTOMATA_RUN_RECORD_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedSourceReference {
    /// Stable identity of the original source asset or dataset item.
    pub source_id: String,
    /// Stable identity of the exact prepared integer plane used for generation
    /// zero. This binds preprocessing without depending on a mutable file path.
    pub prepared_plane_id: String,
    pub size: [u32; 2],
    pub calibration: IntensityCalibration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoiBounds {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub context_halo: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentExperimentReference {
    pub experiment_id: String,
    pub result_revision: u64,
    pub roi: RoiBounds,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentLineage {
    pub experiment_id: String,
    pub experiment_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<ParentExperimentReference>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "model",
    content = "run",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum AutomataRunSpec {
    /// Legacy iteration count remains in the compatibility program settings.
    LegacyEvidenceGrowthV1,
    DiscreteMapV1(DiscreteRunConfig),
    ContinuousLeniaV1(ContinuousRunConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutomataRunProvenance {
    pub schema_version: u32,
    pub run_id: String,
    pub lineage: ExperimentLineage,
    pub source: PreparedSourceReference,
    pub program: AutomataProgramDocument,
    pub run_spec: AutomataRunSpec,
    /// Current v1 engines are deterministic. `None` records that no runtime
    /// randomness was used; a future stochastic initializer or transition may
    /// place its deterministic seed here without changing the record shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub random_seed: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AutomataRunRecord {
    pub schema_version: u32,
    pub provenance: AutomataRunProvenance,
    pub result: AutomataSimulationResult,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunRecordError {
    UnsupportedProvenanceVersion {
        found: u32,
    },
    UnsupportedProgramVersion {
        found: u32,
    },
    UnsupportedProgramComponentVersion {
        component: &'static str,
        found: u32,
    },
    EmptyIdentifier {
        field: &'static str,
    },
    InvalidSourceSize {
        size: [u32; 2],
    },
    InvalidCalibration(String),
    InvalidProgram(String),
    InvalidRoi {
        reason: &'static str,
    },
    InitializerCalibrationMismatch {
        source: IntensityCalibration,
        initializer: IntensityCalibration,
    },
    ModelRunSpecMismatch {
        program: &'static str,
        run_spec: &'static str,
    },
    ModelResultMismatch {
        program: &'static str,
        result: &'static str,
    },
    UnsupportedRunSpecVersion {
        model: &'static str,
        found: u32,
    },
    ResultSizeMismatch {
        source: [u32; 2],
        result: [usize; 2],
    },
    CompletedCyclesMismatch {
        requested: u32,
        completed: u32,
    },
    DiscreteStateCountMismatch {
        program: u32,
        result: u32,
    },
    ContinuousDomainMismatch,
    InvalidResult(String),
}

impl fmt::Display for RunRecordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProvenanceVersion { found } => {
                write!(
                    formatter,
                    "Unsupported automata run provenance version {found}."
                )
            }
            Self::UnsupportedProgramVersion { found } => {
                write!(formatter, "Unsupported automata program version {found}.")
            }
            Self::UnsupportedProgramComponentVersion { component, found } => {
                write!(formatter, "Unsupported {component} version {found}.")
            }
            Self::EmptyIdentifier { field } => {
                write!(
                    formatter,
                    "Automata run provenance requires a non-empty {field}."
                )
            }
            Self::InvalidSourceSize { size } => write!(
                formatter,
                "Prepared source dimensions must be non-zero; found {} x {}.",
                size[0], size[1]
            ),
            Self::InvalidCalibration(error) => formatter.write_str(error),
            Self::InvalidProgram(error) => write!(formatter, "Invalid automata program: {error}"),
            Self::InvalidRoi { reason } => write!(formatter, "Invalid parent ROI: {reason}."),
            Self::InitializerCalibrationMismatch {
                source,
                initializer,
            } => write!(
                formatter,
                "Prepared source calibration {source:?} does not match initializer calibration {initializer:?}."
            ),
            Self::ModelRunSpecMismatch { program, run_spec } => write!(
                formatter,
                "The {program} program cannot be recorded with the {run_spec} run specification."
            ),
            Self::ModelResultMismatch { program, result } => write!(
                formatter,
                "The {program} program cannot be recorded with the {result} result."
            ),
            Self::UnsupportedRunSpecVersion { model, found } => {
                write!(
                    formatter,
                    "Unsupported {model} run specification version {found}."
                )
            }
            Self::ResultSizeMismatch { source, result } => write!(
                formatter,
                "Result size {} x {} does not match prepared source size {} x {}.",
                result[0], result[1], source[0], source[1]
            ),
            Self::CompletedCyclesMismatch {
                requested,
                completed,
            } => write!(
                formatter,
                "Result completed {completed} cycles, but the run requested {requested}."
            ),
            Self::DiscreteStateCountMismatch { program, result } => write!(
                formatter,
                "Discrete result uses {result} states, but the recorded program uses {program}."
            ),
            Self::ContinuousDomainMismatch => formatter.write_str(
                "Continuous result domain does not match the recorded Lenia program domain.",
            ),
            Self::InvalidResult(error) => write!(formatter, "Invalid simulation result: {error}"),
        }
    }
}

impl std::error::Error for RunRecordError {}

pub fn record_completed_run(
    provenance: AutomataRunProvenance,
    result: AutomataSimulationResult,
) -> Result<AutomataRunRecord, RunRecordError> {
    record_completed_run_with_limits(provenance, result, AutomataCompileLimits::default())
}

pub fn record_completed_run_with_limits(
    provenance: AutomataRunProvenance,
    result: AutomataSimulationResult,
    compile_limits: AutomataCompileLimits,
) -> Result<AutomataRunRecord, RunRecordError> {
    validate_provenance_with_limits(&provenance, compile_limits)?;
    validate_simulation_result(&result)
        .map_err(|error| RunRecordError::InvalidResult(error.to_string()))?;
    validate_result_diagnostics(&result)?;
    validate_result_binding(&provenance, &result)?;
    Ok(AutomataRunRecord {
        schema_version: AUTOMATA_RUN_RECORD_VERSION,
        provenance,
        result,
    })
}

pub fn validate_provenance(provenance: &AutomataRunProvenance) -> Result<(), RunRecordError> {
    validate_provenance_with_limits(provenance, AutomataCompileLimits::default())
}

pub fn validate_provenance_with_limits(
    provenance: &AutomataRunProvenance,
    compile_limits: AutomataCompileLimits,
) -> Result<(), RunRecordError> {
    if provenance.schema_version != AUTOMATA_RUN_PROVENANCE_VERSION {
        return Err(RunRecordError::UnsupportedProvenanceVersion {
            found: provenance.schema_version,
        });
    }
    if provenance.program.schema_version != AUTOMATA_PROGRAM_SCHEMA_VERSION {
        return Err(RunRecordError::UnsupportedProgramVersion {
            found: provenance.program.schema_version,
        });
    }
    require_identifier(&provenance.run_id, "run_id")?;
    require_identifier(&provenance.lineage.experiment_id, "experiment_id")?;
    require_identifier(&provenance.source.source_id, "source_id")?;
    require_identifier(&provenance.source.prepared_plane_id, "prepared_plane_id")?;
    if provenance.source.size[0] == 0 || provenance.source.size[1] == 0 {
        return Err(RunRecordError::InvalidSourceSize {
            size: provenance.source.size,
        });
    }
    provenance
        .source
        .calibration
        .validate()
        .map_err(RunRecordError::InvalidCalibration)?;
    if let Some(parent) = &provenance.lineage.parent {
        require_identifier(&parent.experiment_id, "parent experiment_id")?;
        validate_roi(parent.roi)?;
    }

    match &provenance.program.transition_model {
        AutomataModelDefinition::LegacyEvidenceGrowth(program) => {
            require_component_version(
                "legacy evidence-growth program",
                program.version,
                LEGACY_EVIDENCE_GROWTH_VERSION,
            )?;
            if !matches!(provenance.run_spec, AutomataRunSpec::LegacyEvidenceGrowthV1) {
                return Err(RunRecordError::ModelRunSpecMismatch {
                    program: "legacy evidence-growth",
                    run_spec: run_spec_name(&provenance.run_spec),
                });
            }
        }
        AutomataModelDefinition::DiscreteMap(definition) => {
            require_component_version(
                "histogram state initializer",
                definition.initializer.version,
                HISTOGRAM_INITIALIZER_VERSION,
            )?;
            require_component_version(
                "discrete-map program",
                definition.program.version,
                DISCRETE_PROGRAM_VERSION,
            )?;
            require_initializer_calibration(
                provenance.source.calibration,
                definition.initializer.calibration,
            )?;
            let AutomataRunSpec::DiscreteMapV1(config) = &provenance.run_spec else {
                return Err(RunRecordError::ModelRunSpecMismatch {
                    program: "discrete-map",
                    run_spec: run_spec_name(&provenance.run_spec),
                });
            };
            if config.version != DISCRETE_RUN_CONFIG_VERSION {
                return Err(RunRecordError::UnsupportedRunSpecVersion {
                    model: "discrete-map",
                    found: config.version,
                });
            }
        }
        AutomataModelDefinition::ContinuousLenia(definition) => {
            require_component_version(
                "continuous state initializer",
                definition.initializer.version,
                CONTINUOUS_INITIALIZER_VERSION,
            )?;
            require_component_version(
                "continuous Lenia program",
                definition.program.version,
                CONTINUOUS_LENIA_PROGRAM_VERSION,
            )?;
            require_initializer_calibration(
                provenance.source.calibration,
                definition.initializer.calibration,
            )?;
            let AutomataRunSpec::ContinuousLeniaV1(config) = &provenance.run_spec else {
                return Err(RunRecordError::ModelRunSpecMismatch {
                    program: "continuous-lenia",
                    run_spec: run_spec_name(&provenance.run_spec),
                });
            };
            if config.version != CONTINUOUS_RUN_CONFIG_VERSION {
                return Err(RunRecordError::UnsupportedRunSpecVersion {
                    model: "continuous-lenia",
                    found: config.version,
                });
            }
        }
    }
    validate_automata_program_document(&provenance.program, compile_limits)
        .map_err(|error| RunRecordError::InvalidProgram(error.to_string()))?;
    Ok(())
}

fn validate_result_binding(
    provenance: &AutomataRunProvenance,
    result: &AutomataSimulationResult,
) -> Result<(), RunRecordError> {
    let result_size = result_size(result);
    if usize::try_from(provenance.source.size[0]).ok() != Some(result_size[0])
        || usize::try_from(provenance.source.size[1]).ok() != Some(result_size[1])
    {
        return Err(RunRecordError::ResultSizeMismatch {
            source: provenance.source.size,
            result: result_size,
        });
    }
    match (
        &provenance.program.transition_model,
        &provenance.run_spec,
        result,
    ) {
        (
            AutomataModelDefinition::LegacyEvidenceGrowth(program),
            AutomataRunSpec::LegacyEvidenceGrowthV1,
            AutomataSimulationResult::LegacyEvidenceGrowthV1(result),
        ) => {
            if program.settings.iterations.clamp(1, 512) != result.summary.iterations_run {
                return Err(RunRecordError::InvalidResult(
                    "Legacy iterations run do not match the recorded program.".to_owned(),
                ));
            }
            Ok(())
        }
        (
            AutomataModelDefinition::DiscreteMap(definition),
            AutomataRunSpec::DiscreteMapV1(config),
            AutomataSimulationResult::DiscreteMapV1(result),
        ) => {
            require_completed_cycles(config.cycles, result.cycles_completed)?;
            if definition.program.state_count != result.state_count {
                return Err(RunRecordError::DiscreteStateCountMismatch {
                    program: definition.program.state_count,
                    result: result.state_count,
                });
            }
            Ok(())
        }
        (
            AutomataModelDefinition::ContinuousLenia(definition),
            AutomataRunSpec::ContinuousLeniaV1(config),
            AutomataSimulationResult::ContinuousLeniaV1(result),
        ) => {
            require_completed_cycles(config.cycles, result.cycles_completed)?;
            if definition.program.domain != result.domain {
                return Err(RunRecordError::ContinuousDomainMismatch);
            }
            let expected_within_tolerance = result
                .diagnostics
                .last_transition_max_delta
                .map(|delta| delta <= config.convergence_tolerance);
            if result.diagnostics.last_transition_within_tolerance != expected_within_tolerance {
                return Err(RunRecordError::InvalidResult(
                    "Continuous tolerance diagnostics do not match the recorded run specification."
                        .to_owned(),
                ));
            }
            Ok(())
        }
        (program, _, result) => Err(RunRecordError::ModelResultMismatch {
            program: program_name(program),
            result: result_name(result),
        }),
    }
}

fn validate_result_diagnostics(result: &AutomataSimulationResult) -> Result<(), RunRecordError> {
    let invalid = |message: &str| RunRecordError::InvalidResult(message.to_owned());
    match result {
        AutomataSimulationResult::LegacyEvidenceGrowthV1(result) => {
            let count = result.state.len();
            let mean = result.state.iter().map(|value| *value as f64).sum::<f64>() / count as f64;
            let active =
                result.state.iter().filter(|value| **value >= 0.1).count() as f64 / count as f64;
            if result.summary.iterations_run == 0
                || !approximately_equal(result.summary.mean_activation as f64, mean)
                || !approximately_equal(result.summary.active_fraction as f64, active)
            {
                return Err(invalid(
                    "Legacy result summary does not match its numeric state.",
                ));
            }
        }
        AutomataSimulationResult::DiscreteMapV1(result) => {
            let nonzero = (0..result.final_state.len())
                .filter(|index| {
                    result
                        .final_state
                        .get(*index)
                        .is_some_and(|value| value != 0)
                })
                .count();
            if nonzero != result.diagnostics.final_nonzero_cells
                || result.diagnostics.last_transition_changed_cells > result.final_state.len()
                || result.diagnostics.transition_steps_executed > result.cycles_completed
                || (result.cycles_completed == 0
                    && (result.diagnostics.transition_steps_executed != 0
                        || result.diagnostics.fixed_point_generation.is_some()
                        || result.diagnostics.cycle.is_some()))
                || (result.cycles_completed > 0
                    && (result.diagnostics.transition_steps_executed == 0
                        || (result.diagnostics.last_transition_changed_cells == 0
                            && result.diagnostics.fixed_point_generation.is_none())))
            {
                return Err(invalid(
                    "Discrete result diagnostics do not match its numeric state or cycle count.",
                ));
            }
            if let Some(generation) = result.diagnostics.fixed_point_generation
                && generation >= result.cycles_completed
            {
                return Err(invalid(
                    "Discrete fixed-point generation is outside the completed run.",
                ));
            }
            if let Some(cycle) = result.diagnostics.cycle {
                let expected_repeat = cycle.start_generation.checked_add(cycle.period);
                if cycle.period == 0
                    || expected_repeat != Some(cycle.first_repeat_generation)
                    || cycle.first_repeat_generation > result.cycles_completed
                {
                    return Err(invalid(
                        "Discrete cycle diagnostics are internally inconsistent.",
                    ));
                }
                if result.diagnostics.distinct_states_observed
                    != Some(u64::from(cycle.first_repeat_generation))
                {
                    return Err(invalid(
                        "Discrete distinct-state diagnostics disagree with the detected cycle.",
                    ));
                }
                match (cycle.period, result.diagnostics.fixed_point_generation) {
                    (1, Some(generation)) if generation == cycle.start_generation => {}
                    (1, _) => {
                        return Err(invalid(
                            "Discrete fixed-point and period-one diagnostics disagree.",
                        ));
                    }
                    (_, Some(_)) => {
                        return Err(invalid(
                            "Discrete fixed-point diagnostics cannot describe a multi-state cycle.",
                        ));
                    }
                    (_, None) => {}
                }
            } else if result.diagnostics.fixed_point_generation.is_some() {
                return Err(invalid(
                    "Discrete fixed-point diagnostics require a matching period-one cycle.",
                ));
            }
        }
        AutomataSimulationResult::ContinuousLeniaV1(result) => {
            let mut minimum = f32::INFINITY;
            let mut maximum = f32::NEG_INFINITY;
            let mut mass = 0.0_f64;
            for value in &result.final_state {
                minimum = minimum.min(*value);
                maximum = maximum.max(*value);
                mass += *value as f64;
            }
            let mean = mass / result.final_state.len() as f64;
            if result.diagnostics.transition_steps_executed > result.cycles_completed
                || result.diagnostics.final_minimum != minimum
                || result.diagnostics.final_maximum != maximum
                || !approximately_equal(result.diagnostics.final_mean as f64, mean)
                || !approximately_equal(result.diagnostics.final_mass, mass)
            {
                return Err(invalid(
                    "Continuous result diagnostics do not match its numeric state or cycle count.",
                ));
            }
            if let Some(generation) = result.diagnostics.exact_fixed_point_generation
                && generation >= result.cycles_completed
            {
                return Err(invalid(
                    "Continuous fixed-point generation is outside the completed run.",
                ));
            }
            match (
                result.diagnostics.transition_steps_executed,
                result.diagnostics.last_transition_max_delta,
                result.diagnostics.last_transition_within_tolerance,
                result.diagnostics.exact_fixed_point_generation,
            ) {
                (0, None, None, None) if result.cycles_completed == 0 => {}
                (0, _, _, _) => {
                    return Err(invalid(
                        "Continuous zero-step diagnostics contain transition measurements.",
                    ));
                }
                (_, Some(delta), Some(_), fixed) => {
                    if (delta == 0.0) != fixed.is_some() {
                        return Err(invalid(
                            "Continuous exact fixed-point diagnostics disagree with the final transition delta.",
                        ));
                    }
                    if let Some(generation) = fixed
                        && generation.checked_add(1)
                            != Some(result.diagnostics.transition_steps_executed)
                    {
                        return Err(invalid(
                            "Continuous fixed-point generation disagrees with transitions executed.",
                        ));
                    }
                }
                _ => {
                    return Err(invalid(
                        "Continuous transition diagnostics are only partially populated.",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1.0e-6 * scale
}

fn require_identifier(value: &str, field: &'static str) -> Result<(), RunRecordError> {
    if value.trim().is_empty() {
        return Err(RunRecordError::EmptyIdentifier { field });
    }
    Ok(())
}

fn require_component_version(
    component: &'static str,
    found: u32,
    expected: u32,
) -> Result<(), RunRecordError> {
    if found != expected {
        return Err(RunRecordError::UnsupportedProgramComponentVersion { component, found });
    }
    Ok(())
}

fn validate_roi(roi: RoiBounds) -> Result<(), RunRecordError> {
    if roi.width == 0 || roi.height == 0 {
        return Err(RunRecordError::InvalidRoi {
            reason: "width and height must be non-zero",
        });
    }
    roi.x
        .checked_add(roi.width)
        .ok_or(RunRecordError::InvalidRoi {
            reason: "horizontal bounds overflow u32",
        })?;
    roi.y
        .checked_add(roi.height)
        .ok_or(RunRecordError::InvalidRoi {
            reason: "vertical bounds overflow u32",
        })?;
    Ok(())
}

fn require_initializer_calibration(
    source: IntensityCalibration,
    initializer: IntensityCalibration,
) -> Result<(), RunRecordError> {
    if source != initializer {
        return Err(RunRecordError::InitializerCalibrationMismatch {
            source,
            initializer,
        });
    }
    Ok(())
}

fn require_completed_cycles(requested: u32, completed: u32) -> Result<(), RunRecordError> {
    if requested != completed {
        return Err(RunRecordError::CompletedCyclesMismatch {
            requested,
            completed,
        });
    }
    Ok(())
}

fn result_size(result: &AutomataSimulationResult) -> [usize; 2] {
    match result {
        AutomataSimulationResult::LegacyEvidenceGrowthV1(result) => result.size,
        AutomataSimulationResult::DiscreteMapV1(result) => result.size,
        AutomataSimulationResult::ContinuousLeniaV1(result) => result.size,
    }
}

fn program_name(program: &AutomataModelDefinition) -> &'static str {
    match program {
        AutomataModelDefinition::LegacyEvidenceGrowth(_) => "legacy evidence-growth",
        AutomataModelDefinition::DiscreteMap(_) => "discrete-map",
        AutomataModelDefinition::ContinuousLenia(_) => "continuous-lenia",
    }
}

fn run_spec_name(run_spec: &AutomataRunSpec) -> &'static str {
    match run_spec {
        AutomataRunSpec::LegacyEvidenceGrowthV1 => "legacy evidence-growth",
        AutomataRunSpec::DiscreteMapV1(_) => "discrete-map",
        AutomataRunSpec::ContinuousLeniaV1(_) => "continuous-lenia",
    }
}

fn result_name(result: &AutomataSimulationResult) -> &'static str {
    match result {
        AutomataSimulationResult::LegacyEvidenceGrowthV1(_) => "legacy evidence-growth",
        AutomataSimulationResult::DiscreteMapV1(_) => "discrete-map",
        AutomataSimulationResult::ContinuousLeniaV1(_) => "continuous-lenia",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        automata::{
            AUTOMATA_PROGRAM_SCHEMA_VERSION, AutomataModelDefinition,
            ContinuousLeniaProgramDefinition, DiscreteMapProgram,
        },
        continuous_automata::{
            CONTINUOUS_INITIALIZER_VERSION, CONTINUOUS_LENIA_PROGRAM_VERSION,
            CONTINUOUS_RESULT_VERSION, ContinuousBoundaryPolicy, ContinuousDiagnostics,
            ContinuousDomain, ContinuousGrowthFunction, ContinuousLeniaProgram,
            ContinuousSimulationResult, ContinuousStateInitializer, DenseContinuousKernel,
            KernelNormalization,
        },
        discrete_automata::{
            BoundaryPolicy, CompactState, DISCRETE_PROGRAM_VERSION, DISCRETE_RESULT_VERSION,
            DiscreteDiagnostics, DiscreteProgram, DiscreteRule, DiscreteRunConfig,
            DiscreteSimulationResult, StateStorage,
        },
        quantization::HistogramStateInitializer,
    };

    fn continuous_provenance() -> AutomataRunProvenance {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        AutomataRunProvenance {
            schema_version: AUTOMATA_RUN_PROVENANCE_VERSION,
            run_id: "run-0001".to_owned(),
            lineage: ExperimentLineage {
                experiment_id: "experiment-child".to_owned(),
                experiment_revision: 3,
                parent: Some(ParentExperimentReference {
                    experiment_id: "experiment-parent".to_owned(),
                    result_revision: 2,
                    roi: RoiBounds {
                        x: 10,
                        y: 20,
                        width: 2,
                        height: 1,
                        context_halo: 4,
                    },
                }),
            },
            source: PreparedSourceReference {
                source_id: "sha256:source".to_owned(),
                prepared_plane_id: "sha256:prepared-plane".to_owned(),
                size: [2, 1],
                calibration,
            },
            program: AutomataProgramDocument {
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
                                weights: vec![1.0],
                            },
                            kernel_normalization: KernelNormalization::SumToOne,
                            boundary: ContinuousBoundaryPolicy::Wrap,
                            growth: ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1),
                            time_step: 0.1,
                        },
                    },
                ),
            },
            run_spec: AutomataRunSpec::ContinuousLeniaV1(ContinuousRunConfig::new(2)),
            random_seed: None,
        }
    }

    fn continuous_result(cycles: u32) -> AutomataSimulationResult {
        AutomataSimulationResult::ContinuousLeniaV1(ContinuousSimulationResult {
            version: CONTINUOUS_RESULT_VERSION,
            size: [2, 1],
            domain: ContinuousDomain::UNIT,
            cycles_completed: cycles,
            final_state: vec![0.25, 0.75],
            diagnostics: ContinuousDiagnostics {
                transition_steps_executed: cycles,
                exact_fixed_point_generation: None,
                last_transition_max_delta: Some(0.1),
                last_transition_within_tolerance: Some(false),
                final_minimum: 0.25,
                final_maximum: 0.75,
                final_mean: 0.5,
                final_mass: 1.0,
            },
        })
    }

    fn discrete_provenance() -> AutomataRunProvenance {
        let calibration = IntensityCalibration::full_scale(256).unwrap();
        AutomataRunProvenance {
            schema_version: AUTOMATA_RUN_PROVENANCE_VERSION,
            run_id: "run-discrete".to_owned(),
            lineage: ExperimentLineage {
                experiment_id: "experiment-discrete".to_owned(),
                experiment_revision: 1,
                parent: None,
            },
            source: PreparedSourceReference {
                source_id: "sha256:source".to_owned(),
                prepared_plane_id: "sha256:prepared-plane".to_owned(),
                size: [2, 1],
                calibration,
            },
            program: AutomataProgramDocument {
                schema_version: AUTOMATA_PROGRAM_SCHEMA_VERSION,
                transition_model: AutomataModelDefinition::DiscreteMap(DiscreteMapProgram {
                    initializer: HistogramStateInitializer::identity(calibration, 2),
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
            },
            run_spec: AutomataRunSpec::DiscreteMapV1(DiscreteRunConfig::new(2)),
            random_seed: None,
        }
    }

    fn discrete_result(state_count: u32) -> AutomataSimulationResult {
        AutomataSimulationResult::DiscreteMapV1(DiscreteSimulationResult {
            version: DISCRETE_RESULT_VERSION,
            size: [2, 1],
            state_count,
            storage: StateStorage::U8,
            cycles_completed: 2,
            final_state: CompactState::U8(vec![0, 1]),
            diagnostics: DiscreteDiagnostics {
                transition_steps_executed: 2,
                distinct_states_observed: None,
                fixed_point_generation: None,
                cycle: None,
                final_nonzero_cells: 1,
                last_transition_changed_cells: 1,
            },
        })
    }

    #[test]
    fn provenance_round_trips_and_completed_record_binds_every_snapshot() {
        let provenance = continuous_provenance();
        let json = serde_json::to_string(&provenance).unwrap();
        let decoded: AutomataRunProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, provenance);

        let record = record_completed_run(provenance.clone(), continuous_result(2)).unwrap();
        assert_eq!(record.schema_version, AUTOMATA_RUN_RECORD_VERSION);
        assert_eq!(record.provenance, provenance);
        let record_json = serde_json::to_string(&record).unwrap();
        assert!(record_json.contains("\"model\":\"continuous_lenia\""));
        assert!(record_json.contains("\"prepared_plane_id\":\"sha256:prepared-plane\""));
        assert!(record_json.contains("\"experiment-parent\""));
    }

    #[test]
    fn provenance_rejects_empty_ids_invalid_rois_and_calibration_drift() {
        let mut empty_id = continuous_provenance();
        empty_id.run_id = "  ".to_owned();
        assert!(matches!(
            validate_provenance(&empty_id),
            Err(RunRecordError::EmptyIdentifier { field: "run_id" })
        ));

        let mut invalid_roi = continuous_provenance();
        invalid_roi.lineage.parent.as_mut().unwrap().roi.width = 0;
        assert!(matches!(
            validate_provenance(&invalid_roi),
            Err(RunRecordError::InvalidRoi { .. })
        ));

        let mut drifted = continuous_provenance();
        drifted.source.calibration = IntensityCalibration::full_scale(65_536).unwrap();
        assert!(matches!(
            validate_provenance(&drifted),
            Err(RunRecordError::InitializerCalibrationMismatch { .. })
        ));

        let mut unknown_program = continuous_provenance();
        let AutomataModelDefinition::ContinuousLenia(definition) =
            &mut unknown_program.program.transition_model
        else {
            unreachable!()
        };
        definition.program.version = 99;
        assert!(matches!(
            validate_provenance(&unknown_program),
            Err(RunRecordError::UnsupportedProgramComponentVersion {
                component: "continuous Lenia program",
                found: 99,
            })
        ));
    }

    #[test]
    fn record_rejects_model_cycle_size_and_numeric_result_mismatches() {
        assert!(matches!(
            record_completed_run(continuous_provenance(), continuous_result(1)),
            Err(RunRecordError::CompletedCyclesMismatch {
                requested: 2,
                completed: 1,
            })
        ));

        let mut wrong_run = continuous_provenance();
        wrong_run.run_spec = AutomataRunSpec::DiscreteMapV1(DiscreteRunConfig::new(2));
        assert!(matches!(
            record_completed_run(wrong_run, continuous_result(2)),
            Err(RunRecordError::ModelRunSpecMismatch { .. })
        ));

        let mut wrong_size = continuous_result(2);
        let AutomataSimulationResult::ContinuousLeniaV1(result) = &mut wrong_size else {
            unreachable!()
        };
        result.size = [1, 2];
        assert!(matches!(
            record_completed_run(continuous_provenance(), wrong_size),
            Err(RunRecordError::ResultSizeMismatch { .. })
        ));

        let mut invalid_numeric = continuous_result(2);
        let AutomataSimulationResult::ContinuousLeniaV1(result) = &mut invalid_numeric else {
            unreachable!()
        };
        result.final_state[0] = f32::NAN;
        assert!(matches!(
            record_completed_run(continuous_provenance(), invalid_numeric),
            Err(RunRecordError::InvalidResult(_))
        ));
    }

    #[test]
    fn record_rejects_invalid_programs_result_domains_state_counts_and_diagnostics() {
        let mut invalid_program = continuous_provenance();
        let AutomataModelDefinition::ContinuousLenia(definition) =
            &mut invalid_program.program.transition_model
        else {
            unreachable!()
        };
        definition.program.kernel.weights.clear();
        assert!(matches!(
            record_completed_run(invalid_program, continuous_result(2)),
            Err(RunRecordError::InvalidProgram(_))
        ));

        let mut wrong_domain = continuous_result(2);
        let AutomataSimulationResult::ContinuousLeniaV1(result) = &mut wrong_domain else {
            unreachable!()
        };
        result.domain = ContinuousDomain {
            minimum: -1.0,
            maximum: 1.0,
        };
        assert!(matches!(
            record_completed_run(continuous_provenance(), wrong_domain),
            Err(RunRecordError::ContinuousDomainMismatch)
        ));

        assert!(matches!(
            record_completed_run(discrete_provenance(), discrete_result(3)),
            Err(RunRecordError::DiscreteStateCountMismatch {
                program: 2,
                result: 3,
            })
        ));

        let mut inconsistent = continuous_result(2);
        let AutomataSimulationResult::ContinuousLeniaV1(result) = &mut inconsistent else {
            unreachable!()
        };
        result.diagnostics.final_mass = 9.0;
        assert!(matches!(
            record_completed_run(continuous_provenance(), inconsistent),
            Err(RunRecordError::InvalidResult(_))
        ));
    }

    #[test]
    fn provenance_json_is_strict() {
        let mut value = serde_json::to_value(continuous_provenance()).unwrap();
        value["source"]["surprise"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<AutomataRunProvenance>(value).is_err());
    }
}
