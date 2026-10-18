use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fmt, mem::size_of};

use crate::{
    automata::AutomataSimulationResult,
    continuous_automata::{CONTINUOUS_RESULT_VERSION, ContinuousDomain},
    discrete_automata::{CompactState, DISCRETE_RESULT_VERSION, StateStorage},
};

pub const OUTPUT_PROJECTION_VERSION: u32 = 1;
pub const PROJECTED_FIELD_VERSION: u32 = 1;

const DEFAULT_PROJECTION_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProjectionSource {
    NormalizedFinalState,
    DiscreteStateMembership { states: Vec<u32> },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProjectionTransfer {
    Identity,
    Power { exponent: f32 },
    Window { lower: f32, upper: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericProjectionSpec {
    pub version: u32,
    pub source: ProjectionSource,
    pub transfer: ProjectionTransfer,
}

impl NumericProjectionSpec {
    pub fn normalized_final_state() -> Self {
        Self {
            version: OUTPUT_PROJECTION_VERSION,
            source: ProjectionSource::NormalizedFinalState,
            transfer: ProjectionTransfer::Identity,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskProjectionSpec {
    pub version: u32,
    pub field: NumericProjectionSpec,
    pub minimum: f32,
    pub maximum: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayProjectionSpec {
    pub version: u32,
    pub field: NumericProjectionSpec,
    pub color: [u8; 3],
    pub maximum_alpha: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionLimits {
    /// Maximum logical bytes in the returned projection, excluding allocator
    /// metadata. Projection streams directly from the numeric simulation
    /// result and does not allocate an intermediate field.
    pub max_output_bytes: u64,
}

impl Default for ProjectionLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: DEFAULT_PROJECTION_BUDGET_BYTES,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NumericField {
    pub version: u32,
    pub size: [usize; 2],
    pub values: Vec<f32>,
    pub minimum: f32,
    pub maximum: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MaskField {
    pub version: u32,
    pub size: [usize; 2],
    /// One byte per cell: zero is excluded and one is included.
    pub values: Vec<u8>,
    pub included_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RgbaOverlay {
    pub version: u32,
    pub size: [usize; 2],
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionError {
    UnsupportedProjectionVersion {
        found: u32,
    },
    UnsupportedResultVersion {
        model: &'static str,
        found: u32,
    },
    InvalidTransfer(String),
    InvalidMaskRange {
        minimum: f32,
        maximum: f32,
    },
    ProjectionRequiresDiscreteResult,
    DuplicateDiscreteState {
        state: u32,
    },
    DiscreteStateOutOfRange {
        state: u32,
        state_count: u32,
    },
    InvalidResult(String),
    SizeOverflow {
        context: &'static str,
    },
    BudgetExceeded {
        context: &'static str,
        required_bytes: u64,
        budget_bytes: u64,
    },
    AllocationFailed {
        context: &'static str,
        bytes: u64,
    },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProjectionVersion { found } => {
                write!(formatter, "Unsupported output projection version {found}.")
            }
            Self::UnsupportedResultVersion { model, found } => {
                write!(formatter, "Unsupported {model} result version {found}.")
            }
            Self::InvalidTransfer(error) | Self::InvalidResult(error) => formatter.write_str(error),
            Self::InvalidMaskRange { minimum, maximum } => write!(
                formatter,
                "Mask thresholds must be finite and satisfy 0 <= minimum <= maximum <= 1; found [{minimum}, {maximum}]."
            ),
            Self::ProjectionRequiresDiscreteResult => {
                formatter.write_str("Discrete state membership requires a discrete-map result.")
            }
            Self::DuplicateDiscreteState { state } => {
                write!(
                    formatter,
                    "Discrete projection state {state} is duplicated."
                )
            }
            Self::DiscreteStateOutOfRange { state, state_count } => write!(
                formatter,
                "Discrete projection state {state} is outside state_count {state_count}."
            ),
            Self::SizeOverflow { context } => {
                write!(formatter, "{context} size overflowed addressable memory.")
            }
            Self::BudgetExceeded {
                context,
                required_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "{context} requires {required_bytes} bytes, exceeding the {budget_bytes}-byte budget."
            ),
            Self::AllocationFailed { context, bytes } => {
                write!(formatter, "Could not allocate {bytes} bytes for {context}.")
            }
        }
    }
}

impl std::error::Error for ProjectionError {}

pub fn project_numeric_field(
    result: &AutomataSimulationResult,
    spec: &NumericProjectionSpec,
    limits: ProjectionLimits,
) -> Result<NumericField, ProjectionError> {
    let validated = ValidatedProjection::new(result, spec)?;
    let size = result_size(result);
    let cell_count = validated.cell_count;
    let required_bytes = checked_bytes::<f32>(cell_count, "numeric projection")?;
    enforce_budget(
        "numeric projection",
        required_bytes,
        limits.max_output_bytes,
    )?;
    let mut values = try_vec_with_capacity(cell_count, required_bytes, "numeric projection")?;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    validated.visit(|value| {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        values.push(value);
    });
    Ok(NumericField {
        version: PROJECTED_FIELD_VERSION,
        size,
        values,
        minimum,
        maximum,
    })
}

/// Validates the structural and state-domain invariants required by every
/// downstream projection or persisted run record without allocating output.
pub fn validate_simulation_result(
    result: &AutomataSimulationResult,
) -> Result<(), ProjectionError> {
    validate_result(result).map(|_| ())
}

pub fn project_mask(
    result: &AutomataSimulationResult,
    spec: &MaskProjectionSpec,
    limits: ProjectionLimits,
) -> Result<MaskField, ProjectionError> {
    if spec.version != OUTPUT_PROJECTION_VERSION {
        return Err(ProjectionError::UnsupportedProjectionVersion {
            found: spec.version,
        });
    }
    if !spec.minimum.is_finite()
        || !spec.maximum.is_finite()
        || spec.minimum < 0.0
        || spec.maximum > 1.0
        || spec.minimum > spec.maximum
    {
        return Err(ProjectionError::InvalidMaskRange {
            minimum: spec.minimum,
            maximum: spec.maximum,
        });
    }
    let validated = ValidatedProjection::new(result, &spec.field)?;
    let required_bytes = checked_bytes::<u8>(validated.cell_count, "mask projection")?;
    enforce_budget("mask projection", required_bytes, limits.max_output_bytes)?;
    let mut values =
        try_vec_with_capacity(validated.cell_count, required_bytes, "mask projection")?;
    let mut included_cells = 0_usize;
    validated.visit(|value| {
        let included = u8::from((spec.minimum..=spec.maximum).contains(&value));
        included_cells += included as usize;
        values.push(included);
    });
    Ok(MaskField {
        version: PROJECTED_FIELD_VERSION,
        size: result_size(result),
        values,
        included_cells,
    })
}

pub fn project_rgba_overlay(
    result: &AutomataSimulationResult,
    spec: &OverlayProjectionSpec,
    limits: ProjectionLimits,
) -> Result<RgbaOverlay, ProjectionError> {
    if spec.version != OUTPUT_PROJECTION_VERSION {
        return Err(ProjectionError::UnsupportedProjectionVersion {
            found: spec.version,
        });
    }
    let validated = ValidatedProjection::new(result, &spec.field)?;
    let rgba_len = validated
        .cell_count
        .checked_mul(4)
        .ok_or(ProjectionError::SizeOverflow {
            context: "RGBA overlay",
        })?;
    let required_bytes = u64::try_from(rgba_len).map_err(|_| ProjectionError::SizeOverflow {
        context: "RGBA overlay",
    })?;
    enforce_budget("RGBA overlay", required_bytes, limits.max_output_bytes)?;
    let mut rgba = try_vec_with_capacity(rgba_len, required_bytes, "RGBA overlay")?;
    validated.visit(|value| {
        let alpha = (value * spec.maximum_alpha as f32)
            .round()
            .clamp(0.0, spec.maximum_alpha as f32) as u8;
        rgba.extend_from_slice(&[spec.color[0], spec.color[1], spec.color[2], alpha]);
    });
    Ok(RgbaOverlay {
        version: PROJECTED_FIELD_VERSION,
        size: result_size(result),
        rgba,
    })
}

/// Projects a borrowed continuous state without constructing or cloning a
/// persisted simulation result. This is used by interactive playback while
/// retaining the same versioned transfer and overlay semantics.
pub fn project_continuous_state_overlay(
    state: &[f32],
    size: [usize; 2],
    domain: ContinuousDomain,
    spec: &OverlayProjectionSpec,
    limits: ProjectionLimits,
) -> Result<RgbaOverlay, ProjectionError> {
    if spec.version != OUTPUT_PROJECTION_VERSION || spec.field.version != OUTPUT_PROJECTION_VERSION
    {
        return Err(ProjectionError::UnsupportedProjectionVersion {
            found: if spec.version != OUTPUT_PROJECTION_VERSION {
                spec.version
            } else {
                spec.field.version
            },
        });
    }
    if !matches!(spec.field.source, ProjectionSource::NormalizedFinalState) {
        return Err(ProjectionError::InvalidResult(
            "Borrowed continuous-state projection requires normalized_final_state.".to_owned(),
        ));
    }
    validate_transfer(spec.field.transfer)?;
    domain
        .validate()
        .map_err(|error| ProjectionError::InvalidResult(error.to_string()))?;
    let cell_count = size[0]
        .checked_mul(size[1])
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            ProjectionError::InvalidResult(
                "Continuous projection dimensions must be non-zero and addressable.".to_owned(),
            )
        })?;
    if state.len() != cell_count {
        return Err(ProjectionError::InvalidResult(format!(
            "Continuous projection has {} values; expected {cell_count}.",
            state.len()
        )));
    }
    if let Some((index, value)) = state
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !domain.contains(*value))
    {
        return Err(ProjectionError::InvalidResult(format!(
            "Continuous projection value {value} at index {index} is outside its domain."
        )));
    }
    let rgba_len = cell_count
        .checked_mul(4)
        .ok_or(ProjectionError::SizeOverflow {
            context: "RGBA overlay",
        })?;
    let required_bytes = u64::try_from(rgba_len).map_err(|_| ProjectionError::SizeOverflow {
        context: "RGBA overlay",
    })?;
    enforce_budget("RGBA overlay", required_bytes, limits.max_output_bytes)?;
    let mut rgba = try_vec_with_capacity(rgba_len, required_bytes, "RGBA overlay")?;
    let domain_width = domain.maximum as f64 - domain.minimum as f64;
    for value in state {
        let normalized = ((*value as f64 - domain.minimum as f64) / domain_width) as f32;
        let projected = apply_transfer(normalized, spec.field.transfer);
        let alpha = (projected * spec.maximum_alpha as f32)
            .round()
            .clamp(0.0, spec.maximum_alpha as f32) as u8;
        rgba.extend_from_slice(&[spec.color[0], spec.color[1], spec.color[2], alpha]);
    }
    Ok(RgbaOverlay {
        version: PROJECTED_FIELD_VERSION,
        size,
        rgba,
    })
}

/// Projects a borrowed discrete state without constructing or cloning a
/// persisted simulation result. This mirrors
/// [`project_continuous_state_overlay`] for interactive discrete playback.
pub fn project_discrete_state_overlay(
    state: &CompactState,
    size: [usize; 2],
    state_count: u32,
    spec: &OverlayProjectionSpec,
    limits: ProjectionLimits,
) -> Result<RgbaOverlay, ProjectionError> {
    if spec.version != OUTPUT_PROJECTION_VERSION || spec.field.version != OUTPUT_PROJECTION_VERSION
    {
        return Err(ProjectionError::UnsupportedProjectionVersion {
            found: if spec.version != OUTPUT_PROJECTION_VERSION {
                spec.version
            } else {
                spec.field.version
            },
        });
    }
    validate_transfer(spec.field.transfer)?;
    let expected_storage = StateStorage::for_state_count(state_count)
        .map_err(|error| ProjectionError::InvalidResult(error.to_string()))?;
    if state.storage() != expected_storage {
        return Err(ProjectionError::InvalidResult(
            "Discrete projection storage is not canonical for its state count.".to_owned(),
        ));
    }
    let cell_count = size[0]
        .checked_mul(size[1])
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            ProjectionError::InvalidResult(
                "Discrete projection dimensions must be non-zero and addressable.".to_owned(),
            )
        })?;
    if state.len() != cell_count {
        return Err(ProjectionError::InvalidResult(format!(
            "Discrete projection has {} values; expected {cell_count}.",
            state.len()
        )));
    }
    let membership = match &spec.field.source {
        ProjectionSource::NormalizedFinalState => None,
        ProjectionSource::DiscreteStateMembership { states } => {
            let mut unique = BTreeSet::new();
            for value in states {
                if *value >= state_count {
                    return Err(ProjectionError::DiscreteStateOutOfRange {
                        state: *value,
                        state_count,
                    });
                }
                if !unique.insert(*value) {
                    return Err(ProjectionError::DuplicateDiscreteState { state: *value });
                }
            }
            Some(unique)
        }
    };
    for index in 0..cell_count {
        let value = state.get(index).unwrap();
        if value >= state_count {
            return Err(ProjectionError::InvalidResult(format!(
                "Discrete projection state {value} at index {index} is outside state_count {state_count}."
            )));
        }
    }
    let rgba_len = cell_count
        .checked_mul(4)
        .ok_or(ProjectionError::SizeOverflow {
            context: "RGBA overlay",
        })?;
    let required_bytes = u64::try_from(rgba_len).map_err(|_| ProjectionError::SizeOverflow {
        context: "RGBA overlay",
    })?;
    enforce_budget("RGBA overlay", required_bytes, limits.max_output_bytes)?;
    let mut rgba = try_vec_with_capacity(rgba_len, required_bytes, "RGBA overlay")?;
    for index in 0..cell_count {
        let value = state.get(index).unwrap();
        let normalized = match &membership {
            None => discrete_normalized_value(value, state_count),
            Some(states) => discrete_membership_value(value, states),
        };
        let projected = apply_transfer(normalized, spec.field.transfer);
        let alpha = (projected * spec.maximum_alpha as f32)
            .round()
            .clamp(0.0, spec.maximum_alpha as f32) as u8;
        rgba.extend_from_slice(&[spec.color[0], spec.color[1], spec.color[2], alpha]);
    }
    Ok(RgbaOverlay {
        version: PROJECTED_FIELD_VERSION,
        size,
        rgba,
    })
}

fn discrete_normalized_value(state: u32, state_count: u32) -> f32 {
    state as f32 / (state_count - 1) as f32
}

fn discrete_membership_value(state: u32, states: &BTreeSet<u32>) -> f32 {
    f32::from(states.contains(&state))
}

struct ValidatedProjection<'a> {
    result: &'a AutomataSimulationResult,
    source: ValidatedSource,
    transfer: ProjectionTransfer,
    cell_count: usize,
}

enum ValidatedSource {
    NormalizedFinalState,
    DiscreteStateMembership(BTreeSet<u32>),
}

impl<'a> ValidatedProjection<'a> {
    fn new(
        result: &'a AutomataSimulationResult,
        spec: &NumericProjectionSpec,
    ) -> Result<Self, ProjectionError> {
        if spec.version != OUTPUT_PROJECTION_VERSION {
            return Err(ProjectionError::UnsupportedProjectionVersion {
                found: spec.version,
            });
        }
        validate_transfer(spec.transfer)?;
        let cell_count = validate_result(result)?;
        let source = match &spec.source {
            ProjectionSource::NormalizedFinalState => ValidatedSource::NormalizedFinalState,
            ProjectionSource::DiscreteStateMembership { states } => {
                let AutomataSimulationResult::DiscreteMapV1(discrete) = result else {
                    return Err(ProjectionError::ProjectionRequiresDiscreteResult);
                };
                let mut unique = BTreeSet::new();
                for state in states {
                    if *state >= discrete.state_count {
                        return Err(ProjectionError::DiscreteStateOutOfRange {
                            state: *state,
                            state_count: discrete.state_count,
                        });
                    }
                    if !unique.insert(*state) {
                        return Err(ProjectionError::DuplicateDiscreteState { state: *state });
                    }
                }
                ValidatedSource::DiscreteStateMembership(unique)
            }
        };
        Ok(Self {
            result,
            source,
            transfer: spec.transfer,
            cell_count,
        })
    }

    fn visit(&self, mut visitor: impl FnMut(f32)) {
        match (self.result, &self.source) {
            (
                AutomataSimulationResult::LegacyEvidenceGrowthV1(result),
                ValidatedSource::NormalizedFinalState,
            ) => {
                for value in &result.state {
                    visitor(apply_transfer(*value, self.transfer));
                }
            }
            (
                AutomataSimulationResult::ContinuousLeniaV1(result),
                ValidatedSource::NormalizedFinalState,
            ) => {
                let domain_width = result.domain.maximum as f64 - result.domain.minimum as f64;
                for value in &result.final_state {
                    let normalized =
                        ((*value as f64 - result.domain.minimum as f64) / domain_width) as f32;
                    visitor(apply_transfer(normalized, self.transfer));
                }
            }
            (
                AutomataSimulationResult::DiscreteMapV1(result),
                ValidatedSource::NormalizedFinalState,
            ) => {
                for index in 0..result.final_state.len() {
                    let state = result.final_state.get(index).unwrap();
                    visitor(apply_transfer(
                        discrete_normalized_value(state, result.state_count),
                        self.transfer,
                    ));
                }
            }
            (
                AutomataSimulationResult::DiscreteMapV1(result),
                ValidatedSource::DiscreteStateMembership(states),
            ) => {
                for index in 0..result.final_state.len() {
                    let state = result.final_state.get(index).unwrap();
                    visitor(apply_transfer(
                        discrete_membership_value(state, states),
                        self.transfer,
                    ));
                }
            }
            _ => unreachable!("projection source and result were validated together"),
        }
    }
}

fn validate_transfer(transfer: ProjectionTransfer) -> Result<(), ProjectionError> {
    match transfer {
        ProjectionTransfer::Identity => Ok(()),
        ProjectionTransfer::Power { exponent } if exponent.is_finite() && exponent > 0.0 => Ok(()),
        ProjectionTransfer::Window { lower, upper }
            if lower.is_finite()
                && upper.is_finite()
                && lower >= 0.0
                && upper <= 1.0
                && lower < upper =>
        {
            Ok(())
        }
        ProjectionTransfer::Power { .. } => Err(ProjectionError::InvalidTransfer(
            "Power projection requires a finite exponent greater than zero.".to_owned(),
        )),
        ProjectionTransfer::Window { .. } => Err(ProjectionError::InvalidTransfer(
            "Window projection requires finite bounds satisfying 0 <= lower < upper <= 1."
                .to_owned(),
        )),
    }
}

fn apply_transfer(value: f32, transfer: ProjectionTransfer) -> f32 {
    let value = value.clamp(0.0, 1.0);
    match transfer {
        ProjectionTransfer::Identity => value,
        ProjectionTransfer::Power { exponent: 0.5 } => value.sqrt(),
        ProjectionTransfer::Power { exponent } => value.powf(exponent),
        ProjectionTransfer::Window { lower, upper } => {
            ((value - lower) / (upper - lower)).clamp(0.0, 1.0)
        }
    }
}

fn validate_result(result: &AutomataSimulationResult) -> Result<usize, ProjectionError> {
    let size = result_size(result);
    let cell_count = size[0]
        .checked_mul(size[1])
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            ProjectionError::InvalidResult(
                "Simulation result dimensions must be non-zero and addressable.".to_owned(),
            )
        })?;
    match result {
        AutomataSimulationResult::LegacyEvidenceGrowthV1(result) => {
            if result.state.len() != cell_count {
                return Err(ProjectionError::InvalidResult(format!(
                    "Legacy result has {} values; expected {cell_count}.",
                    result.state.len()
                )));
            }
            if let Some((index, value)) = result
                .state
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite() || !(0.0..=1.0).contains(value))
            {
                return Err(ProjectionError::InvalidResult(format!(
                    "Legacy result value {value} at index {index} is outside [0, 1]."
                )));
            }
        }
        AutomataSimulationResult::DiscreteMapV1(result) => {
            if result.version != DISCRETE_RESULT_VERSION {
                return Err(ProjectionError::UnsupportedResultVersion {
                    model: "discrete-map",
                    found: result.version,
                });
            }
            let expected_storage = StateStorage::for_state_count(result.state_count)
                .map_err(|error| ProjectionError::InvalidResult(error.to_string()))?;
            if result.storage != expected_storage
                || result.final_state.storage() != expected_storage
            {
                return Err(ProjectionError::InvalidResult(
                    "Discrete result storage is not canonical for its state count.".to_owned(),
                ));
            }
            if result.final_state.len() != cell_count {
                return Err(ProjectionError::InvalidResult(format!(
                    "Discrete result has {} values; expected {cell_count}.",
                    result.final_state.len()
                )));
            }
            for index in 0..cell_count {
                let value = result.final_state.get(index).unwrap();
                if value >= result.state_count {
                    return Err(ProjectionError::InvalidResult(format!(
                        "Discrete result state {value} at index {index} is outside state_count {}.",
                        result.state_count
                    )));
                }
            }
        }
        AutomataSimulationResult::ContinuousLeniaV1(result) => {
            if result.version != CONTINUOUS_RESULT_VERSION {
                return Err(ProjectionError::UnsupportedResultVersion {
                    model: "continuous-lenia",
                    found: result.version,
                });
            }
            result
                .domain
                .validate()
                .map_err(|error| ProjectionError::InvalidResult(error.to_string()))?;
            if result.final_state.len() != cell_count {
                return Err(ProjectionError::InvalidResult(format!(
                    "Continuous result has {} values; expected {cell_count}.",
                    result.final_state.len()
                )));
            }
            if let Some((index, value)) = result
                .final_state
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !result.domain.contains(*value))
            {
                return Err(ProjectionError::InvalidResult(format!(
                    "Continuous result value {value} at index {index} is outside its domain."
                )));
            }
        }
    }
    Ok(cell_count)
}

fn result_size(result: &AutomataSimulationResult) -> [usize; 2] {
    match result {
        AutomataSimulationResult::LegacyEvidenceGrowthV1(result) => result.size,
        AutomataSimulationResult::DiscreteMapV1(result) => result.size,
        AutomataSimulationResult::ContinuousLeniaV1(result) => result.size,
    }
}

fn checked_bytes<T>(len: usize, context: &'static str) -> Result<u64, ProjectionError> {
    let bytes = len
        .checked_mul(size_of::<T>())
        .ok_or(ProjectionError::SizeOverflow { context })?;
    u64::try_from(bytes).map_err(|_| ProjectionError::SizeOverflow { context })
}

fn enforce_budget(
    context: &'static str,
    required_bytes: u64,
    budget_bytes: u64,
) -> Result<(), ProjectionError> {
    if required_bytes > budget_bytes {
        return Err(ProjectionError::BudgetExceeded {
            context,
            required_bytes,
            budget_bytes,
        });
    }
    Ok(())
}

fn try_vec_with_capacity<T>(
    len: usize,
    bytes: u64,
    context: &'static str,
) -> Result<Vec<T>, ProjectionError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| ProjectionError::AllocationFailed { context, bytes })?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        automata::{AutomataSimulationResult, CellularAutomataResult, CellularAutomataSummary},
        continuous_automata::{
            CONTINUOUS_RESULT_VERSION, ContinuousDiagnostics, ContinuousDomain,
            ContinuousSimulationResult,
        },
        discrete_automata::{
            CompactState, CycleDiagnostic, DISCRETE_RESULT_VERSION, DiscreteDiagnostics,
            DiscreteSimulationResult, StateStorage,
        },
    };

    fn legacy_result(values: Vec<f32>) -> AutomataSimulationResult {
        AutomataSimulationResult::LegacyEvidenceGrowthV1(CellularAutomataResult {
            size: [values.len(), 1],
            state: values,
            summary: CellularAutomataSummary {
                iterations_run: 1,
                converged: false,
                mean_activation: 0.0,
                active_fraction: 0.0,
            },
        })
    }

    fn discrete_result(values: Vec<u8>, state_count: u32) -> AutomataSimulationResult {
        AutomataSimulationResult::DiscreteMapV1(DiscreteSimulationResult {
            version: DISCRETE_RESULT_VERSION,
            size: [values.len(), 1],
            state_count,
            storage: StateStorage::U8,
            cycles_completed: 1,
            final_state: CompactState::U8(values),
            diagnostics: DiscreteDiagnostics {
                transition_steps_executed: 1,
                distinct_states_observed: None,
                fixed_point_generation: None,
                cycle: None::<CycleDiagnostic>,
                final_nonzero_cells: 0,
                last_transition_changed_cells: 0,
            },
        })
    }

    fn continuous_result(values: Vec<f32>) -> AutomataSimulationResult {
        AutomataSimulationResult::ContinuousLeniaV1(ContinuousSimulationResult {
            version: CONTINUOUS_RESULT_VERSION,
            size: [values.len(), 1],
            domain: ContinuousDomain {
                minimum: -1.0,
                maximum: 1.0,
            },
            cycles_completed: 1,
            final_state: values,
            diagnostics: ContinuousDiagnostics {
                transition_steps_executed: 1,
                exact_fixed_point_generation: None,
                last_transition_max_delta: Some(0.1),
                last_transition_within_tolerance: Some(false),
                final_minimum: -1.0,
                final_maximum: 1.0,
                final_mean: 0.0,
                final_mass: 0.0,
            },
        })
    }

    #[test]
    fn normalized_final_state_has_consistent_meaning_across_models() {
        let spec = NumericProjectionSpec::normalized_final_state();
        let limits = ProjectionLimits::default();

        let legacy =
            project_numeric_field(&legacy_result(vec![0.0, 0.5, 1.0]), &spec, limits).unwrap();
        let discrete =
            project_numeric_field(&discrete_result(vec![0, 1, 2], 3), &spec, limits).unwrap();
        let continuous =
            project_numeric_field(&continuous_result(vec![-1.0, 0.0, 1.0]), &spec, limits).unwrap();

        assert_eq!(legacy.values, [0.0, 0.5, 1.0]);
        assert_eq!(discrete.values, [0.0, 0.5, 1.0]);
        assert_eq!(continuous.values, [0.0, 0.5, 1.0]);
        assert_eq!(continuous.minimum, 0.0);
        assert_eq!(continuous.maximum, 1.0);
    }

    #[test]
    fn legacy_overlay_projection_preserves_the_existing_palette_and_sqrt_alpha() {
        let overlay = project_rgba_overlay(
            &legacy_result(vec![0.0, 0.25, 1.0]),
            &OverlayProjectionSpec {
                version: OUTPUT_PROJECTION_VERSION,
                field: NumericProjectionSpec {
                    version: OUTPUT_PROJECTION_VERSION,
                    source: ProjectionSource::NormalizedFinalState,
                    transfer: ProjectionTransfer::Power { exponent: 0.5 },
                },
                color: [30, 210, 190],
                maximum_alpha: 190,
            },
            ProjectionLimits::default(),
        )
        .unwrap();

        assert_eq!(
            overlay.rgba,
            [30, 210, 190, 0, 30, 210, 190, 95, 30, 210, 190, 190,]
        );
    }

    #[test]
    fn borrowed_continuous_overlay_matches_a_persisted_result_projection() {
        let values = vec![-1.0, 0.0, 1.0];
        let domain = ContinuousDomain {
            minimum: -1.0,
            maximum: 1.0,
        };
        let specification = OverlayProjectionSpec {
            version: OUTPUT_PROJECTION_VERSION,
            field: NumericProjectionSpec {
                version: OUTPUT_PROJECTION_VERSION,
                source: ProjectionSource::NormalizedFinalState,
                transfer: ProjectionTransfer::Power { exponent: 0.5 },
            },
            color: [48, 221, 255],
            maximum_alpha: 224,
        };

        let borrowed = project_continuous_state_overlay(
            &values,
            [values.len(), 1],
            domain,
            &specification,
            ProjectionLimits::default(),
        )
        .unwrap();
        let persisted = project_rgba_overlay(
            &continuous_result(values),
            &specification,
            ProjectionLimits::default(),
        )
        .unwrap();

        assert_eq!(borrowed, persisted);
    }

    #[test]
    fn borrowed_discrete_overlay_matches_a_persisted_result_projection() {
        let values = vec![0_u8, 1, 2];
        let state = CompactState::U8(values.clone());
        let specification = OverlayProjectionSpec {
            version: OUTPUT_PROJECTION_VERSION,
            field: NumericProjectionSpec {
                version: OUTPUT_PROJECTION_VERSION,
                source: ProjectionSource::NormalizedFinalState,
                transfer: ProjectionTransfer::Power { exponent: 0.5 },
            },
            color: [48, 221, 255],
            maximum_alpha: 224,
        };

        let borrowed = project_discrete_state_overlay(
            &state,
            [values.len(), 1],
            3,
            &specification,
            ProjectionLimits::default(),
        )
        .unwrap();
        let persisted = project_rgba_overlay(
            &discrete_result(values, 3),
            &specification,
            ProjectionLimits::default(),
        )
        .unwrap();

        assert_eq!(borrowed, persisted);
    }

    #[test]
    fn borrowed_discrete_membership_overlay_matches_a_persisted_result_projection() {
        let values = vec![0_u8, 2, 1, 2];
        let state = CompactState::U8(values.clone());
        let specification = OverlayProjectionSpec {
            version: OUTPUT_PROJECTION_VERSION,
            field: NumericProjectionSpec {
                version: OUTPUT_PROJECTION_VERSION,
                source: ProjectionSource::DiscreteStateMembership { states: vec![2] },
                transfer: ProjectionTransfer::Identity,
            },
            color: [10, 20, 30],
            maximum_alpha: 255,
        };

        let borrowed = project_discrete_state_overlay(
            &state,
            [values.len(), 1],
            3,
            &specification,
            ProjectionLimits::default(),
        )
        .unwrap();
        let persisted = project_rgba_overlay(
            &discrete_result(values, 3),
            &specification,
            ProjectionLimits::default(),
        )
        .unwrap();

        assert_eq!(borrowed, persisted);
    }

    #[test]
    fn discrete_state_overlay_rejects_size_and_range_mismatches() {
        let state = CompactState::U8(vec![0, 1, 2]);
        let specification = OverlayProjectionSpec {
            version: OUTPUT_PROJECTION_VERSION,
            field: NumericProjectionSpec::normalized_final_state(),
            color: [0, 0, 0],
            maximum_alpha: 255,
        };
        assert!(matches!(
            project_discrete_state_overlay(
                &state,
                [2, 1],
                3,
                &specification,
                ProjectionLimits::default(),
            ),
            Err(ProjectionError::InvalidResult(_))
        ));
        let out_of_range = CompactState::U8(vec![0, 1, 5]);
        assert!(matches!(
            project_discrete_state_overlay(
                &out_of_range,
                [3, 1],
                3,
                &specification,
                ProjectionLimits::default(),
            ),
            Err(ProjectionError::InvalidResult(_))
        ));
    }

    #[test]
    fn discrete_membership_projects_to_a_mask_without_state_order_aliasing() {
        let result = discrete_result(vec![0, 2, 1, 2], 3);
        let field = NumericProjectionSpec {
            version: OUTPUT_PROJECTION_VERSION,
            source: ProjectionSource::DiscreteStateMembership { states: vec![2] },
            transfer: ProjectionTransfer::Identity,
        };
        let mask = project_mask(
            &result,
            &MaskProjectionSpec {
                version: OUTPUT_PROJECTION_VERSION,
                field,
                minimum: 1.0,
                maximum: 1.0,
            },
            ProjectionLimits::default(),
        )
        .unwrap();

        assert_eq!(mask.values, [0, 1, 0, 1]);
        assert_eq!(mask.included_cells, 2);
    }

    #[test]
    fn projection_specs_are_strict_and_invalid_model_state_and_budgets_fail() {
        let spec = NumericProjectionSpec {
            version: OUTPUT_PROJECTION_VERSION,
            source: ProjectionSource::DiscreteStateMembership { states: vec![1] },
            transfer: ProjectionTransfer::Identity,
        };
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(
            serde_json::from_str::<NumericProjectionSpec>(&json).unwrap(),
            spec
        );
        let mut value = serde_json::to_value(&spec).unwrap();
        value["surprise"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<NumericProjectionSpec>(value).is_err());

        assert!(matches!(
            project_numeric_field(
                &legacy_result(vec![0.0]),
                &spec,
                ProjectionLimits::default(),
            ),
            Err(ProjectionError::ProjectionRequiresDiscreteResult)
        ));
        let duplicate = NumericProjectionSpec {
            source: ProjectionSource::DiscreteStateMembership { states: vec![1, 1] },
            ..spec.clone()
        };
        assert!(matches!(
            project_numeric_field(
                &discrete_result(vec![0], 2),
                &duplicate,
                ProjectionLimits::default(),
            ),
            Err(ProjectionError::DuplicateDiscreteState { state: 1 })
        ));
        assert!(matches!(
            project_numeric_field(
                &discrete_result(vec![0, 1], 2),
                &NumericProjectionSpec::normalized_final_state(),
                ProjectionLimits {
                    max_output_bytes: 7,
                },
            ),
            Err(ProjectionError::BudgetExceeded {
                context: "numeric projection",
                required_bytes: 8,
                ..
            })
        ));
    }
}
