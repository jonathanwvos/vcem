use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::quantization::IntensityCalibration;

pub const CONTINUOUS_INITIALIZER_VERSION: u32 = 1;
pub const CONTINUOUS_LENIA_PROGRAM_VERSION: u32 = 1;
pub const CONTINUOUS_RUN_CONFIG_VERSION: u32 = 1;
pub const CONTINUOUS_RESULT_VERSION: u32 = 1;

const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousDomain {
    pub minimum: f32,
    pub maximum: f32,
}

impl ContinuousDomain {
    pub const UNIT: Self = Self {
        minimum: 0.0,
        maximum: 1.0,
    };

    pub fn validate(self) -> Result<(), ContinuousAutomataError> {
        if !self.minimum.is_finite() || !self.maximum.is_finite() || self.minimum >= self.maximum {
            return Err(ContinuousAutomataError::InvalidDomain {
                minimum: self.minimum,
                maximum: self.maximum,
            });
        }
        Ok(())
    }

    pub fn contains(self, value: f32) -> bool {
        value.is_finite() && (self.minimum..=self.maximum).contains(&value)
    }

    fn clamp_f64(self, value: f64) -> f32 {
        value.clamp(self.minimum as f64, self.maximum as f64) as f32
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousStateInitializer {
    pub version: u32,
    pub calibration: IntensityCalibration,
    pub domain: ContinuousDomain,
}

impl ContinuousStateInitializer {
    pub fn compile(&self) -> Result<CompiledContinuousStateInitializer, ContinuousAutomataError> {
        if self.version != CONTINUOUS_INITIALIZER_VERSION {
            return Err(ContinuousAutomataError::UnsupportedInitializerVersion {
                found: self.version,
            });
        }
        self.calibration
            .validate()
            .map_err(ContinuousAutomataError::InvalidCalibration)?;
        if self.calibration.level_count() < 2 {
            return Err(ContinuousAutomataError::CalibrationHasTooFewLevels {
                level_count: self.calibration.level_count(),
            });
        }
        self.domain.validate()?;
        Ok(CompiledContinuousStateInitializer {
            calibration: self.calibration,
            domain: self.domain,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompiledContinuousStateInitializer {
    calibration: IntensityCalibration,
    domain: ContinuousDomain,
}

impl CompiledContinuousStateInitializer {
    pub fn calibration(self) -> IntensityCalibration {
        self.calibration
    }

    pub fn domain(self) -> ContinuousDomain {
        self.domain
    }

    pub fn initialize(
        self,
        values: &[u32],
        max_state_bytes: u64,
    ) -> Result<Vec<f32>, ContinuousAutomataError> {
        let required_bytes = checked_f32_bytes(values.len(), "continuous initial state")?;
        enforce_budget("continuous initial state", required_bytes, max_state_bytes)?;
        let mut state = Vec::new();
        state.try_reserve_exact(values.len()).map_err(|_| {
            ContinuousAutomataError::AllocationFailed {
                context: "continuous initial state",
                bytes: required_bytes,
            }
        })?;

        let calibration_minimum = self.calibration.minimum as u64;
        let calibration_maximum = self.calibration.maximum_exclusive;
        let denominator = (self.calibration.level_count() - 1) as f64;
        let domain_minimum = self.domain.minimum as f64;
        let domain_width = self.domain.maximum as f64 - self.domain.minimum as f64;
        for (index, value) in values.iter().copied().enumerate() {
            let value_u64 = value as u64;
            if value_u64 < calibration_minimum || value_u64 >= calibration_maximum {
                return Err(ContinuousAutomataError::IntensityOutsideCalibration {
                    index,
                    value,
                    calibration: self.calibration,
                });
            }
            let unit = (value_u64 - calibration_minimum) as f64 / denominator;
            state.push((domain_minimum + unit * domain_width) as f32);
        }
        Ok(state)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseContinuousKernel {
    pub width: u32,
    pub height: u32,
    pub anchor_x: u32,
    pub anchor_y: u32,
    pub weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelNormalization {
    None,
    SumToOne,
    L1ToOne,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ContinuousBoundaryPolicy {
    Clamp,
    Reflect,
    Wrap,
    Constant { value: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ContinuousGrowthFunction {
    /// `offset + amplitude * exp(-0.5 * ((u - center) / width)^2)`.
    Gaussian {
        center: f32,
        width: f32,
        amplitude: f32,
        offset: f32,
    },
    Band {
        lower: f32,
        upper: f32,
        inside: f32,
        outside: f32,
    },
    Linear {
        slope: f32,
        intercept: f32,
    },
}

impl ContinuousGrowthFunction {
    pub const fn canonical_gaussian(center: f32, width: f32) -> Self {
        Self::Gaussian {
            center,
            width,
            amplitude: 2.0,
            offset: -1.0,
        }
    }

    fn validate(self) -> Result<(), ContinuousAutomataError> {
        match self {
            Self::Gaussian {
                center,
                width,
                amplitude,
                offset,
            } => {
                if !center.is_finite()
                    || !width.is_finite()
                    || width <= 0.0
                    || !amplitude.is_finite()
                    || !offset.is_finite()
                {
                    return Err(ContinuousAutomataError::InvalidGrowthFunction(
                        "Gaussian growth requires finite parameters and width greater than zero."
                            .to_owned(),
                    ));
                }
            }
            Self::Band {
                lower,
                upper,
                inside,
                outside,
            } => {
                if !lower.is_finite()
                    || !upper.is_finite()
                    || lower > upper
                    || !inside.is_finite()
                    || !outside.is_finite()
                {
                    return Err(ContinuousAutomataError::InvalidGrowthFunction(
                        "Band growth requires finite parameters and lower <= upper.".to_owned(),
                    ));
                }
            }
            Self::Linear { slope, intercept } => {
                if !slope.is_finite() || !intercept.is_finite() {
                    return Err(ContinuousAutomataError::InvalidGrowthFunction(
                        "Linear growth requires finite parameters.".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn evaluate(self, sensed: f64) -> f64 {
        match self {
            Self::Gaussian {
                center,
                width,
                amplitude,
                offset,
            } => {
                let normalized = (sensed - center as f64) / width as f64;
                offset as f64 + amplitude as f64 * (-0.5 * normalized * normalized).exp()
            }
            Self::Band {
                lower,
                upper,
                inside,
                outside,
            } => {
                if (lower as f64..=upper as f64).contains(&sensed) {
                    inside as f64
                } else {
                    outside as f64
                }
            }
            Self::Linear { slope, intercept } => slope as f64 * sensed + intercept as f64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousLeniaProgram {
    pub version: u32,
    pub domain: ContinuousDomain,
    pub kernel: DenseContinuousKernel,
    pub kernel_normalization: KernelNormalization,
    pub boundary: ContinuousBoundaryPolicy,
    pub growth: ContinuousGrowthFunction,
    pub time_step: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousCompileLimits {
    /// Maximum logical bytes occupied by the dense f32 kernel. The owned
    /// weight buffer is normalized in place and moved into the compiled form.
    pub max_kernel_bytes: u64,
}

impl Default for ContinuousCompileLimits {
    fn default() -> Self {
        Self {
            max_kernel_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousRunConfig {
    pub version: u32,
    pub cycles: u32,
    /// Bounds the owned final-state and scratch buffers, excluding the
    /// caller-owned initial state and allocator metadata.
    pub max_working_memory_bytes: u64,
    /// Report whether the final evaluated transition falls within this
    /// tolerance. Approximate convergence never changes the requested run.
    pub convergence_tolerance: f32,
}

impl ContinuousRunConfig {
    pub const fn new(cycles: u32) -> Self {
        Self {
            version: CONTINUOUS_RUN_CONFIG_VERSION,
            cycles,
            max_working_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            convergence_tolerance: 1.0e-6,
        }
    }
}

impl Default for ContinuousRunConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ContinuousDiagnostics {
    /// Actual transitions evaluated. An exact fixed point may be safely
    /// fast-forwarded to the requested cycle count.
    pub transition_steps_executed: u32,
    pub exact_fixed_point_generation: Option<u32>,
    pub last_transition_max_delta: Option<f32>,
    pub last_transition_within_tolerance: Option<bool>,
    pub final_minimum: f32,
    pub final_maximum: f32,
    pub final_mean: f32,
    pub final_mass: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContinuousSimulationResult {
    pub version: u32,
    pub size: [usize; 2],
    pub domain: ContinuousDomain,
    pub cycles_completed: u32,
    pub final_state: Vec<f32>,
    pub diagnostics: ContinuousDiagnostics,
}

#[derive(Clone, Debug)]
pub struct CompiledContinuousLenia {
    domain: ContinuousDomain,
    kernel_width: usize,
    kernel_height: usize,
    anchor_x: usize,
    anchor_y: usize,
    kernel_weights: Arc<Vec<f32>>,
    kernel_normalization: KernelNormalization,
    boundary: ContinuousBoundaryPolicy,
    growth: ContinuousGrowthFunction,
    time_step: f32,
}

impl CompiledContinuousLenia {
    pub fn domain(&self) -> ContinuousDomain {
        self.domain
    }

    pub fn kernel_weights(&self) -> &[f32] {
        self.kernel_weights.as_slice()
    }

    pub fn kernel_normalization(&self) -> KernelNormalization {
        self.kernel_normalization
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ContinuousAutomataError {
    UnsupportedInitializerVersion {
        found: u32,
    },
    UnsupportedProgramVersion {
        found: u32,
    },
    UnsupportedRunConfigVersion {
        found: u32,
    },
    InvalidCalibration(String),
    CalibrationHasTooFewLevels {
        level_count: u64,
    },
    IntensityOutsideCalibration {
        index: usize,
        value: u32,
        calibration: IntensityCalibration,
    },
    InvalidDomain {
        minimum: f32,
        maximum: f32,
    },
    InvalidKernelDimensions {
        width: u32,
        height: u32,
    },
    KernelDimensionsOverflow,
    KernelLengthMismatch {
        expected: usize,
        actual: usize,
    },
    KernelAnchorOutOfRange {
        anchor_x: u32,
        anchor_y: u32,
        width: u32,
        height: u32,
    },
    NonFiniteKernelWeight {
        index: usize,
    },
    ZeroKernelNormalization {
        normalization: KernelNormalization,
    },
    InvalidNormalizedKernelWeight {
        index: usize,
    },
    InvalidBoundaryValue {
        value: f32,
        domain: ContinuousDomain,
    },
    InvalidGrowthFunction(String),
    InvalidTimeStep {
        value: f32,
    },
    InvalidConvergenceTolerance {
        value: f32,
    },
    Cancelled,
    InvalidDimensions {
        width: usize,
        height: usize,
    },
    StateLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidInitialStateValue {
        index: usize,
        value: f32,
        domain: ContinuousDomain,
    },
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

impl fmt::Display for ContinuousAutomataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedInitializerVersion { found } => {
                write!(
                    formatter,
                    "Unsupported continuous initializer version {found}."
                )
            }
            Self::UnsupportedProgramVersion { found } => {
                write!(
                    formatter,
                    "Unsupported continuous Lenia program version {found}."
                )
            }
            Self::UnsupportedRunConfigVersion { found } => {
                write!(
                    formatter,
                    "Unsupported continuous run configuration version {found}."
                )
            }
            Self::InvalidCalibration(error) | Self::InvalidGrowthFunction(error) => {
                formatter.write_str(error)
            }
            Self::CalibrationHasTooFewLevels { level_count } => write!(
                formatter,
                "Continuous initialization requires at least two calibrated levels; found {level_count}."
            ),
            Self::IntensityOutsideCalibration {
                index,
                value,
                calibration,
            } => write!(
                formatter,
                "Intensity {value} at index {index} is outside calibration [{}, {}).",
                calibration.minimum, calibration.maximum_exclusive
            ),
            Self::InvalidDomain { minimum, maximum } => write!(
                formatter,
                "Continuous domain bounds must be finite and satisfy minimum < maximum; found [{minimum}, {maximum}]."
            ),
            Self::InvalidKernelDimensions { width, height } => write!(
                formatter,
                "Continuous kernels require non-zero dimensions; found {width} x {height}."
            ),
            Self::KernelDimensionsOverflow => {
                formatter.write_str("Continuous kernel dimensions overflow addressable memory.")
            }
            Self::KernelLengthMismatch { expected, actual } => write!(
                formatter,
                "Continuous kernel has {actual} weights; expected {expected}."
            ),
            Self::KernelAnchorOutOfRange {
                anchor_x,
                anchor_y,
                width,
                height,
            } => write!(
                formatter,
                "Kernel anchor ({anchor_x}, {anchor_y}) is outside {width} x {height}."
            ),
            Self::NonFiniteKernelWeight { index } => {
                write!(formatter, "Continuous kernel weight {index} is not finite.")
            }
            Self::ZeroKernelNormalization { normalization } => write!(
                formatter,
                "Kernel normalization {normalization:?} requires a non-zero divisor."
            ),
            Self::InvalidNormalizedKernelWeight { index } => write!(
                formatter,
                "Normalized continuous kernel weight {index} is not finite."
            ),
            Self::InvalidBoundaryValue { value, domain } => write!(
                formatter,
                "Constant boundary value {value} is outside [{}, {}].",
                domain.minimum, domain.maximum
            ),
            Self::InvalidTimeStep { value } => write!(
                formatter,
                "Continuous Lenia time_step must be finite and greater than zero; found {value}."
            ),
            Self::InvalidConvergenceTolerance { value } => write!(
                formatter,
                "Continuous convergence tolerance must be finite and non-negative; found {value}."
            ),
            Self::Cancelled => formatter.write_str("Continuous automata run was cancelled."),
            Self::InvalidDimensions { width, height } => write!(
                formatter,
                "Continuous automata require non-zero addressable dimensions; found {width} x {height}."
            ),
            Self::StateLengthMismatch { expected, actual } => write!(
                formatter,
                "Continuous state has {actual} values; expected {expected}."
            ),
            Self::InvalidInitialStateValue {
                index,
                value,
                domain,
            } => write!(
                formatter,
                "Continuous state value {value} at index {index} is outside [{}, {}].",
                domain.minimum, domain.maximum
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

impl std::error::Error for ContinuousAutomataError {}

pub fn compile_continuous_lenia(
    mut program: ContinuousLeniaProgram,
    limits: ContinuousCompileLimits,
) -> Result<CompiledContinuousLenia, ContinuousAutomataError> {
    validate_continuous_lenia_program(&program, limits)?;
    let kernel_width = program.kernel.width as usize;
    let kernel_height = program.kernel.height as usize;
    normalize_kernel(&mut program.kernel.weights, program.kernel_normalization)?;

    Ok(CompiledContinuousLenia {
        domain: program.domain,
        kernel_width,
        kernel_height,
        anchor_x: program.kernel.anchor_x as usize,
        anchor_y: program.kernel.anchor_y as usize,
        kernel_weights: Arc::new(program.kernel.weights),
        kernel_normalization: program.kernel_normalization,
        boundary: program.boundary,
        growth: program.growth,
        time_step: program.time_step,
    })
}

/// Validates a serialized continuous program without cloning or consuming its
/// dense kernel. Persistence code uses this before binding a result record.
pub fn validate_continuous_lenia_program(
    program: &ContinuousLeniaProgram,
    limits: ContinuousCompileLimits,
) -> Result<(), ContinuousAutomataError> {
    if program.version != CONTINUOUS_LENIA_PROGRAM_VERSION {
        return Err(ContinuousAutomataError::UnsupportedProgramVersion {
            found: program.version,
        });
    }
    program.domain.validate()?;
    if !program.time_step.is_finite() || program.time_step <= 0.0 {
        return Err(ContinuousAutomataError::InvalidTimeStep {
            value: program.time_step,
        });
    }
    program.growth.validate()?;
    if let ContinuousBoundaryPolicy::Constant { value } = program.boundary
        && !program.domain.contains(value)
    {
        return Err(ContinuousAutomataError::InvalidBoundaryValue {
            value,
            domain: program.domain,
        });
    }

    let kernel_width = usize::try_from(program.kernel.width)
        .map_err(|_| ContinuousAutomataError::KernelDimensionsOverflow)?;
    let kernel_height = usize::try_from(program.kernel.height)
        .map_err(|_| ContinuousAutomataError::KernelDimensionsOverflow)?;
    if kernel_width == 0 || kernel_height == 0 {
        return Err(ContinuousAutomataError::InvalidKernelDimensions {
            width: program.kernel.width,
            height: program.kernel.height,
        });
    }
    let kernel_len = kernel_width
        .checked_mul(kernel_height)
        .ok_or(ContinuousAutomataError::KernelDimensionsOverflow)?;
    if program.kernel.weights.len() != kernel_len {
        return Err(ContinuousAutomataError::KernelLengthMismatch {
            expected: kernel_len,
            actual: program.kernel.weights.len(),
        });
    }
    if program.kernel.anchor_x >= program.kernel.width
        || program.kernel.anchor_y >= program.kernel.height
    {
        return Err(ContinuousAutomataError::KernelAnchorOutOfRange {
            anchor_x: program.kernel.anchor_x,
            anchor_y: program.kernel.anchor_y,
            width: program.kernel.width,
            height: program.kernel.height,
        });
    }
    let kernel_bytes = checked_f32_bytes(kernel_len, "continuous kernel")?;
    enforce_budget("continuous kernel", kernel_bytes, limits.max_kernel_bytes)?;
    if let Some((index, _)) = program
        .kernel
        .weights
        .iter()
        .enumerate()
        .find(|(_, weight)| !weight.is_finite())
    {
        return Err(ContinuousAutomataError::NonFiniteKernelWeight { index });
    }

    validate_kernel_normalization(&program.kernel.weights, program.kernel_normalization)?;
    Ok(())
}

pub fn run_continuous_lenia(
    program: &CompiledContinuousLenia,
    initial_state: &[f32],
    width: usize,
    height: usize,
    config: &ContinuousRunConfig,
) -> Result<ContinuousSimulationResult, ContinuousAutomataError> {
    run_continuous_lenia_inner(program, initial_state, width, height, config, None)
}

pub fn run_continuous_lenia_with_cancellation(
    program: &CompiledContinuousLenia,
    initial_state: &[f32],
    width: usize,
    height: usize,
    config: &ContinuousRunConfig,
    cancellation: &AtomicBool,
) -> Result<ContinuousSimulationResult, ContinuousAutomataError> {
    run_continuous_lenia_inner(
        program,
        initial_state,
        width,
        height,
        config,
        Some(cancellation),
    )
}

fn run_continuous_lenia_inner(
    program: &CompiledContinuousLenia,
    initial_state: &[f32],
    width: usize,
    height: usize,
    config: &ContinuousRunConfig,
    cancellation: Option<&AtomicBool>,
) -> Result<ContinuousSimulationResult, ContinuousAutomataError> {
    if config.version != CONTINUOUS_RUN_CONFIG_VERSION {
        return Err(ContinuousAutomataError::UnsupportedRunConfigVersion {
            found: config.version,
        });
    }
    if !config.convergence_tolerance.is_finite() || config.convergence_tolerance < 0.0 {
        return Err(ContinuousAutomataError::InvalidConvergenceTolerance {
            value: config.convergence_tolerance,
        });
    }
    let cell_count = rectangular_len(width, height)?;
    if initial_state.len() != cell_count {
        return Err(ContinuousAutomataError::StateLengthMismatch {
            expected: cell_count,
            actual: initial_state.len(),
        });
    }
    if let Some((index, value)) = initial_state
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !program.domain.contains(*value))
    {
        return Err(ContinuousAutomataError::InvalidInitialStateValue {
            index,
            value,
            domain: program.domain,
        });
    }

    let state_bytes = checked_f32_bytes(cell_count, "continuous state")?;
    let buffer_count = if config.cycles == 0 { 1 } else { 2 };
    let working_bytes =
        state_bytes
            .checked_mul(buffer_count)
            .ok_or(ContinuousAutomataError::SizeOverflow {
                context: "continuous working memory",
            })?;
    enforce_budget(
        "continuous working memory",
        working_bytes,
        config.max_working_memory_bytes,
    )?;

    let mut state = try_clone_f32(initial_state, state_bytes, "continuous final state")?;
    let mut next = if config.cycles == 0 {
        Vec::new()
    } else {
        try_zeroed_f32(cell_count, state_bytes, "continuous scratch state")?
    };
    let mut transition_steps_executed = 0_u32;
    let mut exact_fixed_point_generation = None;
    let mut last_transition_max_delta = None;

    for generation in 0..config.cycles {
        if cancellation.is_some_and(|token| token.load(Ordering::Relaxed)) {
            return Err(ContinuousAutomataError::Cancelled);
        }
        let maximum_delta = step_continuous(program, &state, &mut next, width, height);
        transition_steps_executed += 1;
        last_transition_max_delta = Some(maximum_delta);
        std::mem::swap(&mut state, &mut next);
        if maximum_delta == 0.0 {
            exact_fixed_point_generation = Some(generation);
            break;
        }
    }

    let (final_minimum, final_maximum, final_mean, final_mass) = summarize(&state);
    Ok(ContinuousSimulationResult {
        version: CONTINUOUS_RESULT_VERSION,
        size: [width, height],
        domain: program.domain,
        cycles_completed: config.cycles,
        final_state: state,
        diagnostics: ContinuousDiagnostics {
            transition_steps_executed,
            exact_fixed_point_generation,
            last_transition_max_delta,
            last_transition_within_tolerance: last_transition_max_delta
                .map(|delta| delta <= config.convergence_tolerance),
            final_minimum,
            final_maximum,
            final_mean,
            final_mass,
        },
    })
}

fn normalize_kernel(
    weights: &mut [f32],
    normalization: KernelNormalization,
) -> Result<(), ContinuousAutomataError> {
    let divisor: f64 = match normalization {
        KernelNormalization::None => return Ok(()),
        KernelNormalization::SumToOne => weights.iter().map(|weight| *weight as f64).sum(),
        KernelNormalization::L1ToOne => weights.iter().map(|weight| weight.abs() as f64).sum(),
    };
    if divisor == 0.0 {
        return Err(ContinuousAutomataError::ZeroKernelNormalization { normalization });
    }
    for (index, weight) in weights.iter_mut().enumerate() {
        *weight = (*weight as f64 / divisor) as f32;
        if !weight.is_finite() {
            return Err(ContinuousAutomataError::InvalidNormalizedKernelWeight { index });
        }
    }
    Ok(())
}

fn validate_kernel_normalization(
    weights: &[f32],
    normalization: KernelNormalization,
) -> Result<(), ContinuousAutomataError> {
    let divisor: f64 = match normalization {
        KernelNormalization::None => return Ok(()),
        KernelNormalization::SumToOne => weights.iter().map(|weight| *weight as f64).sum(),
        KernelNormalization::L1ToOne => weights.iter().map(|weight| weight.abs() as f64).sum(),
    };
    if divisor == 0.0 {
        return Err(ContinuousAutomataError::ZeroKernelNormalization { normalization });
    }
    for (index, weight) in weights.iter().enumerate() {
        if !(*weight as f64 / divisor).is_finite() {
            return Err(ContinuousAutomataError::InvalidNormalizedKernelWeight { index });
        }
    }
    Ok(())
}

fn step_continuous(
    program: &CompiledContinuousLenia,
    current: &[f32],
    next: &mut [f32],
    width: usize,
    height: usize,
) -> f32 {
    next.par_chunks_mut(width)
        .enumerate()
        .map(|(y, row)| {
            let mut row_maximum_delta = 0.0_f32;
            for (x, output) in row.iter_mut().enumerate() {
                let mut sensed = 0.0_f64;
                for kernel_y in 0..program.kernel_height {
                    let sample_y = y as i128 + kernel_y as i128 - program.anchor_y as i128;
                    let kernel_row = kernel_y * program.kernel_width;
                    for kernel_x in 0..program.kernel_width {
                        let weight = program.kernel_weights[kernel_row + kernel_x];
                        if weight == 0.0 {
                            continue;
                        }
                        let sample_x = x as i128 + kernel_x as i128 - program.anchor_x as i128;
                        sensed += sample_continuous(
                            current,
                            width,
                            height,
                            sample_x,
                            sample_y,
                            program.boundary,
                        ) as f64
                            * weight as f64;
                    }
                }
                let current_value = current[y * width + x];
                let growth = program.growth.evaluate(sensed);
                let next_value = program
                    .domain
                    .clamp_f64(current_value as f64 + program.time_step as f64 * growth);
                row_maximum_delta = row_maximum_delta.max((next_value - current_value).abs());
                *output = next_value;
            }
            row_maximum_delta
        })
        .reduce(|| 0.0, f32::max)
}

fn sample_continuous(
    state: &[f32],
    width: usize,
    height: usize,
    x: i128,
    y: i128,
    boundary: ContinuousBoundaryPolicy,
) -> f32 {
    let inside = x >= 0 && y >= 0 && x < width as i128 && y < height as i128;
    if inside {
        return state[y as usize * width + x as usize];
    }
    let (x, y) = match boundary {
        ContinuousBoundaryPolicy::Clamp => (
            x.clamp(0, width as i128 - 1) as usize,
            y.clamp(0, height as i128 - 1) as usize,
        ),
        ContinuousBoundaryPolicy::Reflect => {
            (reflect_coordinate(x, width), reflect_coordinate(y, height))
        }
        ContinuousBoundaryPolicy::Wrap => (
            x.rem_euclid(width as i128) as usize,
            y.rem_euclid(height as i128) as usize,
        ),
        ContinuousBoundaryPolicy::Constant { value } => return value,
    };
    state[y * width + x]
}

fn reflect_coordinate(coordinate: i128, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as i128 - 1);
    let reflected = coordinate.rem_euclid(period);
    if reflected < length as i128 {
        reflected as usize
    } else {
        (period - reflected) as usize
    }
}

fn rectangular_len(width: usize, height: usize) -> Result<usize, ContinuousAutomataError> {
    if width == 0 || height == 0 {
        return Err(ContinuousAutomataError::InvalidDimensions { width, height });
    }
    width
        .checked_mul(height)
        .ok_or(ContinuousAutomataError::InvalidDimensions { width, height })
}

fn checked_f32_bytes(len: usize, context: &'static str) -> Result<u64, ContinuousAutomataError> {
    let bytes = len
        .checked_mul(size_of::<f32>())
        .ok_or(ContinuousAutomataError::SizeOverflow { context })?;
    u64::try_from(bytes).map_err(|_| ContinuousAutomataError::SizeOverflow { context })
}

fn enforce_budget(
    context: &'static str,
    required_bytes: u64,
    budget_bytes: u64,
) -> Result<(), ContinuousAutomataError> {
    if required_bytes > budget_bytes {
        return Err(ContinuousAutomataError::BudgetExceeded {
            context,
            required_bytes,
            budget_bytes,
        });
    }
    Ok(())
}

fn try_clone_f32(
    source: &[f32],
    bytes: u64,
    context: &'static str,
) -> Result<Vec<f32>, ContinuousAutomataError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(|_| ContinuousAutomataError::AllocationFailed { context, bytes })?;
    values.extend_from_slice(source);
    Ok(values)
}

fn try_zeroed_f32(
    len: usize,
    bytes: u64,
    context: &'static str,
) -> Result<Vec<f32>, ContinuousAutomataError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| ContinuousAutomataError::AllocationFailed { context, bytes })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn summarize(state: &[f32]) -> (f32, f32, f32, f64) {
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut mass = 0.0_f64;
    for value in state {
        minimum = minimum.min(*value);
        maximum = maximum.max(*value);
        mass += *value as f64;
    }
    (minimum, maximum, (mass / state.len() as f64) as f32, mass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    fn linear_program(
        kernel: DenseContinuousKernel,
        boundary: ContinuousBoundaryPolicy,
        domain: ContinuousDomain,
        slope: f32,
        intercept: f32,
    ) -> ContinuousLeniaProgram {
        ContinuousLeniaProgram {
            version: CONTINUOUS_LENIA_PROGRAM_VERSION,
            domain,
            kernel,
            kernel_normalization: KernelNormalization::None,
            boundary,
            growth: ContinuousGrowthFunction::Linear { slope, intercept },
            time_step: 1.0,
        }
    }

    fn one_cell_program(growth: ContinuousGrowthFunction) -> ContinuousLeniaProgram {
        ContinuousLeniaProgram {
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
            growth,
            time_step: 0.1,
        }
    }

    fn assert_approximately(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn calibrated_initializer_maps_encoded_endpoints_into_the_domain() {
        let calibration = IntensityCalibration::new(10, 14).unwrap();
        let initializer = ContinuousStateInitializer {
            version: CONTINUOUS_INITIALIZER_VERSION,
            calibration,
            domain: ContinuousDomain {
                minimum: -1.0,
                maximum: 1.0,
            },
        }
        .compile()
        .unwrap();

        let state = initializer.initialize(&[10, 11, 13], 12).unwrap();
        assert_eq!(state[0], -1.0);
        assert_approximately(state[1], -1.0 / 3.0);
        assert_eq!(state[2], 1.0);
        assert!(matches!(
            initializer.initialize(&[14], 4),
            Err(ContinuousAutomataError::IntensityOutsideCalibration {
                index: 0,
                value: 14,
                ..
            })
        ));
        assert!(matches!(
            initializer.initialize(&[10, 11, 12], 11),
            Err(ContinuousAutomataError::BudgetExceeded { .. })
        ));
    }

    #[test]
    fn normalization_is_explicit_and_moves_the_dense_buffer_without_copying() {
        let mut program = one_cell_program(ContinuousGrowthFunction::Linear {
            slope: 0.0,
            intercept: 0.0,
        });
        program.kernel.width = 2;
        program.kernel.weights = vec![2.0, 6.0];
        let original_pointer = program.kernel.weights.as_ptr();

        let compiled =
            compile_continuous_lenia(program, ContinuousCompileLimits::default()).unwrap();

        assert_eq!(compiled.kernel_weights().as_ptr(), original_pointer);
        assert_eq!(
            compiled.kernel_normalization(),
            KernelNormalization::SumToOne
        );
        assert_eq!(compiled.kernel_weights(), [0.25, 0.75]);

        let zero_sum = ContinuousLeniaProgram {
            kernel: DenseContinuousKernel {
                width: 2,
                height: 1,
                anchor_x: 0,
                anchor_y: 0,
                weights: vec![-1.0, 1.0],
            },
            ..one_cell_program(ContinuousGrowthFunction::Linear {
                slope: 0.0,
                intercept: 0.0,
            })
        };
        assert!(matches!(
            compile_continuous_lenia(zero_sum, ContinuousCompileLimits::default()),
            Err(ContinuousAutomataError::ZeroKernelNormalization {
                normalization: KernelNormalization::SumToOne,
            })
        ));
    }

    #[test]
    fn continuous_lenia_is_autonomous_and_has_no_evidence_gate_or_decay() {
        let mut program = one_cell_program(ContinuousGrowthFunction::Linear {
            slope: 0.0,
            intercept: 1.0,
        });
        program.kernel.weights[0] = 0.0;
        program.kernel_normalization = KernelNormalization::None;
        program.time_step = 0.2;
        let compiled =
            compile_continuous_lenia(program, ContinuousCompileLimits::default()).unwrap();

        let result =
            run_continuous_lenia(&compiled, &[0.0], 1, 1, &ContinuousRunConfig::new(3)).unwrap();

        assert_approximately(result.final_state[0], 0.6);
        assert_eq!(result.diagnostics.transition_steps_executed, 3);
        assert_eq!(result.diagnostics.exact_fixed_point_generation, None);
    }

    #[test]
    fn state_updates_use_dt_growth_and_clamp_to_the_declared_domain() {
        let compiled = compile_continuous_lenia(
            one_cell_program(ContinuousGrowthFunction::Linear {
                slope: 0.0,
                intercept: 0.5,
            }),
            ContinuousCompileLimits::default(),
        )
        .unwrap();
        let result =
            run_continuous_lenia(&compiled, &[0.25], 1, 1, &ContinuousRunConfig::new(20)).unwrap();

        assert_eq!(result.final_state, vec![1.0]);
        assert!(result.diagnostics.transition_steps_executed < 20);
        assert_eq!(result.diagnostics.final_maximum, 1.0);
        assert_eq!(result.diagnostics.final_mass, 1.0);
    }

    #[test]
    fn canonical_gaussian_has_the_lenia_peak_and_background_response() {
        let growth = ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1);
        assert_approximately(growth.evaluate(0.5) as f32, 1.0);
        assert_approximately(growth.evaluate(10.0) as f32, -1.0);

        let compiled =
            compile_continuous_lenia(one_cell_program(growth), ContinuousCompileLimits::default())
                .unwrap();
        let result =
            run_continuous_lenia(&compiled, &[0.5], 1, 1, &ContinuousRunConfig::new(1)).unwrap();
        assert_approximately(result.final_state[0], 0.6);
    }

    #[test]
    fn every_boundary_policy_has_explicit_asymmetric_sampling() {
        let kernel = || DenseContinuousKernel {
            width: 2,
            height: 1,
            anchor_x: 0,
            anchor_y: 0,
            weights: vec![0.0, 1.0],
        };
        let domain = ContinuousDomain {
            minimum: 0.0,
            maximum: 100.0,
        };
        let cases = [
            (ContinuousBoundaryPolicy::Clamp, vec![3.0, 4.0]),
            (ContinuousBoundaryPolicy::Reflect, vec![3.0, 3.0]),
            (ContinuousBoundaryPolicy::Wrap, vec![3.0, 3.0]),
            (
                ContinuousBoundaryPolicy::Constant { value: 0.0 },
                vec![3.0, 2.0],
            ),
        ];

        for (boundary, expected) in cases {
            let compiled = compile_continuous_lenia(
                linear_program(kernel(), boundary, domain, 1.0, 0.0),
                ContinuousCompileLimits::default(),
            )
            .unwrap();
            let result =
                run_continuous_lenia(&compiled, &[1.0, 2.0], 2, 1, &ContinuousRunConfig::new(1))
                    .unwrap();
            assert_eq!(result.final_state, expected, "{boundary:?}");
        }
    }

    #[test]
    fn exact_fixed_points_fast_forward_but_approximate_tolerance_does_not() {
        let exact = compile_continuous_lenia(
            one_cell_program(ContinuousGrowthFunction::Linear {
                slope: 0.0,
                intercept: 0.0,
            }),
            ContinuousCompileLimits::default(),
        )
        .unwrap();
        let exact_result =
            run_continuous_lenia(&exact, &[0.4], 1, 1, &ContinuousRunConfig::new(100)).unwrap();
        assert_eq!(exact_result.diagnostics.transition_steps_executed, 1);
        assert_eq!(
            exact_result.diagnostics.exact_fixed_point_generation,
            Some(0)
        );
        assert_eq!(exact_result.cycles_completed, 100);

        let approximate = compile_continuous_lenia(
            one_cell_program(ContinuousGrowthFunction::Linear {
                slope: 0.0,
                intercept: 0.001,
            }),
            ContinuousCompileLimits::default(),
        )
        .unwrap();
        let config = ContinuousRunConfig {
            convergence_tolerance: 0.01,
            ..ContinuousRunConfig::new(3)
        };
        let approximate_result = run_continuous_lenia(&approximate, &[0.4], 1, 1, &config).unwrap();
        assert_eq!(approximate_result.diagnostics.transition_steps_executed, 3);
        assert_eq!(
            approximate_result
                .diagnostics
                .last_transition_within_tolerance,
            Some(true)
        );
    }

    #[test]
    fn zero_cycles_returns_generation_zero_with_only_the_final_buffer_budgeted() {
        let compiled = compile_continuous_lenia(
            one_cell_program(ContinuousGrowthFunction::Linear {
                slope: 0.0,
                intercept: 1.0,
            }),
            ContinuousCompileLimits::default(),
        )
        .unwrap();
        let config = ContinuousRunConfig {
            max_working_memory_bytes: 8,
            ..ContinuousRunConfig::new(0)
        };
        let result = run_continuous_lenia(&compiled, &[0.25, 0.75], 2, 1, &config).unwrap();

        assert_eq!(result.final_state, [0.25, 0.75]);
        assert_eq!(result.diagnostics.transition_steps_executed, 0);
        assert_eq!(result.diagnostics.last_transition_max_delta, None);
        assert_eq!(result.diagnostics.last_transition_within_tolerance, None);
        assert_eq!(result.diagnostics.final_mean, 0.5);

        assert!(matches!(
            run_continuous_lenia(
                &compiled,
                &[0.25, 0.75],
                2,
                1,
                &ContinuousRunConfig {
                    max_working_memory_bytes: 7,
                    ..config
                },
            ),
            Err(ContinuousAutomataError::BudgetExceeded {
                required_bytes: 8,
                ..
            })
        ));
    }

    #[test]
    fn multi_row_evolution_is_synchronous_and_thread_deterministic() {
        let program = ContinuousLeniaProgram {
            version: CONTINUOUS_LENIA_PROGRAM_VERSION,
            domain: ContinuousDomain::UNIT,
            kernel: DenseContinuousKernel {
                width: 3,
                height: 3,
                anchor_x: 1,
                anchor_y: 1,
                weights: vec![0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0],
            },
            kernel_normalization: KernelNormalization::SumToOne,
            boundary: ContinuousBoundaryPolicy::Wrap,
            growth: ContinuousGrowthFunction::Linear {
                slope: 0.2,
                intercept: -0.05,
            },
            time_step: 0.1,
        };
        let compiled =
            compile_continuous_lenia(program, ContinuousCompileLimits::default()).unwrap();
        let initial = (0..35)
            .map(|index| (index % 11) as f32 / 10.0)
            .collect::<Vec<_>>();
        let config = ContinuousRunConfig::new(7);

        let one_thread = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run_continuous_lenia(&compiled, &initial, 7, 5, &config).unwrap());
        let four_threads = ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| run_continuous_lenia(&compiled, &initial, 7, 5, &config).unwrap());

        assert_eq!(one_thread, four_threads);
    }

    #[test]
    fn compilation_and_runtime_reject_invalid_versions_shapes_values_and_budgets() {
        let mut program = one_cell_program(ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1));
        program.version = 99;
        assert!(matches!(
            compile_continuous_lenia(program, ContinuousCompileLimits::default()),
            Err(ContinuousAutomataError::UnsupportedProgramVersion { found: 99 })
        ));

        let mut program = one_cell_program(ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1));
        program.kernel.weights = vec![f32::NAN];
        assert!(matches!(
            compile_continuous_lenia(
                program,
                ContinuousCompileLimits {
                    max_kernel_bytes: 3,
                },
            ),
            Err(ContinuousAutomataError::BudgetExceeded {
                context: "continuous kernel",
                ..
            })
        ));

        let malformed_kernel = ContinuousLeniaProgram {
            kernel: DenseContinuousKernel {
                width: 2,
                height: 2,
                anchor_x: 2,
                anchor_y: 0,
                weights: vec![1.0; 4],
            },
            ..one_cell_program(ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1))
        };
        assert!(matches!(
            compile_continuous_lenia(malformed_kernel, ContinuousCompileLimits::default()),
            Err(ContinuousAutomataError::KernelAnchorOutOfRange { .. })
        ));

        let invalid_boundary = ContinuousLeniaProgram {
            boundary: ContinuousBoundaryPolicy::Constant { value: 2.0 },
            ..one_cell_program(ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1))
        };
        assert!(matches!(
            compile_continuous_lenia(invalid_boundary, ContinuousCompileLimits::default()),
            Err(ContinuousAutomataError::InvalidBoundaryValue { value: 2.0, .. })
        ));

        let compiled = compile_continuous_lenia(
            one_cell_program(ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1)),
            ContinuousCompileLimits::default(),
        )
        .unwrap();
        assert!(matches!(
            run_continuous_lenia(&compiled, &[f32::NAN], 1, 1, &ContinuousRunConfig::new(1),),
            Err(ContinuousAutomataError::InvalidInitialStateValue { index: 0, .. })
        ));
        assert!(matches!(
            run_continuous_lenia(
                &compiled,
                &[0.0, 0.0],
                2,
                1,
                &ContinuousRunConfig {
                    max_working_memory_bytes: 15,
                    ..ContinuousRunConfig::new(1)
                },
            ),
            Err(ContinuousAutomataError::BudgetExceeded {
                context: "continuous working memory",
                required_bytes: 16,
                ..
            })
        ));
        assert!(matches!(
            run_continuous_lenia(
                &compiled,
                &[0.0],
                1,
                1,
                &ContinuousRunConfig {
                    version: 99,
                    ..ContinuousRunConfig::new(1)
                },
            ),
            Err(ContinuousAutomataError::UnsupportedRunConfigVersion { found: 99 })
        ));
        assert!(matches!(
            run_continuous_lenia(&compiled, &[0.0], 0, 1, &ContinuousRunConfig::new(1),),
            Err(ContinuousAutomataError::InvalidDimensions {
                width: 0,
                height: 1,
            })
        ));
    }

    #[test]
    fn program_json_is_versioned_tagged_and_strict() {
        let program = one_cell_program(ContinuousGrowthFunction::canonical_gaussian(0.5, 0.1));
        let json = serde_json::to_string(&program).unwrap();
        let decoded: ContinuousLeniaProgram = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, program);
        assert!(json.contains("\"kernel_normalization\":\"sum_to_one\""));
        assert!(json.contains("\"growth\":{\"gaussian\""));

        let mut value = serde_json::to_value(program).unwrap();
        value["kernel"]["surprise"] = serde_json::Value::Bool(true);
        assert!(serde_json::from_value::<ContinuousLeniaProgram>(value).is_err());
    }
}
