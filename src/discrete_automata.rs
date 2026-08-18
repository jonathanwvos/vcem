use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    fmt,
    hash::{Hash, Hasher},
    mem::size_of,
    sync::Arc,
};

pub const DISCRETE_PROGRAM_VERSION: u32 = 1;
pub const DISCRETE_RUN_CONFIG_VERSION: u32 = 1;
pub const DISCRETE_RESULT_VERSION: u32 = 1;

const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateStorage {
    U8,
    U16,
    U32,
}

impl StateStorage {
    pub fn for_state_count(state_count: u32) -> Result<Self, DiscreteAutomataError> {
        if state_count < 2 {
            return Err(DiscreteAutomataError::InvalidStateCount { state_count });
        }
        Ok(if state_count <= u8::MAX as u32 + 1 {
            Self::U8
        } else if state_count <= u16::MAX as u32 + 1 {
            Self::U16
        } else {
            Self::U32
        })
    }

    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 => size_of::<u8>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
        }
    }
}

/// A row-major state field or dense rule table in its exact compact width.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "storage",
    content = "values",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum CompactState {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl CompactState {
    pub fn pack(values: Vec<u32>, state_count: u32) -> Result<Self, DiscreteAutomataError> {
        Self::pack_with_budget(values, state_count, u64::MAX)
    }

    /// Packs logical `u32` states into the canonical width for `state_count`.
    /// The budget bounds the logical size of the packed result, excluding
    /// allocator metadata. It is enforced even when a canonical u32 source
    /// buffer can be reused without another allocation.
    pub fn pack_with_budget(
        values: Vec<u32>,
        state_count: u32,
        max_packed_bytes: u64,
    ) -> Result<Self, DiscreteAutomataError> {
        let storage = StateStorage::for_state_count(state_count)?;
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value >= state_count)
        {
            return Err(DiscreteAutomataError::StateValueOutOfRange {
                context: "packed state",
                index,
                value,
                state_count,
            });
        }
        let required = checked_bytes(values.len(), storage, "packed state")?;
        enforce_budget("packed state", required, max_packed_bytes)?;

        match storage {
            StateStorage::U8 => {
                let mut packed =
                    try_vec_with_capacity::<u8>(values.len(), required, "packed state")?;
                packed.extend(values.into_iter().map(|value| value as u8));
                Ok(Self::U8(packed))
            }
            StateStorage::U16 => {
                let mut packed =
                    try_vec_with_capacity::<u16>(values.len(), required, "packed state")?;
                packed.extend(values.into_iter().map(|value| value as u16));
                Ok(Self::U16(packed))
            }
            StateStorage::U32 => Ok(Self::U32(values)),
        }
    }

    pub fn filled_with_budget(
        len: usize,
        state_count: u32,
        value: u32,
        max_packed_bytes: u64,
    ) -> Result<Self, DiscreteAutomataError> {
        let storage = StateStorage::for_state_count(state_count)?;
        if value >= state_count {
            return Err(DiscreteAutomataError::StateValueOutOfRange {
                context: "filled state",
                index: 0,
                value,
                state_count,
            });
        }
        let required = checked_bytes(len, storage, "filled state")?;
        enforce_budget("filled state", required, max_packed_bytes)?;
        Ok(match storage {
            StateStorage::U8 => {
                let mut values = try_vec_with_capacity::<u8>(len, required, "filled state")?;
                values.resize(len, value as u8);
                Self::U8(values)
            }
            StateStorage::U16 => {
                let mut values = try_vec_with_capacity::<u16>(len, required, "filled state")?;
                values.resize(len, value as u16);
                Self::U16(values)
            }
            StateStorage::U32 => {
                let mut values = try_vec_with_capacity::<u32>(len, required, "filled state")?;
                values.resize(len, value);
                Self::U32(values)
            }
        })
    }

    pub const fn storage(&self) -> StateStorage {
        match self {
            Self::U8(_) => StateStorage::U8,
            Self::U16(_) => StateStorage::U16,
            Self::U32(_) => StateStorage::U32,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> Option<usize> {
        self.len().checked_mul(self.storage().byte_width())
    }

    pub fn get(&self, index: usize) -> Option<u32> {
        match self {
            Self::U8(values) => values.get(index).map(|value| *value as u32),
            Self::U16(values) => values.get(index).map(|value| *value as u32),
            Self::U32(values) => values.get(index).copied(),
        }
    }

    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::U16(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<&[u32]> {
        match self {
            Self::U32(values) => Some(values),
            _ => None,
        }
    }

    fn first_out_of_range(&self, state_count: u32) -> Option<(usize, u32)> {
        (0..self.len()).find_map(|index| {
            let value = self.get(index).expect("index is bounded by state length");
            (value >= state_count).then_some((index, value))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelOffset {
    pub dx: i32,
    pub dy: i32,
}

impl KernelOffset {
    pub const fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "state",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum BoundaryPolicy {
    Clamp,
    Reflect,
    Wrap,
    Constant(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateReducer {
    RoundedMean,
    Sum,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresenceReduction {
    LowestTrigger,
    HighestTrigger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresenceTransition {
    pub trigger_state: u32,
    pub output_state: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DiscreteRule {
    Direct {
        table: CompactState,
    },
    Aggregate {
        reducer: AggregateReducer,
        table: CompactState,
    },
    OrderedNeighborhood {
        table: CompactState,
    },
    PresenceTriggered {
        transitions: Vec<PresenceTransition>,
        reduction: PresenceReduction,
        fallback: CompactState,
    },
}

impl DiscreteRule {
    pub const fn kind(&self) -> DiscreteRuleKind {
        match self {
            Self::Direct { .. } => DiscreteRuleKind::Direct,
            Self::Aggregate { .. } => DiscreteRuleKind::Aggregate,
            Self::OrderedNeighborhood { .. } => DiscreteRuleKind::OrderedNeighborhood,
            Self::PresenceTriggered { .. } => DiscreteRuleKind::PresenceTriggered,
        }
    }

    const fn requires_kernel(&self) -> bool {
        !matches!(self, Self::Direct { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscreteRuleKind {
    Direct,
    Aggregate,
    OrderedNeighborhood,
    PresenceTriggered,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscreteProgram {
    pub version: u32,
    pub state_count: u32,
    pub storage: StateStorage,
    /// Stable sample order. For ordered rules, offset zero is the least
    /// significant base-q digit.
    pub kernel: Vec<KernelOffset>,
    pub boundary: BoundaryPolicy,
    pub rule: DiscreteRule,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileLimits {
    /// Maximum logical bytes occupied by dense transition tables. The limit
    /// excludes sparse presence entries, Arc/Vec metadata, and allocator
    /// rounding. Compilation moves table buffers into Arc<Vec<T>> without
    /// allocating a second dense buffer.
    pub max_table_bytes: u64,
}

impl Default for CompileLimits {
    fn default() -> Self {
        Self {
            max_table_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscreteRunConfig {
    pub version: u32,
    pub cycles: u32,
    /// Bounds current/scratch result allocations, excluding the caller-owned
    /// initial state, compiled rule tables, and allocator metadata.
    pub max_working_memory_bytes: u64,
    /// Enables collision-checked detection of cycles longer than two. This
    /// stores compact states up to the diagnostic budget. Fixed points and
    /// period-two cycles are always detected without state history.
    pub exact_cycle_detection: bool,
    /// Bounds exact state history plus the deterministic cycle index. The
    /// engine rejects a run before stepping if exact diagnostics do not fit.
    pub max_diagnostic_memory_bytes: u64,
}

impl DiscreteRunConfig {
    pub const fn new(cycles: u32) -> Self {
        Self {
            version: DISCRETE_RUN_CONFIG_VERSION,
            cycles,
            max_working_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            exact_cycle_detection: false,
            max_diagnostic_memory_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
        }
    }
}

impl Default for DiscreteRunConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CycleDiagnostic {
    pub start_generation: u32,
    pub period: u32,
    pub first_repeat_generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DiscreteDiagnostics {
    /// Actual transitions evaluated. This may be smaller than the requested
    /// cycles when an exact cycle permits deterministic fast-forwarding.
    pub transition_steps_executed: u32,
    pub distinct_states_observed: Option<u64>,
    pub fixed_point_generation: Option<u32>,
    pub cycle: Option<CycleDiagnostic>,
    pub final_nonzero_cells: usize,
    pub last_transition_changed_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DiscreteSimulationResult {
    pub version: u32,
    pub size: [usize; 2],
    pub state_count: u32,
    pub storage: StateStorage,
    pub cycles_completed: u32,
    pub final_state: CompactState,
    pub diagnostics: DiscreteDiagnostics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscreteAutomataError {
    UnsupportedProgramVersion {
        found: u32,
    },
    UnsupportedRunConfigVersion {
        found: u32,
    },
    InvalidStateCount {
        state_count: u32,
    },
    NonCanonicalStorage {
        state_count: u32,
        expected: StateStorage,
        actual: StateStorage,
    },
    EmptyKernel {
        rule: DiscreteRuleKind,
    },
    DuplicateKernelOffset {
        first_index: usize,
        duplicate_index: usize,
        offset: KernelOffset,
    },
    BoundaryStateOutOfRange {
        value: u32,
        state_count: u32,
    },
    TableLengthMismatch {
        rule: DiscreteRuleKind,
        expected: usize,
        actual: usize,
    },
    TableStorageMismatch {
        rule: DiscreteRuleKind,
        expected: StateStorage,
        actual: StateStorage,
    },
    StateValueOutOfRange {
        context: &'static str,
        index: usize,
        value: u32,
        state_count: u32,
    },
    DuplicatePresenceTrigger {
        trigger_state: u32,
    },
    SizeOverflow {
        context: &'static str,
    },
    MemoryBudgetExceeded {
        context: &'static str,
        required: u64,
        budget: u64,
    },
    AllocationFailed {
        context: &'static str,
        requested_bytes: usize,
    },
    InvalidDimensions {
        width: usize,
        height: usize,
        actual_len: usize,
    },
    InitialStorageMismatch {
        expected: StateStorage,
        actual: StateStorage,
    },
    InternalStorageMismatch,
}

impl fmt::Display for DiscreteAutomataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedProgramVersion { found } => {
                write!(formatter, "Unsupported discrete program version {found}.")
            }
            Self::UnsupportedRunConfigVersion { found } => {
                write!(
                    formatter,
                    "Unsupported discrete run config version {found}."
                )
            }
            Self::InvalidStateCount { state_count } => {
                write!(
                    formatter,
                    "Discrete state_count must be at least 2; found {state_count}."
                )
            }
            Self::NonCanonicalStorage {
                state_count,
                expected,
                actual,
            } => write!(
                formatter,
                "State count {state_count} requires {expected:?} storage, not {actual:?}."
            ),
            Self::EmptyKernel { rule } => {
                write!(
                    formatter,
                    "The {rule:?} rule requires at least one kernel sample."
                )
            }
            Self::DuplicateKernelOffset {
                first_index,
                duplicate_index,
                offset,
            } => write!(
                formatter,
                "Kernel offset ({}, {}) is duplicated at indices {first_index} and {duplicate_index}.",
                offset.dx, offset.dy
            ),
            Self::BoundaryStateOutOfRange { value, state_count } => write!(
                formatter,
                "Constant boundary state {value} is outside state_count {state_count}."
            ),
            Self::TableLengthMismatch {
                rule,
                expected,
                actual,
            } => write!(
                formatter,
                "The {rule:?} table has {actual} entries; expected {expected}."
            ),
            Self::TableStorageMismatch {
                rule,
                expected,
                actual,
            } => write!(
                formatter,
                "The {rule:?} table uses {actual:?} storage; expected {expected:?}."
            ),
            Self::StateValueOutOfRange {
                context,
                index,
                value,
                state_count,
            } => write!(
                formatter,
                "{context} entry {index} is state {value}, outside state_count {state_count}."
            ),
            Self::DuplicatePresenceTrigger { trigger_state } => write!(
                formatter,
                "Presence trigger state {trigger_state} is defined more than once."
            ),
            Self::SizeOverflow { context } => {
                write!(formatter, "{context} size overflowed this platform.")
            }
            Self::MemoryBudgetExceeded {
                context,
                required,
                budget,
            } => write!(
                formatter,
                "{context} requires {required} bytes, exceeding its {budget}-byte budget."
            ),
            Self::AllocationFailed {
                context,
                requested_bytes,
            } => write!(
                formatter,
                "Could not allocate {requested_bytes} bytes for {context}."
            ),
            Self::InvalidDimensions {
                width,
                height,
                actual_len,
            } => write!(
                formatter,
                "A {width} x {height} state field does not match {actual_len} entries."
            ),
            Self::InitialStorageMismatch { expected, actual } => write!(
                formatter,
                "Initial state uses {actual:?} storage; expected {expected:?}."
            ),
            Self::InternalStorageMismatch => {
                formatter.write_str("Compiled rule and state storage do not match.")
            }
        }
    }
}

impl std::error::Error for DiscreteAutomataError {}

#[derive(Clone, Debug)]
enum SharedTable {
    U8(Arc<Vec<u8>>),
    U16(Arc<Vec<u16>>),
    U32(Arc<Vec<u32>>),
}

impl SharedTable {
    fn from_compact(table: CompactState) -> Self {
        match table {
            CompactState::U8(values) => Self::U8(Arc::new(values)),
            CompactState::U16(values) => Self::U16(Arc::new(values)),
            CompactState::U32(values) => Self::U32(Arc::new(values)),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    #[cfg(test)]
    fn strong_count(&self) -> usize {
        match self {
            Self::U8(values) => Arc::strong_count(values),
            Self::U16(values) => Arc::strong_count(values),
            Self::U32(values) => Arc::strong_count(values),
        }
    }
}

#[derive(Clone, Debug)]
enum CompiledRule {
    Direct {
        table: SharedTable,
    },
    Aggregate {
        reducer: AggregateReducer,
        table: SharedTable,
    },
    OrderedNeighborhood {
        table: SharedTable,
    },
    PresenceTriggered {
        transitions: Arc<Vec<PresenceTransition>>,
        reduction: PresenceReduction,
        fallback: SharedTable,
    },
}

impl CompiledRule {
    fn kind(&self) -> DiscreteRuleKind {
        match self {
            Self::Direct { .. } => DiscreteRuleKind::Direct,
            Self::Aggregate { .. } => DiscreteRuleKind::Aggregate,
            Self::OrderedNeighborhood { .. } => DiscreteRuleKind::OrderedNeighborhood,
            Self::PresenceTriggered { .. } => DiscreteRuleKind::PresenceTriggered,
        }
    }

    fn table(&self) -> &SharedTable {
        match self {
            Self::Direct { table }
            | Self::Aggregate { table, .. }
            | Self::OrderedNeighborhood { table } => table,
            Self::PresenceTriggered { fallback, .. } => fallback,
        }
    }
}

/// A validated discrete model. Clones share the dense transition table.
#[derive(Clone, Debug)]
pub struct CompiledDiscreteAutomaton {
    state_count: u32,
    storage: StateStorage,
    kernel: Arc<Vec<KernelOffset>>,
    boundary: BoundaryPolicy,
    rule: CompiledRule,
}

impl CompiledDiscreteAutomaton {
    pub fn state_count(&self) -> u32 {
        self.state_count
    }

    pub fn storage(&self) -> StateStorage {
        self.storage
    }

    pub fn kernel(&self) -> &[KernelOffset] {
        self.kernel.as_slice()
    }

    pub fn boundary(&self) -> BoundaryPolicy {
        self.boundary
    }

    pub fn rule_kind(&self) -> DiscreteRuleKind {
        self.rule.kind()
    }

    pub fn dense_table_len(&self) -> usize {
        self.rule.table().len()
    }

    pub fn dense_table_bytes(&self) -> usize {
        self.dense_table_len() * self.storage.byte_width()
    }
}

/// Validates and compiles a serialized discrete program. The program is
/// consumed so its already-packed dense table can move into shared storage
/// without a second table-sized allocation.
pub fn compile_discrete_program(
    program: DiscreteProgram,
    limits: CompileLimits,
) -> Result<CompiledDiscreteAutomaton, DiscreteAutomataError> {
    if program.version != DISCRETE_PROGRAM_VERSION {
        return Err(DiscreteAutomataError::UnsupportedProgramVersion {
            found: program.version,
        });
    }
    let expected_storage = StateStorage::for_state_count(program.state_count)?;
    let state_count_usize =
        usize::try_from(program.state_count).map_err(|_| DiscreteAutomataError::SizeOverflow {
            context: "discrete state count",
        })?;
    if program.storage != expected_storage {
        return Err(DiscreteAutomataError::NonCanonicalStorage {
            state_count: program.state_count,
            expected: expected_storage,
            actual: program.storage,
        });
    }
    if let BoundaryPolicy::Constant(value) = program.boundary
        && value >= program.state_count
    {
        return Err(DiscreteAutomataError::BoundaryStateOutOfRange {
            value,
            state_count: program.state_count,
        });
    }
    if program.rule.requires_kernel() && program.kernel.is_empty() {
        return Err(DiscreteAutomataError::EmptyKernel {
            rule: program.rule.kind(),
        });
    }
    let mut offset_indices = HashMap::new();
    offset_indices
        .try_reserve(program.kernel.len())
        .map_err(|_| DiscreteAutomataError::AllocationFailed {
            context: "kernel offset index",
            requested_bytes: program
                .kernel
                .len()
                .saturating_mul(size_of::<(KernelOffset, usize)>()),
        })?;
    for (index, offset) in program.kernel.iter().copied().enumerate() {
        if let Some(first_index) = offset_indices.insert(offset, index) {
            return Err(DiscreteAutomataError::DuplicateKernelOffset {
                first_index,
                duplicate_index: index,
                offset,
            });
        }
    }

    let state_count = program.state_count;
    let kernel_len = program.kernel.len();
    let rule = match program.rule {
        DiscreteRule::Direct { table } => {
            validate_dense_table(
                &table,
                state_count,
                expected_storage,
                DiscreteRuleKind::Direct,
                state_count_usize,
                limits.max_table_bytes,
            )?;
            CompiledRule::Direct {
                table: SharedTable::from_compact(table),
            }
        }
        DiscreteRule::Aggregate { reducer, table } => {
            let expected_len = match reducer {
                AggregateReducer::RoundedMean => state_count_usize,
                AggregateReducer::Sum => kernel_len
                    .checked_mul(state_count_usize - 1)
                    .and_then(|maximum| maximum.checked_add(1))
                    .ok_or(DiscreteAutomataError::SizeOverflow {
                        context: "aggregate sum table",
                    })?,
            };
            validate_dense_table(
                &table,
                state_count,
                expected_storage,
                DiscreteRuleKind::Aggregate,
                expected_len,
                limits.max_table_bytes,
            )?;
            CompiledRule::Aggregate {
                reducer,
                table: SharedTable::from_compact(table),
            }
        }
        DiscreteRule::OrderedNeighborhood { table } => {
            let expected_len =
                checked_power(state_count_usize, kernel_len, "ordered-neighborhood table")?;
            validate_dense_table(
                &table,
                state_count,
                expected_storage,
                DiscreteRuleKind::OrderedNeighborhood,
                expected_len,
                limits.max_table_bytes,
            )?;
            CompiledRule::OrderedNeighborhood {
                table: SharedTable::from_compact(table),
            }
        }
        DiscreteRule::PresenceTriggered {
            mut transitions,
            reduction,
            fallback,
        } => {
            validate_dense_table(
                &fallback,
                state_count,
                expected_storage,
                DiscreteRuleKind::PresenceTriggered,
                state_count_usize,
                limits.max_table_bytes,
            )?;
            transitions.sort_unstable_by_key(|transition| transition.trigger_state);
            for (index, transition) in transitions.iter().copied().enumerate() {
                if transition.trigger_state >= state_count {
                    return Err(DiscreteAutomataError::StateValueOutOfRange {
                        context: "presence trigger",
                        index,
                        value: transition.trigger_state,
                        state_count,
                    });
                }
                if transition.output_state >= state_count {
                    return Err(DiscreteAutomataError::StateValueOutOfRange {
                        context: "presence output",
                        index,
                        value: transition.output_state,
                        state_count,
                    });
                }
                if index > 0 && transitions[index - 1].trigger_state == transition.trigger_state {
                    return Err(DiscreteAutomataError::DuplicatePresenceTrigger {
                        trigger_state: transition.trigger_state,
                    });
                }
            }
            CompiledRule::PresenceTriggered {
                transitions: Arc::new(transitions),
                reduction,
                fallback: SharedTable::from_compact(fallback),
            }
        }
    };

    Ok(CompiledDiscreteAutomaton {
        state_count,
        storage: expected_storage,
        kernel: Arc::new(program.kernel),
        boundary: program.boundary,
        rule,
    })
}

fn validate_dense_table(
    table: &CompactState,
    state_count: u32,
    expected_storage: StateStorage,
    rule: DiscreteRuleKind,
    expected_len: usize,
    budget: u64,
) -> Result<(), DiscreteAutomataError> {
    if table.storage() != expected_storage {
        return Err(DiscreteAutomataError::TableStorageMismatch {
            rule,
            expected: expected_storage,
            actual: table.storage(),
        });
    }
    if table.len() != expected_len {
        return Err(DiscreteAutomataError::TableLengthMismatch {
            rule,
            expected: expected_len,
            actual: table.len(),
        });
    }
    let required = checked_bytes(expected_len, expected_storage, "transition table")?;
    enforce_budget("transition table", required, budget)?;
    if let Some((index, value)) = table.first_out_of_range(state_count) {
        return Err(DiscreteAutomataError::StateValueOutOfRange {
            context: "transition table",
            index,
            value,
            state_count,
        });
    }
    Ok(())
}

fn checked_power(
    base: usize,
    exponent: usize,
    context: &'static str,
) -> Result<usize, DiscreteAutomataError> {
    let mut result = 1_usize;
    for _ in 0..exponent {
        result = result
            .checked_mul(base)
            .ok_or(DiscreteAutomataError::SizeOverflow { context })?;
    }
    Ok(result)
}

trait Symbol: Copy + Eq + Hash + Send + Sync + 'static {
    fn from_u32(value: u32) -> Self;
    fn to_u32(self) -> u32;
}

impl Symbol for u8 {
    fn from_u32(value: u32) -> Self {
        value as u8
    }

    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl Symbol for u16 {
    fn from_u32(value: u32) -> Self {
        value as u16
    }

    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl Symbol for u32 {
    fn from_u32(value: u32) -> Self {
        value
    }

    fn to_u32(self) -> u32 {
        self
    }
}

#[derive(Clone, Copy)]
enum TypedRule<'a, T> {
    Direct {
        table: &'a [T],
    },
    Aggregate {
        reducer: AggregateReducer,
        table: &'a [T],
    },
    OrderedNeighborhood {
        table: &'a [T],
    },
    PresenceTriggered {
        transitions: &'a [PresenceTransition],
        reduction: PresenceReduction,
        fallback: &'a [T],
    },
}

macro_rules! typed_rule_view {
    ($name:ident, $variant:ident, $symbol:ty) => {
        fn $name(&self) -> Result<TypedRule<'_, $symbol>, DiscreteAutomataError> {
            Ok(match self {
                Self::Direct {
                    table: SharedTable::$variant(table),
                } => TypedRule::Direct {
                    table: table.as_slice(),
                },
                Self::Aggregate {
                    reducer,
                    table: SharedTable::$variant(table),
                } => TypedRule::Aggregate {
                    reducer: *reducer,
                    table: table.as_slice(),
                },
                Self::OrderedNeighborhood {
                    table: SharedTable::$variant(table),
                } => TypedRule::OrderedNeighborhood {
                    table: table.as_slice(),
                },
                Self::PresenceTriggered {
                    transitions,
                    reduction,
                    fallback: SharedTable::$variant(fallback),
                } => TypedRule::PresenceTriggered {
                    transitions: transitions.as_slice(),
                    reduction: *reduction,
                    fallback: fallback.as_slice(),
                },
                _ => return Err(DiscreteAutomataError::InternalStorageMismatch),
            })
        }
    };
}

impl CompiledRule {
    typed_rule_view!(view_u8, U8, u8);
    typed_rule_view!(view_u16, U16, u16);
    typed_rule_view!(view_u32, U32, u32);
}

#[derive(Clone, Copy, Debug, Default)]
struct StepStats {
    changed_cells: usize,
    matches_two_generations_back: bool,
}

/// Evolves an already-quantized row-major state field. The source image is
/// not available to the transition loop.
pub fn run_discrete_automaton(
    automaton: &CompiledDiscreteAutomaton,
    initial_state: &CompactState,
    width: usize,
    height: usize,
    config: &DiscreteRunConfig,
) -> Result<DiscreteSimulationResult, DiscreteAutomataError> {
    if config.version != DISCRETE_RUN_CONFIG_VERSION {
        return Err(DiscreteAutomataError::UnsupportedRunConfigVersion {
            found: config.version,
        });
    }
    let expected_len = width
        .checked_mul(height)
        .ok_or(DiscreteAutomataError::SizeOverflow {
            context: "discrete state dimensions",
        })?;
    if width == 0 || height == 0 || initial_state.len() != expected_len {
        return Err(DiscreteAutomataError::InvalidDimensions {
            width,
            height,
            actual_len: initial_state.len(),
        });
    }
    if initial_state.storage() != automaton.storage {
        return Err(DiscreteAutomataError::InitialStorageMismatch {
            expected: automaton.storage,
            actual: initial_state.storage(),
        });
    }
    if let Some((index, value)) = initial_state.first_out_of_range(automaton.state_count) {
        return Err(DiscreteAutomataError::StateValueOutOfRange {
            context: "initial state",
            index,
            value,
            state_count: automaton.state_count,
        });
    }

    let state_bytes = checked_bytes(expected_len, automaton.storage, "discrete state")?;
    let working_bytes = state_bytes
        .checked_mul(2)
        .ok_or(DiscreteAutomataError::SizeOverflow {
            context: "discrete working buffers",
        })?;
    enforce_budget(
        "discrete working buffers",
        working_bytes,
        config.max_working_memory_bytes,
    )?;
    if config.exact_cycle_detection {
        let record_bytes = state_bytes
            .checked_add(size_of::<(u64, Vec<u32>)>())
            .ok_or(DiscreteAutomataError::SizeOverflow {
                context: "exact cycle diagnostics",
            })?;
        let record_count = usize::try_from(config.cycles)
            .ok()
            .and_then(|cycles| cycles.checked_add(1))
            .ok_or(DiscreteAutomataError::SizeOverflow {
                context: "exact cycle diagnostics",
            })?;
        let diagnostic_bytes =
            record_count
                .checked_mul(record_bytes)
                .ok_or(DiscreteAutomataError::SizeOverflow {
                    context: "exact cycle diagnostics",
                })?;
        enforce_budget(
            "exact cycle diagnostics",
            diagnostic_bytes,
            config.max_diagnostic_memory_bytes,
        )?;
    }

    let (final_state, diagnostics) = match initial_state {
        CompactState::U8(values) => {
            let (state, diagnostics) = run_typed(
                automaton,
                values,
                automaton.rule.view_u8()?,
                width,
                height,
                config,
            )?;
            (CompactState::U8(state), diagnostics)
        }
        CompactState::U16(values) => {
            let (state, diagnostics) = run_typed(
                automaton,
                values,
                automaton.rule.view_u16()?,
                width,
                height,
                config,
            )?;
            (CompactState::U16(state), diagnostics)
        }
        CompactState::U32(values) => {
            let (state, diagnostics) = run_typed(
                automaton,
                values,
                automaton.rule.view_u32()?,
                width,
                height,
                config,
            )?;
            (CompactState::U32(state), diagnostics)
        }
    };

    Ok(DiscreteSimulationResult {
        version: DISCRETE_RESULT_VERSION,
        size: [width, height],
        state_count: automaton.state_count,
        storage: automaton.storage,
        cycles_completed: config.cycles,
        final_state,
        diagnostics,
    })
}

fn run_typed<T: Symbol>(
    automaton: &CompiledDiscreteAutomaton,
    initial: &[T],
    rule: TypedRule<'_, T>,
    width: usize,
    height: usize,
    config: &DiscreteRunConfig,
) -> Result<(Vec<T>, DiscreteDiagnostics), DiscreteAutomataError> {
    let byte_len =
        initial
            .len()
            .checked_mul(size_of::<T>())
            .ok_or(DiscreteAutomataError::SizeOverflow {
                context: "discrete state",
            })?;
    let mut current = try_clone_slice(initial, byte_len, "discrete current state")?;
    let mut next = try_vec_with_capacity::<T>(initial.len(), byte_len, "discrete scratch state")?;
    next.resize(initial.len(), T::from_u32(0));

    let initial_nonzero = current.iter().filter(|value| value.to_u32() != 0).count();
    if config.cycles == 0 {
        return Ok((
            current,
            DiscreteDiagnostics {
                transition_steps_executed: 0,
                distinct_states_observed: config.exact_cycle_detection.then_some(1),
                fixed_point_generation: None,
                cycle: None,
                final_nonzero_cells: initial_nonzero,
                last_transition_changed_cells: 0,
            },
        ));
    }

    let mut history = if config.exact_cycle_detection {
        let record_count = usize::try_from(config.cycles)
            .ok()
            .and_then(|cycles| cycles.checked_add(1))
            .ok_or(DiscreteAutomataError::SizeOverflow {
                context: "exact cycle diagnostic records",
            })?;
        let mut history = Vec::new();
        history.try_reserve_exact(record_count).map_err(|_| {
            DiscreteAutomataError::AllocationFailed {
                context: "exact cycle diagnostic records",
                requested_bytes: usize::try_from(config.max_diagnostic_memory_bytes)
                    .unwrap_or(usize::MAX),
            }
        })?;
        history.push((
            state_fingerprint(&current),
            try_clone_slice(&current, byte_len, "cycle history")?,
        ));
        Some(history)
    } else {
        None
    };

    let mut transition_steps_executed = 0_u32;
    let mut known_distinct_states = None;
    let mut fixed_point_generation = None;
    let mut detected_cycle = None;
    let mut last_stats = StepStats {
        changed_cells: 0,
        matches_two_generations_back: false,
    };

    for generation in 1..=config.cycles {
        last_stats = step_typed(
            &current,
            &mut next,
            width,
            height,
            automaton.state_count,
            automaton.kernel.as_slice(),
            automaton.boundary,
            rule,
            generation >= 2,
        );
        transition_steps_executed += 1;
        std::mem::swap(&mut current, &mut next);

        let cycle = if last_stats.changed_cells == 0 {
            fixed_point_generation = Some(generation - 1);
            Some(CycleDiagnostic {
                start_generation: generation - 1,
                period: 1,
                first_repeat_generation: generation,
            })
        } else if generation >= 2 && last_stats.matches_two_generations_back {
            Some(CycleDiagnostic {
                start_generation: generation - 2,
                period: 2,
                first_repeat_generation: generation,
            })
        } else if let Some(history) = &history {
            let fingerprint = state_fingerprint(&current);
            history
                .iter()
                .enumerate()
                .find(|(_, (candidate_fingerprint, candidate))| {
                    *candidate_fingerprint == fingerprint && candidate.as_slice() == current
                })
                .map(|(start_generation, _)| {
                    let start_generation = u32::try_from(start_generation)
                        .expect("history length is bounded by the u32 cycle count");
                    CycleDiagnostic {
                        start_generation,
                        period: generation - start_generation,
                        first_repeat_generation: generation,
                    }
                })
        } else {
            None
        };

        if let Some(cycle) = cycle {
            known_distinct_states = Some(u64::from(cycle.first_repeat_generation));
            detected_cycle = Some(cycle);
            let remaining = config.cycles - generation;
            let required_steps = remaining % cycle.period;
            for _ in 0..required_steps {
                last_stats = step_typed(
                    &current,
                    &mut next,
                    width,
                    height,
                    automaton.state_count,
                    automaton.kernel.as_slice(),
                    automaton.boundary,
                    rule,
                    false,
                );
                transition_steps_executed += 1;
                std::mem::swap(&mut current, &mut next);
            }
            break;
        }

        if let Some(history) = &mut history {
            history.push((
                state_fingerprint(&current),
                try_clone_slice(&current, byte_len, "cycle history")?,
            ));
            known_distinct_states = Some(
                u64::try_from(history.len())
                    .expect("history length is addressable and therefore fits u64"),
            );
        }
    }

    let final_nonzero_cells = current.iter().filter(|value| value.to_u32() != 0).count();
    Ok((
        current,
        DiscreteDiagnostics {
            transition_steps_executed,
            distinct_states_observed: known_distinct_states,
            fixed_point_generation,
            cycle: detected_cycle,
            final_nonzero_cells,
            last_transition_changed_cells: last_stats.changed_cells,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn step_typed<T: Symbol>(
    current: &[T],
    next: &mut [T],
    width: usize,
    height: usize,
    state_count: u32,
    kernel: &[KernelOffset],
    boundary: BoundaryPolicy,
    rule: TypedRule<'_, T>,
    compare_two_generations_back: bool,
) -> StepStats {
    match rule {
        TypedRule::Direct { table } => step_cells(
            current,
            next,
            width,
            compare_two_generations_back,
            |_, _, index| table[current[index].to_u32() as usize],
        ),
        TypedRule::Aggregate { reducer, table } => step_cells(
            current,
            next,
            width,
            compare_two_generations_back,
            |x, y, _| {
                let sum = kernel.iter().fold(0_u128, |sum, offset| {
                    sum + sample_state(current, width, height, x, y, *offset, boundary) as u128
                });
                let table_index = match reducer {
                    AggregateReducer::RoundedMean => {
                        let count = kernel.len() as u128;
                        ((sum * 2 + count) / (count * 2)) as usize
                    }
                    AggregateReducer::Sum => sum as usize,
                };
                table[table_index]
            },
        ),
        TypedRule::OrderedNeighborhood { table } => {
            let state_count =
                usize::try_from(state_count).expect("compiled discrete state count is addressable");
            step_cells(
                current,
                next,
                width,
                compare_two_generations_back,
                |x, y, _| {
                    let mut table_index = 0_usize;
                    let mut multiplier = 1_usize;
                    for (sample_index, offset) in kernel.iter().enumerate() {
                        let value =
                            sample_state(current, width, height, x, y, *offset, boundary) as usize;
                        table_index += value * multiplier;
                        if sample_index + 1 < kernel.len() {
                            multiplier *= state_count;
                        }
                    }
                    table[table_index]
                },
            )
        }
        TypedRule::PresenceTriggered {
            transitions,
            reduction,
            fallback,
        } => step_cells(
            current,
            next,
            width,
            compare_two_generations_back,
            |x, y, index| {
                let mut selected = None::<PresenceTransition>;
                for offset in kernel {
                    let sensed = sample_state(current, width, height, x, y, *offset, boundary);
                    if let Ok(transition_index) = transitions
                        .binary_search_by_key(&sensed, |transition| transition.trigger_state)
                    {
                        let transition = transitions[transition_index];
                        selected = Some(match (selected, reduction) {
                            (None, _) => transition,
                            (Some(previous), PresenceReduction::LowestTrigger) => {
                                if transition.trigger_state < previous.trigger_state {
                                    transition
                                } else {
                                    previous
                                }
                            }
                            (Some(previous), PresenceReduction::HighestTrigger) => {
                                if transition.trigger_state > previous.trigger_state {
                                    transition
                                } else {
                                    previous
                                }
                            }
                        });
                    }
                }
                selected.map_or_else(
                    || fallback[current[index].to_u32() as usize],
                    |transition| T::from_u32(transition.output_state),
                )
            },
        ),
    }
}

fn step_cells<T: Symbol>(
    current: &[T],
    next: &mut [T],
    width: usize,
    compare_two_generations_back: bool,
    transition: impl Fn(usize, usize, usize) -> T + Sync,
) -> StepStats {
    next.par_chunks_mut(width)
        .enumerate()
        .map(|(y, row)| {
            let mut stats = StepStats {
                matches_two_generations_back: compare_two_generations_back,
                ..StepStats::default()
            };
            for (x, output) in row.iter_mut().enumerate() {
                let index = y * width + x;
                let previous_output = *output;
                let updated = transition(x, y, index);
                stats.changed_cells += usize::from(updated != current[index]);
                if compare_two_generations_back && updated != previous_output {
                    stats.matches_two_generations_back = false;
                }
                *output = updated;
            }
            stats
        })
        .reduce(
            || StepStats {
                matches_two_generations_back: compare_two_generations_back,
                ..StepStats::default()
            },
            |left, right| StepStats {
                changed_cells: left.changed_cells + right.changed_cells,
                matches_two_generations_back: left.matches_two_generations_back
                    && right.matches_two_generations_back,
            },
        )
}

fn sample_state<T: Symbol>(
    state: &[T],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    offset: KernelOffset,
    boundary: BoundaryPolicy,
) -> u32 {
    let sample_x = resolve_coordinate(x, offset.dx, width, boundary);
    let sample_y = resolve_coordinate(y, offset.dy, height, boundary);
    match (sample_x, sample_y, boundary) {
        (Some(sample_x), Some(sample_y), _) => state[sample_y * width + sample_x].to_u32(),
        (_, _, BoundaryPolicy::Constant(value)) => value,
        _ => unreachable!("non-constant boundary policies always resolve coordinates"),
    }
}

fn resolve_coordinate(
    coordinate: usize,
    offset: i32,
    length: usize,
    boundary: BoundaryPolicy,
) -> Option<usize> {
    let coordinate = coordinate as i128 + offset as i128;
    let length = length as i128;
    match boundary {
        BoundaryPolicy::Constant(_) if !(0..length).contains(&coordinate) => None,
        BoundaryPolicy::Constant(_) | BoundaryPolicy::Clamp => {
            Some(coordinate.clamp(0, length - 1) as usize)
        }
        BoundaryPolicy::Wrap => Some(coordinate.rem_euclid(length) as usize),
        BoundaryPolicy::Reflect => {
            if length == 1 {
                return Some(0);
            }
            let period = 2 * (length - 1);
            let reflected = coordinate.rem_euclid(period);
            Some(if reflected < length {
                reflected as usize
            } else {
                (period - reflected) as usize
            })
        }
    }
}

fn state_fingerprint<T: Hash>(state: &[T]) -> u64 {
    let mut hasher = DefaultHasher::new();
    state.hash(&mut hasher);
    hasher.finish()
}

fn try_clone_slice<T: Copy>(
    source: &[T],
    requested_bytes: usize,
    context: &'static str,
) -> Result<Vec<T>, DiscreteAutomataError> {
    let mut values = try_vec_with_capacity::<T>(source.len(), requested_bytes, context)?;
    values.extend_from_slice(source);
    Ok(values)
}

fn checked_bytes(
    len: usize,
    storage: StateStorage,
    context: &'static str,
) -> Result<usize, DiscreteAutomataError> {
    len.checked_mul(storage.byte_width())
        .ok_or(DiscreteAutomataError::SizeOverflow { context })
}

fn enforce_budget(
    context: &'static str,
    required: usize,
    budget: u64,
) -> Result<(), DiscreteAutomataError> {
    let required =
        u64::try_from(required).map_err(|_| DiscreteAutomataError::SizeOverflow { context })?;
    if required > budget {
        return Err(DiscreteAutomataError::MemoryBudgetExceeded {
            context,
            required,
            budget,
        });
    }
    Ok(())
}

fn try_vec_with_capacity<T>(
    len: usize,
    requested_bytes: usize,
    context: &'static str,
) -> Result<Vec<T>, DiscreteAutomataError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| DiscreteAutomataError::AllocationFailed {
            context,
            requested_bytes,
        })?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(values: impl Into<Vec<u32>>, state_count: u32) -> CompactState {
        CompactState::pack(values.into(), state_count).unwrap()
    }

    fn program(
        state_count: u32,
        kernel: Vec<KernelOffset>,
        boundary: BoundaryPolicy,
        rule: DiscreteRule,
    ) -> DiscreteProgram {
        DiscreteProgram {
            version: DISCRETE_PROGRAM_VERSION,
            state_count,
            storage: StateStorage::for_state_count(state_count).unwrap(),
            kernel,
            boundary,
            rule,
        }
    }

    fn compiled(program: DiscreteProgram) -> CompiledDiscreteAutomaton {
        compile_discrete_program(program, CompileLimits::default()).unwrap()
    }

    fn run(
        automaton: &CompiledDiscreteAutomaton,
        values: Vec<u32>,
        width: usize,
        height: usize,
        cycles: u32,
    ) -> DiscreteSimulationResult {
        let initial = packed(values, automaton.state_count());
        run_discrete_automaton(
            automaton,
            &initial,
            width,
            height,
            &DiscreteRunConfig::new(cycles),
        )
        .unwrap()
    }

    fn values(state: &CompactState) -> Vec<u32> {
        (0..state.len())
            .map(|index| state.get(index).unwrap())
            .collect()
    }

    #[test]
    fn compact_storage_uses_canonical_width_thresholds() {
        assert_eq!(
            StateStorage::for_state_count(256).unwrap(),
            StateStorage::U8
        );
        assert_eq!(
            StateStorage::for_state_count(257).unwrap(),
            StateStorage::U16
        );
        assert_eq!(
            StateStorage::for_state_count(65_536).unwrap(),
            StateStorage::U16
        );
        assert_eq!(
            StateStorage::for_state_count(65_537).unwrap(),
            StateStorage::U32
        );

        assert_eq!(packed(vec![0, 255], 256).storage(), StateStorage::U8);
        assert_eq!(packed(vec![0, 256], 257).storage(), StateStorage::U16);
        assert_eq!(packed(vec![0, 65_535], 65_536).storage(), StateStorage::U16);
        assert_eq!(packed(vec![0, 65_536], 65_537).storage(), StateStorage::U32);
    }

    #[test]
    fn compact_storage_rejects_invalid_values_and_budgets() {
        assert!(matches!(
            CompactState::pack(vec![0, 2], 2),
            Err(DiscreteAutomataError::StateValueOutOfRange { .. })
        ));
        assert!(matches!(
            CompactState::pack_with_budget(vec![0; 10], 2, 9),
            Err(DiscreteAutomataError::MemoryBudgetExceeded { .. })
        ));
    }

    #[test]
    fn direct_rule_detects_and_fast_forwards_period_two() {
        let automaton = compiled(program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![1, 0], 2),
            },
        ));

        let result = run(&automaton, vec![0, 1, 0], 3, 1, 100);

        assert_eq!(values(&result.final_state), [0, 1, 0]);
        assert_eq!(result.cycles_completed, 100);
        assert_eq!(result.diagnostics.transition_steps_executed, 2);
        assert_eq!(result.diagnostics.distinct_states_observed, Some(2));
        assert_eq!(
            result.diagnostics.cycle,
            Some(CycleDiagnostic {
                start_generation: 0,
                period: 2,
                first_repeat_generation: 2,
            })
        );
    }

    #[test]
    fn direct_identity_is_an_exact_fixed_point() {
        let automaton = compiled(program(
            3,
            vec![],
            BoundaryPolicy::Clamp,
            DiscreteRule::Direct {
                table: packed(vec![0, 1, 2], 3),
            },
        ));

        let result = run(&automaton, vec![0, 2, 1], 3, 1, 50);

        assert_eq!(values(&result.final_state), [0, 2, 1]);
        assert_eq!(result.diagnostics.transition_steps_executed, 1);
        assert_eq!(result.diagnostics.fixed_point_generation, Some(0));
        assert_eq!(result.diagnostics.final_nonzero_cells, 2);
    }

    #[test]
    fn zero_cycles_returns_generation_zero_without_a_transition() {
        let automaton = compiled(program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![1, 0], 2),
            },
        ));

        let result = run(&automaton, vec![0, 1], 2, 1, 0);

        assert_eq!(values(&result.final_state), [0, 1]);
        assert_eq!(result.diagnostics.transition_steps_executed, 0);
        assert_eq!(result.diagnostics.cycle, None);
    }

    #[test]
    fn an_all_zero_frame_can_reanimate() {
        let automaton = compiled(program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![1, 0], 2),
            },
        ));

        let result = run(&automaton, vec![0, 0, 0], 3, 1, 1);

        assert_eq!(values(&result.final_state), [1, 1, 1]);
        assert_eq!(result.diagnostics.final_nonzero_cells, 3);
    }

    #[test]
    fn opt_in_exact_detection_finds_a_period_three_cycle() {
        let automaton = compiled(program(
            3,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![1, 2, 0], 3),
            },
        ));
        let initial = packed(vec![0], 3);
        let mut config = DiscreteRunConfig::new(100);
        config.exact_cycle_detection = true;

        let result = run_discrete_automaton(&automaton, &initial, 1, 1, &config).unwrap();

        assert_eq!(values(&result.final_state), [1]);
        assert_eq!(result.diagnostics.transition_steps_executed, 4);
        assert_eq!(result.diagnostics.distinct_states_observed, Some(3));
        assert_eq!(
            result.diagnostics.cycle,
            Some(CycleDiagnostic {
                start_generation: 0,
                period: 3,
                first_repeat_generation: 3,
            })
        );
    }

    #[test]
    fn rounded_mean_is_position_independent_and_rounds_half_up() {
        let table = packed(vec![0, 1, 2], 3);
        let first = compiled(program(
            3,
            vec![KernelOffset::new(0, 0), KernelOffset::new(1, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::RoundedMean,
                table: table.clone(),
            },
        ));
        let reversed = compiled(program(
            3,
            vec![KernelOffset::new(1, 0), KernelOffset::new(0, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::RoundedMean,
                table,
            },
        ));

        assert_eq!(
            values(&run(&first, vec![0, 2], 2, 1, 1).final_state),
            [1, 1]
        );
        assert_eq!(
            run(&first, vec![0, 2], 2, 1, 1).final_state,
            run(&reversed, vec![0, 2], 2, 1, 1).final_state
        );

        let binary = compiled(program(
            2,
            vec![KernelOffset::new(0, 0), KernelOffset::new(1, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::RoundedMean,
                table: packed(vec![0, 1], 2),
            },
        ));
        assert_eq!(
            values(&run(&binary, vec![0, 1], 2, 1, 1).final_state),
            [1, 1]
        );
    }

    #[test]
    fn aggregate_sum_uses_the_checked_full_sum_domain() {
        let automaton = compiled(program(
            3,
            vec![KernelOffset::new(0, 0), KernelOffset::new(1, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::Sum,
                table: packed(vec![0, 1, 2, 0, 1], 3),
            },
        ));

        assert_eq!(
            values(&run(&automaton, vec![0, 2], 2, 1, 1).final_state),
            [2, 2]
        );
    }

    #[test]
    fn ordered_neighborhood_uses_serialized_least_significant_first_order() {
        let table = packed(vec![0, 0, 1, 1], 2);
        let center_then_right = compiled(program(
            2,
            vec![KernelOffset::new(0, 0), KernelOffset::new(1, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::OrderedNeighborhood {
                table: table.clone(),
            },
        ));
        let right_then_center = compiled(program(
            2,
            vec![KernelOffset::new(1, 0), KernelOffset::new(0, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::OrderedNeighborhood { table },
        ));

        assert_eq!(
            values(&run(&center_then_right, vec![0, 1], 2, 1, 1).final_state),
            [1, 0]
        );
        assert_eq!(
            values(&run(&right_then_center, vec![0, 1], 2, 1, 1).final_state),
            [0, 1]
        );
    }

    #[test]
    fn every_boundary_policy_samples_non_square_corners_explicitly() {
        let run_boundary = |boundary| {
            let automaton = compiled(program(
                4,
                vec![KernelOffset::new(-1, -1)],
                boundary,
                DiscreteRule::Aggregate {
                    reducer: AggregateReducer::RoundedMean,
                    table: packed(vec![0, 1, 2, 3], 4),
                },
            ));
            values(&run(&automaton, vec![0, 1, 2, 3, 2, 1], 3, 2, 1).final_state)[0]
        };

        assert_eq!(run_boundary(BoundaryPolicy::Clamp), 0);
        assert_eq!(run_boundary(BoundaryPolicy::Wrap), 1);
        assert_eq!(run_boundary(BoundaryPolicy::Reflect), 2);
        assert_eq!(run_boundary(BoundaryPolicy::Constant(3)), 3);
    }

    #[test]
    fn presence_reduction_resolves_simultaneous_matches_explicitly() {
        let make = |reduction| {
            compiled(program(
                3,
                vec![KernelOffset::new(0, 0), KernelOffset::new(1, 0)],
                BoundaryPolicy::Wrap,
                DiscreteRule::PresenceTriggered {
                    transitions: vec![
                        PresenceTransition {
                            trigger_state: 1,
                            output_state: 2,
                        },
                        PresenceTransition {
                            trigger_state: 2,
                            output_state: 1,
                        },
                    ],
                    reduction,
                    fallback: packed(vec![0, 1, 2], 3),
                },
            ))
        };

        assert_eq!(
            values(&run(&make(PresenceReduction::LowestTrigger), vec![1, 2], 2, 1, 1).final_state),
            [2, 2]
        );
        assert_eq!(
            values(
                &run(
                    &make(PresenceReduction::HighestTrigger),
                    vec![1, 2],
                    2,
                    1,
                    1
                )
                .final_state
            ),
            [1, 1]
        );
    }

    #[test]
    fn compilation_rejects_bad_tables_kernels_boundaries_and_versions() {
        let short = program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: CompactState::U8(vec![0]),
            },
        );
        assert!(matches!(
            compile_discrete_program(short, CompileLimits::default()),
            Err(DiscreteAutomataError::TableLengthMismatch { .. })
        ));

        let bad_output = program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: CompactState::U8(vec![0, 2]),
            },
        );
        assert!(matches!(
            compile_discrete_program(bad_output, CompileLimits::default()),
            Err(DiscreteAutomataError::StateValueOutOfRange { .. })
        ));

        let duplicate = program(
            2,
            vec![KernelOffset::new(0, 0), KernelOffset::new(0, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::RoundedMean,
                table: packed(vec![0, 1], 2),
            },
        );
        assert!(matches!(
            compile_discrete_program(duplicate, CompileLimits::default()),
            Err(DiscreteAutomataError::DuplicateKernelOffset { .. })
        ));

        let empty = program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::RoundedMean,
                table: packed(vec![0, 1], 2),
            },
        );
        assert!(matches!(
            compile_discrete_program(empty, CompileLimits::default()),
            Err(DiscreteAutomataError::EmptyKernel { .. })
        ));

        let bad_boundary = program(
            2,
            vec![],
            BoundaryPolicy::Constant(2),
            DiscreteRule::Direct {
                table: packed(vec![0, 1], 2),
            },
        );
        assert!(matches!(
            compile_discrete_program(bad_boundary, CompileLimits::default()),
            Err(DiscreteAutomataError::BoundaryStateOutOfRange { .. })
        ));

        let mut bad_version = program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![0, 1], 2),
            },
        );
        bad_version.version = 99;
        assert!(matches!(
            compile_discrete_program(bad_version, CompileLimits::default()),
            Err(DiscreteAutomataError::UnsupportedProgramVersion { found: 99 })
        ));
    }

    #[test]
    fn ordered_power_and_table_byte_budget_are_checked() {
        let kernel = (0..usize::BITS)
            .map(|index| KernelOffset::new(index as i32, 0))
            .collect();
        let overflowing = program(
            2,
            kernel,
            BoundaryPolicy::Wrap,
            DiscreteRule::OrderedNeighborhood {
                table: CompactState::U8(vec![]),
            },
        );
        assert!(matches!(
            compile_discrete_program(overflowing, CompileLimits::default()),
            Err(DiscreteAutomataError::SizeOverflow { .. })
        ));

        let budgeted = program(
            257,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![0; 257], 257),
            },
        );
        assert!(matches!(
            compile_discrete_program(
                budgeted,
                CompileLimits {
                    max_table_bytes: 513,
                }
            ),
            Err(DiscreteAutomataError::MemoryBudgetExceeded { required: 514, .. })
        ));
    }

    #[test]
    fn runtime_rejects_invalid_dimensions_storage_values_and_budgets() {
        let automaton = compiled(program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![0, 1], 2),
            },
        ));
        let initial = packed(vec![0; 10], 2);
        let mut config = DiscreteRunConfig::new(1);
        config.max_working_memory_bytes = 19;
        assert!(matches!(
            run_discrete_automaton(&automaton, &initial, 10, 1, &config),
            Err(DiscreteAutomataError::MemoryBudgetExceeded { required: 20, .. })
        ));

        assert!(matches!(
            run_discrete_automaton(
                &automaton,
                &CompactState::U8(vec![0, 2]),
                2,
                1,
                &DiscreteRunConfig::new(1)
            ),
            Err(DiscreteAutomataError::StateValueOutOfRange { .. })
        ));
        assert!(matches!(
            run_discrete_automaton(
                &automaton,
                &packed(vec![0], 2),
                usize::MAX,
                2,
                &DiscreteRunConfig::new(1)
            ),
            Err(DiscreteAutomataError::SizeOverflow { .. })
        ));

        let wide = compiled(program(
            257,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![0; 257], 257),
            },
        ));
        assert!(matches!(
            run_discrete_automaton(
                &wide,
                &CompactState::U8(vec![0]),
                1,
                1,
                &DiscreteRunConfig::new(1)
            ),
            Err(DiscreteAutomataError::InitialStorageMismatch { .. })
        ));
    }

    #[test]
    fn exact_diagnostics_have_an_independent_budget() {
        let automaton = compiled(program(
            3,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![1, 2, 0], 3),
            },
        ));
        let initial = packed(vec![0; 10], 3);
        let mut config = DiscreteRunConfig::new(10);
        config.exact_cycle_detection = true;
        config.max_diagnostic_memory_bytes = 1;

        assert!(matches!(
            run_discrete_automaton(&automaton, &initial, 10, 1, &config),
            Err(DiscreteAutomataError::MemoryBudgetExceeded {
                context: "exact cycle diagnostics",
                ..
            })
        ));
    }

    #[test]
    fn compiled_clones_share_the_dense_table_and_runs_are_deterministic() {
        let automaton = compiled(program(
            2,
            vec![KernelOffset::new(0, 0), KernelOffset::new(1, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::OrderedNeighborhood {
                table: packed(vec![0, 1, 1, 0], 2),
            },
        ));
        assert_eq!(automaton.rule.table().strong_count(), 1);
        let cloned = automaton.clone();
        assert_eq!(automaton.rule.table().strong_count(), 2);

        let first = run(&automaton, vec![0, 1, 1, 0], 4, 1, 7);
        let second = run(&cloned, vec![0, 1, 1, 0], 4, 1, 7);
        assert_eq!(first, second);
    }

    #[test]
    fn compilation_moves_the_dense_table_without_copying_its_buffer() {
        let table = CompactState::U8(vec![0, 1]);
        let original_pointer = table.as_u8().unwrap().as_ptr();
        let automaton = compiled(program(
            2,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct { table },
        ));

        let SharedTable::U8(compiled_table) = automaton.rule.table() else {
            panic!("expected u8 table")
        };
        assert_eq!(compiled_table.as_ptr(), original_pointer);
    }

    #[test]
    fn u16_and_u32_storage_paths_execute_without_repacking() {
        let mut u16_table = vec![0; 257];
        u16_table[256] = 1;
        let u16_automaton = compiled(program(
            257,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(u16_table, 257),
            },
        ));
        let u16_result = run(&u16_automaton, vec![256], 1, 1, 1);
        assert_eq!(u16_result.storage, StateStorage::U16);
        assert_eq!(u16_result.final_state.as_u16(), Some([1].as_slice()));

        let mut u32_table = vec![0; 65_537];
        u32_table[65_536] = 2;
        let u32_automaton = compiled(program(
            65_537,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(u32_table, 65_537),
            },
        ));
        let u32_result = run(&u32_automaton, vec![65_536], 1, 1, 1);
        assert_eq!(u32_result.storage, StateStorage::U32);
        assert_eq!(u32_result.final_state.as_u32(), Some([2].as_slice()));
    }

    #[test]
    fn multi_row_cross_row_sensing_is_synchronous_and_thread_deterministic() {
        let automaton = compiled(program(
            10,
            vec![KernelOffset::new(0, 1)],
            BoundaryPolicy::Wrap,
            DiscreteRule::Aggregate {
                reducer: AggregateReducer::RoundedMean,
                table: packed((0..10).collect::<Vec<_>>(), 10),
            },
        ));
        let initial = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let run_in_pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| run(&automaton, initial.clone(), 3, 3, 1))
        };

        let serial = run_in_pool(1);
        let parallel = run_in_pool(4);
        assert_eq!(serial, parallel);
        assert_eq!(values(&serial.final_state), [3, 4, 5, 6, 7, 8, 0, 1, 2]);
    }

    #[test]
    fn transient_period_two_cycle_fast_forwards_with_the_correct_phase() {
        let automaton = compiled(program(
            4,
            vec![],
            BoundaryPolicy::Wrap,
            DiscreteRule::Direct {
                table: packed(vec![1, 2, 3, 2], 4),
            },
        ));

        let result = run(&automaton, vec![0], 1, 1, 9);

        assert_eq!(values(&result.final_state), [3]);
        assert_eq!(result.diagnostics.transition_steps_executed, 5);
        assert_eq!(
            result.diagnostics.cycle,
            Some(CycleDiagnostic {
                start_generation: 2,
                period: 2,
                first_repeat_generation: 4,
            })
        );
    }

    #[test]
    fn presence_fallback_and_duplicate_trigger_validation_are_explicit() {
        let fallback = compiled(program(
            3,
            vec![KernelOffset::new(0, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::PresenceTriggered {
                transitions: vec![PresenceTransition {
                    trigger_state: 2,
                    output_state: 0,
                }],
                reduction: PresenceReduction::LowestTrigger,
                fallback: packed(vec![0, 1, 2], 3),
            },
        ));
        assert_eq!(values(&run(&fallback, vec![1], 1, 1, 1).final_state), [1]);

        let duplicate = program(
            3,
            vec![KernelOffset::new(0, 0)],
            BoundaryPolicy::Wrap,
            DiscreteRule::PresenceTriggered {
                transitions: vec![
                    PresenceTransition {
                        trigger_state: 1,
                        output_state: 0,
                    },
                    PresenceTransition {
                        trigger_state: 1,
                        output_state: 2,
                    },
                ],
                reduction: PresenceReduction::HighestTrigger,
                fallback: packed(vec![0, 1, 2], 3),
            },
        );
        assert!(matches!(
            compile_discrete_program(duplicate, CompileLimits::default()),
            Err(DiscreteAutomataError::DuplicatePresenceTrigger { trigger_state: 1 })
        ));
    }

    #[test]
    fn program_json_is_tagged_versioned_and_strict() {
        let definition = program(
            2,
            vec![KernelOffset::new(0, 0)],
            BoundaryPolicy::Reflect,
            DiscreteRule::OrderedNeighborhood {
                table: packed(vec![0, 1], 2),
            },
        );

        let json = serde_json::to_string(&definition).unwrap();
        let decoded: DiscreteProgram = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, definition);
        assert!(json.contains("\"kind\":\"ordered_neighborhood\""));
        assert!(json.contains("\"storage\":\"u8\""));

        let unknown_field = r#"{
            "version":1,
            "state_count":2,
            "storage":"u8",
            "kernel":[],
            "boundary":{"kind":"wrap"},
            "rule":{"kind":"direct","table":{"storage":"u8","values":[0,1]}},
            "surprise":true
        }"#;
        assert!(serde_json::from_str::<DiscreteProgram>(unknown_field).is_err());

        let unknown_table_field = r#"{
            "version":1,
            "state_count":2,
            "storage":"u8",
            "kernel":[],
            "boundary":{"kind":"wrap"},
            "rule":{
                "kind":"direct",
                "table":{"storage":"u8","values":[0,1],"surprise":true}
            }
        }"#;
        assert!(serde_json::from_str::<DiscreteProgram>(unknown_table_field).is_err());
    }
}
