use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::discrete_automata::CompactState;

pub const HISTOGRAM_INITIALIZER_VERSION: u32 = 1;

/// A stable intensity interval used by histogram rendering and CA state
/// initialization. `maximum_exclusive` is stored as `u64` so the complete
/// `u32` domain can be represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntensityCalibration {
    pub minimum: u32,
    pub maximum_exclusive: u64,
}

impl IntensityCalibration {
    pub fn new(minimum: u32, maximum_exclusive: u64) -> Result<Self, String> {
        let calibration = Self {
            minimum,
            maximum_exclusive,
        };
        calibration.validate()?;
        Ok(calibration)
    }

    pub fn full_scale(maximum_exclusive: u64) -> Result<Self, String> {
        Self::new(0, maximum_exclusive)
    }

    pub fn validate(self) -> Result<(), String> {
        let largest_exclusive = u32::MAX as u64 + 1;
        if self.maximum_exclusive <= self.minimum as u64
            || self.maximum_exclusive > largest_exclusive
        {
            return Err(format!(
                "Intensity calibration must satisfy minimum < maximum_exclusive <= {largest_exclusive}."
            ));
        }
        Ok(())
    }

    pub fn level_count(self) -> u64 {
        self.maximum_exclusive - self.minimum as u64
    }

    pub fn maximum_value(self) -> u32 {
        (self.maximum_exclusive - 1) as u32
    }
}

/// Versioned description of histogram-driven discrete state initialization.
///
/// With no `bin_to_state` table, each histogram bin is one CA state. A table
/// can instead merge bins into a smaller state space without changing the
/// histogram granularity used to inspect the source image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistogramStateInitializer {
    pub version: u32,
    pub calibration: IntensityCalibration,
    pub bin_count: u32,
    pub state_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_to_state: Option<Vec<u32>>,
}

impl HistogramStateInitializer {
    pub fn identity(calibration: IntensityCalibration, bin_count: u32) -> Self {
        Self {
            version: HISTOGRAM_INITIALIZER_VERSION,
            calibration,
            bin_count,
            state_count: bin_count,
            bin_to_state: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != HISTOGRAM_INITIALIZER_VERSION {
            return Err(format!(
                "Unsupported histogram state initializer version {}.",
                self.version
            ));
        }
        self.calibration.validate()?;
        if self.bin_count < 2 {
            return Err("Histogram state initialization requires at least two bins.".to_owned());
        }
        if self.state_count < 2 {
            return Err("Discrete state initialization requires at least two states.".to_owned());
        }

        match &self.bin_to_state {
            None => {
                if self.state_count != self.bin_count {
                    return Err(
                        "An identity histogram initializer requires state_count to equal bin_count."
                            .to_owned(),
                    );
                }
            }
            Some(mapping) => {
                let expected_len = usize::try_from(self.bin_count).map_err(|_| {
                    "Histogram bin count is not addressable on this platform.".to_owned()
                })?;
                if mapping.len() != expected_len {
                    return Err(format!(
                        "Histogram bin-to-state mapping has {} entries; expected {}.",
                        mapping.len(),
                        self.bin_count
                    ));
                }
                if let Some(invalid) = mapping
                    .iter()
                    .copied()
                    .find(|state| *state >= self.state_count)
                {
                    return Err(format!(
                        "Histogram bin-to-state mapping contains state {invalid}, but state_count is {}.",
                        self.state_count
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn compile(&self) -> Result<CompiledHistogramStateInitializer, String> {
        self.validate()?;
        let bin_to_state = self
            .bin_to_state
            .as_ref()
            .map(|mapping| -> Result<Arc<Vec<u32>>, String> {
                let mut compiled_mapping = Vec::new();
                compiled_mapping
                    .try_reserve_exact(mapping.len())
                    .map_err(|_| "Could not allocate histogram bin-to-state mapping.".to_owned())?;
                compiled_mapping.extend_from_slice(mapping);
                Ok(Arc::new(compiled_mapping))
            })
            .transpose()?;

        Ok(CompiledHistogramStateInitializer {
            calibration: self.calibration,
            bin_count: self.bin_count,
            state_count: self.state_count,
            bin_to_state,
        })
    }
}

/// Validated, allocation-free-per-value histogram quantizer.
#[derive(Clone, Debug)]
pub struct CompiledHistogramStateInitializer {
    calibration: IntensityCalibration,
    bin_count: u32,
    state_count: u32,
    bin_to_state: Option<Arc<Vec<u32>>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct QuantizedIntensities {
    pub bin_indices: Vec<u32>,
    pub counts: Vec<u64>,
    pub observed_range: Option<(u32, u32)>,
    pub calibration: IntensityCalibration,
    pub bin_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedHistogram {
    pub counts: Vec<u64>,
    pub observed_range: Option<(u32, u32)>,
    pub calibration: IntensityCalibration,
    pub bin_count: u32,
    pub pixel_count: u64,
}

impl CompiledHistogramStateInitializer {
    pub fn calibration(&self) -> IntensityCalibration {
        self.calibration
    }

    pub fn bin_count(&self) -> u32 {
        self.bin_count
    }

    pub fn state_count(&self) -> u32 {
        self.state_count
    }

    pub fn bin_index(&self, value: u32) -> Result<u32, String> {
        let minimum = self.calibration.minimum as u64;
        let maximum_exclusive = self.calibration.maximum_exclusive;
        let value = value as u64;
        if value < minimum || value >= maximum_exclusive {
            return Err(format!(
                "Intensity {value} is outside calibration [{minimum}, {maximum_exclusive})."
            ));
        }

        // u128 keeps the complete u32 intensity and bin-count domains safe.
        let relative = (value - minimum) as u128;
        let scaled = relative * self.bin_count as u128 / self.calibration.level_count() as u128;
        Ok((scaled as u32).min(self.bin_count - 1))
    }

    pub fn state(&self, value: u32) -> Result<u32, String> {
        let bin = self.bin_index(value)?;
        Ok(self
            .bin_to_state
            .as_ref()
            .map_or(bin, |mapping| mapping[bin as usize]))
    }

    pub fn states(&self, values: &[u32]) -> Result<Vec<u32>, String> {
        let mut states = Vec::new();
        states
            .try_reserve_exact(values.len())
            .map_err(|_| "Could not allocate discrete initial states.".to_owned())?;
        for value in values {
            states.push(self.state(*value)?);
        }
        Ok(states)
    }

    pub fn states_from_bins(&self, bins: &[u32]) -> Result<Vec<u32>, String> {
        let mut states = Vec::new();
        states
            .try_reserve_exact(bins.len())
            .map_err(|_| "Could not allocate discrete initial states.".to_owned())?;
        for bin in bins {
            if *bin >= self.bin_count {
                return Err(format!(
                    "Histogram bin {bin} is outside bin_count {}.",
                    self.bin_count
                ));
            }
            states.push(
                self.bin_to_state
                    .as_ref()
                    .map_or(*bin, |mapping| mapping[*bin as usize]),
            );
        }
        Ok(states)
    }

    pub fn compact_states(
        &self,
        values: &[u32],
        max_packed_state_bytes: u64,
    ) -> Result<CompactState, String> {
        let states = self.states(values)?;
        CompactState::pack_with_budget(states, self.state_count, max_packed_state_bytes)
            .map_err(|error| error.to_string())
    }

    pub fn compact_states_from_bins(
        &self,
        bins: &[u32],
        max_packed_state_bytes: u64,
    ) -> Result<CompactState, String> {
        let mut owned_bins = Vec::new();
        owned_bins
            .try_reserve_exact(bins.len())
            .map_err(|_| "Could not allocate discrete initial states.".to_owned())?;
        owned_bins.extend_from_slice(bins);
        self.compact_states_from_owned_bins(owned_bins, max_packed_state_bytes)
    }

    /// Converts an owned histogram-bin field into the CA state space. Taking
    /// ownership lets an identity initializer reuse the logical u32 buffer
    /// when u32 is the canonical storage width.
    pub fn compact_states_from_owned_bins(
        &self,
        mut bins: Vec<u32>,
        max_packed_state_bytes: u64,
    ) -> Result<CompactState, String> {
        for (index, bin) in bins.iter_mut().enumerate() {
            if *bin >= self.bin_count {
                return Err(format!(
                    "Histogram bin {bin} at index {index} is outside bin_count {}.",
                    self.bin_count
                ));
            }
            if let Some(mapping) = &self.bin_to_state {
                *bin = mapping[*bin as usize];
            }
        }
        CompactState::pack_with_budget(bins, self.state_count, max_packed_state_bytes)
            .map_err(|error| error.to_string())
    }

    pub fn quantize(&self, values: &[u32]) -> Result<QuantizedIntensities, String> {
        let mut counts = self.empty_counts()?;
        let mut bin_indices = Vec::new();
        bin_indices
            .try_reserve_exact(values.len())
            .map_err(|_| "Could not allocate histogram bin identifiers.".to_owned())?;
        let mut minimum = u32::MAX;
        let mut maximum = u32::MIN;

        for value in values {
            let bin = self.bin_index(*value)?;
            counts[bin as usize] = counts[bin as usize]
                .checked_add(1)
                .ok_or_else(|| "Histogram bin count overflowed u64.".to_owned())?;
            bin_indices.push(bin);
            minimum = minimum.min(*value);
            maximum = maximum.max(*value);
        }

        Ok(QuantizedIntensities {
            bin_indices,
            counts,
            observed_range: (!values.is_empty()).then_some((minimum, maximum)),
            calibration: self.calibration,
            bin_count: self.bin_count,
        })
    }

    pub fn histogram(&self, values: &[u32]) -> Result<QuantizedHistogram, String> {
        let mut counts = self.empty_counts()?;
        let mut minimum = u32::MAX;
        let mut maximum = u32::MIN;

        for value in values {
            let bin = self.bin_index(*value)?;
            counts[bin as usize] = counts[bin as usize]
                .checked_add(1)
                .ok_or_else(|| "Histogram bin count overflowed u64.".to_owned())?;
            minimum = minimum.min(*value);
            maximum = maximum.max(*value);
        }

        Ok(QuantizedHistogram {
            counts,
            observed_range: (!values.is_empty()).then_some((minimum, maximum)),
            calibration: self.calibration,
            bin_count: self.bin_count,
            pixel_count: u64::try_from(values.len())
                .map_err(|_| "Histogram pixel count cannot be represented as u64.".to_owned())?,
        })
    }

    fn empty_counts(&self) -> Result<Vec<u64>, String> {
        let count_len = usize::try_from(self.bin_count)
            .map_err(|_| "Histogram bin count is not addressable on this platform.".to_owned())?;
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(count_len)
            .map_err(|_| "Could not allocate histogram counts.".to_owned())?;
        counts.resize(count_len, 0_u64);
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bin_boundaries_match_the_histogram_formula() {
        let quantizer =
            HistogramStateInitializer::identity(IntensityCalibration::full_scale(256).unwrap(), 4)
                .compile()
                .unwrap();

        let values = [0, 63, 64, 127, 128, 191, 192, 255];
        let bins = values
            .into_iter()
            .map(|value| quantizer.bin_index(value).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(bins, [0, 0, 1, 1, 2, 2, 3, 3]);
        assert_eq!(quantizer.states(&values).unwrap(), bins);
    }

    #[test]
    fn full_scale_quantization_matches_the_previous_formula_exhaustively() {
        for level_count in [256_u64, 65_536] {
            for bin_count in [2_u32, 3, 255, 256, 4096] {
                let quantizer = HistogramStateInitializer::identity(
                    IntensityCalibration::full_scale(level_count).unwrap(),
                    bin_count,
                )
                .compile()
                .unwrap();
                for value in 0..level_count as u32 {
                    let previous = (value as u64 * bin_count as u64 / level_count)
                        .min(bin_count as u64 - 1) as u32;
                    assert_eq!(quantizer.bin_index(value).unwrap(), previous);
                }
            }
        }
    }

    #[test]
    fn calibration_can_start_above_zero() {
        let quantizer =
            HistogramStateInitializer::identity(IntensityCalibration::new(100, 200).unwrap(), 4)
                .compile()
                .unwrap();

        assert_eq!(quantizer.bin_index(100).unwrap(), 0);
        assert_eq!(quantizer.bin_index(124).unwrap(), 0);
        assert_eq!(quantizer.bin_index(125).unwrap(), 1);
        assert_eq!(quantizer.bin_index(199).unwrap(), 3);
        assert!(quantizer.bin_index(0).is_err());
        assert!(quantizer.bin_index(500).is_err());
    }

    #[test]
    fn reports_out_of_calibration_values() {
        let initializer =
            HistogramStateInitializer::identity(IntensityCalibration::new(10, 20).unwrap(), 2);
        let quantizer = initializer.compile().unwrap();

        assert!(quantizer.bin_index(9).unwrap_err().contains("outside"));
        assert!(quantizer.bin_index(20).unwrap_err().contains("outside"));
    }

    #[test]
    fn bin_mapping_can_merge_histogram_bins_into_states() {
        let initializer = HistogramStateInitializer {
            version: HISTOGRAM_INITIALIZER_VERSION,
            calibration: IntensityCalibration::full_scale(256).unwrap(),
            bin_count: 8,
            state_count: 3,
            bin_to_state: Some(vec![0, 0, 0, 1, 1, 2, 2, 2]),
        };
        let quantizer = initializer.compile().unwrap();

        assert_eq!(quantizer.state(0).unwrap(), 0);
        assert_eq!(quantizer.state(96).unwrap(), 1);
        assert_eq!(quantizer.state(255).unwrap(), 2);
        assert_eq!(
            quantizer.states_from_bins(&[0, 2, 3, 7]).unwrap(),
            [0, 0, 1, 2]
        );
        let compact = quantizer
            .compact_states_from_owned_bins(vec![0, 2, 3, 7], 4)
            .unwrap();
        assert_eq!(compact.as_u8(), Some([0, 0, 1, 2].as_slice()));
    }

    #[test]
    fn quantization_counts_are_the_exact_fold_of_stored_bin_ids() {
        let quantizer =
            HistogramStateInitializer::identity(IntensityCalibration::full_scale(16).unwrap(), 4)
                .compile()
                .unwrap();
        let quantized = quantizer.quantize(&[0, 1, 4, 7, 8, 15]).unwrap();

        assert_eq!(quantized.bin_indices, [0, 0, 1, 1, 2, 3]);
        assert_eq!(quantized.counts, [2, 2, 1, 1]);
        assert_eq!(quantized.observed_range, Some((0, 15)));
        assert_eq!(
            quantized.counts.iter().sum::<u64>(),
            quantized.bin_indices.len() as u64
        );
    }

    #[test]
    fn inherited_calibration_commutes_with_roi_cropping() {
        let quantizer =
            HistogramStateInitializer::identity(IntensityCalibration::full_scale(256).unwrap(), 8)
                .compile()
                .unwrap();
        let parent_values = [0, 32, 64, 96, 128, 160, 192, 255];
        let parent = quantizer.quantize(&parent_values).unwrap();
        let roi_values = &parent_values[2..6];
        let roi = quantizer.quantize(roi_values).unwrap();

        assert_eq!(roi.bin_indices, parent.bin_indices[2..6]);
        assert_eq!(roi.calibration, parent.calibration);
    }

    #[test]
    fn counts_only_histogram_matches_full_quantization() {
        let quantizer =
            HistogramStateInitializer::identity(IntensityCalibration::full_scale(256).unwrap(), 16)
                .compile()
                .unwrap();
        let values = [0, 1, 15, 16, 127, 128, 254, 255];

        let full = quantizer.quantize(&values).unwrap();
        let histogram = quantizer.histogram(&values).unwrap();

        assert_eq!(histogram.counts, full.counts);
        assert_eq!(histogram.observed_range, full.observed_range);
        assert_eq!(histogram.pixel_count, values.len() as u64);
    }

    #[test]
    fn invalid_calibration_and_mapping_are_rejected() {
        assert!(IntensityCalibration::new(5, 5).is_err());
        assert!(IntensityCalibration::new(0, u32::MAX as u64 + 2).is_err());

        let initializer = HistogramStateInitializer {
            version: HISTOGRAM_INITIALIZER_VERSION,
            calibration: IntensityCalibration::full_scale(256).unwrap(),
            bin_count: 4,
            state_count: 2,
            bin_to_state: Some(vec![0, 1, 2, 0]),
        };
        assert!(
            initializer
                .compile()
                .unwrap_err()
                .contains("contains state 2")
        );
    }

    #[test]
    fn complete_u32_domain_does_not_overflow() {
        let quantizer = HistogramStateInitializer::identity(
            IntensityCalibration::new(0, u32::MAX as u64 + 1).unwrap(),
            u32::MAX,
        )
        .compile()
        .unwrap();

        assert_eq!(quantizer.bin_index(0).unwrap(), 0);
        assert_eq!(quantizer.bin_index(u32::MAX).unwrap(), u32::MAX - 1);
    }

    #[test]
    fn initializer_json_round_trip_preserves_calibration() {
        let initializer = HistogramStateInitializer {
            version: HISTOGRAM_INITIALIZER_VERSION,
            calibration: IntensityCalibration::new(12, 4012).unwrap(),
            bin_count: 4,
            state_count: 2,
            bin_to_state: Some(vec![0, 0, 1, 1]),
        };

        let json = serde_json::to_string(&initializer).unwrap();
        let decoded: HistogramStateInitializer = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, initializer);
    }
}
