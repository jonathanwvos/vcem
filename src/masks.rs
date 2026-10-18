//! Image-bound mask recipes. Captured activations stay in memory; recipes can
//! rebuild them from the source image without serializing millions of floats.
use crate::{
    analysis::processed_image_from_path,
    automata::{AutomataSimulationResult, CellularAutomataResult},
    automata_projection::{
        MaskProjectionSpec, NumericProjectionSpec, OUTPUT_PROJECTION_VERSION, ProjectionLimits,
        project_mask,
    },
    types::{DownsamplingMethod, ExperimentConfiguration, ImageRotation, ProcessedImage},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::UNIX_EPOCH,
};

pub const MAX_LAYERS: usize = 16;
pub const MAX_REGIONS: usize = 128;
pub const MAX_RECIPE_BYTES: u64 = 1_048_576;
const MAX_PIXELS: usize = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceStamp {
    pub path: PathBuf,
    bytes: u64,
    modified_seconds: u64,
    modified_nanos: u32,
}
impl SourceStamp {
    pub fn read(path: &Path) -> Result<Self, String> {
        let path = path
            .canonicalize()
            .map_err(|e| format!("Cannot locate source image: {e}"))?;
        let metadata =
            fs::metadata(&path).map_err(|e| format!("Cannot inspect source image: {e}"))?;
        let modified = metadata
            .modified()
            .map_err(|e| e.to_string())?
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            path,
            bytes: metadata.len(),
            modified_seconds: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskFrame {
    pub source: SourceStamp,
    pub size: [usize; 2],
    pub rotation: ImageRotation,
    pub downsampling: DownsamplingMethod,
}
impl MaskFrame {
    pub fn pixel_count(&self) -> Result<usize, String> {
        self.size[0]
            .checked_mul(self.size[1])
            .filter(|n| self.size[0] > 0 && self.size[1] > 0 && *n <= MAX_PIXELS)
            .ok_or_else(|| {
                "Mask dimensions must be nonzero and fit the preview pixel limit.".to_owned()
            })
    }
}

pub fn process_source(
    path: &Path,
    configuration: &ExperimentConfiguration,
) -> Result<(ProcessedImage, MaskFrame), String> {
    let source = SourceStamp::read(path)?;
    let image = processed_image_from_path(path, configuration)?;
    if SourceStamp::read(path)? != source {
        return Err(
            "Source image changed during processing. Reload it before capturing masks.".to_owned(),
        );
    }
    let frame = MaskFrame {
        source,
        size: [image.size.x as usize, image.size.y as usize],
        rotation: configuration.rotation,
        downsampling: configuration.downsampling,
    };
    Ok((image, frame))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskOperation {
    And,
    Or,
    Xor,
    Subtract,
}
impl MaskOperation {
    pub fn label(self) -> &'static str {
        match self {
            Self::And => "AND",
            Self::Or => "OR",
            Self::Xor => "XOR",
            Self::Subtract => "A minus B",
        }
    }
    fn apply(self, a: u8, b: u8) -> u8 {
        match self {
            Self::And => a & b,
            Self::Or => a | b,
            Self::Xor => a ^ b,
            Self::Subtract => a & (1 - b),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LayerSource {
    Ca {
        configuration: ExperimentConfiguration,
        threshold: f32,
    },
    Combine {
        left: u64,
        operation: MaskOperation,
        right: u64,
    },
    Invert {
        input: u64,
    },
}
impl LayerSource {
    fn references(&self, id: u64) -> bool {
        match self {
            Self::Ca { .. } => false,
            Self::Combine { left, right, .. } => *left == id || *right == id,
            Self::Invert { input } => *input == id,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskLayer {
    pub id: u64,
    pub name: String,
    pub source: LayerSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewClass {
    Fibril,
    Contamination,
    Uncertain,
}
impl ReviewClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fibril => "Likely fibrils",
            Self::Contamination => "Suspected contamination",
            Self::Uncertain => "Uncertain / overlap",
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewRegion {
    pub class: ReviewClass,
    /// Normalized coordinates in the document's rotated image, [left, top, right, bottom].
    pub bounds: [f32; 4],
}
impl ReviewRegion {
    pub fn pixel_bounds(&self, size: [usize; 2]) -> [usize; 4] {
        [
            (self.bounds[0] * size[0] as f32).floor() as usize,
            (self.bounds[1] * size[1] as f32).floor() as usize,
            ((self.bounds[2] * size[0] as f32).ceil() as usize).min(size[0]),
            ((self.bounds[3] * size[1] as f32).ceil() as usize).min(size[1]),
        ]
    }
    pub fn coverage(&self, mask: &[u8], size: [usize; 2]) -> f32 {
        let [x0, y0, x1, y1] = self.pixel_bounds(size);
        let total = (x1 - x0) * (y1 - y0);
        if total == 0 {
            return 0.0;
        }
        let count: usize = (y0..y1)
            .map(|y| {
                mask[y * size[0] + x0..y * size[0] + x1]
                    .iter()
                    .map(|v| usize::from(*v))
                    .sum::<usize>()
            })
            .sum();
        count as f32 / total as f32
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskDocument {
    pub schema_version: u32,
    pub frame: MaskFrame,
    pub layers: Vec<MaskLayer>,
    pub regions: Vec<ReviewRegion>,
}
impl MaskDocument {
    pub fn new(frame: MaskFrame) -> Self {
        Self {
            schema_version: 1,
            frame,
            layers: Vec::new(),
            regions: Vec::new(),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("Unsupported mask recipe version.".to_owned());
        }
        self.frame.pixel_count()?;
        if self.layers.len() > MAX_LAYERS || self.regions.len() > MAX_REGIONS {
            return Err("Mask recipe exceeds the layer or review-region limit.".to_owned());
        }
        let mut seen = std::collections::BTreeSet::new();
        for layer in &self.layers {
            if layer.id == 0
                || seen.contains(&layer.id)
                || layer.name.trim().is_empty()
                || layer.name.len() > 120
            {
                return Err("Layers need unique nonzero IDs and names of 1–120 bytes.".to_owned());
            }
            match &layer.source {
                LayerSource::Ca {
                    configuration,
                    threshold,
                } => {
                    validate_ca_configuration(configuration)?;
                    if configuration.rotation != self.frame.rotation
                        || configuration.downsampling != self.frame.downsampling
                    {
                        return Err("Layer coordinates do not match the mask recipe.".to_owned());
                    }
                    if !threshold.is_finite() || !(0.0..=1.0).contains(threshold) {
                        return Err("Mask threshold must be between zero and one.".to_owned());
                    }
                }
                LayerSource::Combine { left, right, .. }
                    if !seen.contains(left) || !seen.contains(right) =>
                {
                    return Err(
                        "Combinations must reference earlier layers in the recipe.".to_owned()
                    );
                }
                LayerSource::Invert { input } if !seen.contains(input) => {
                    return Err("Inversion must reference an earlier layer.".to_owned());
                }
                _ => {}
            }
            seen.insert(layer.id);
        }
        for region in &self.regions {
            let b = region.bounds;
            if b.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
                || b[0] >= b[2]
                || b[1] >= b[3]
            {
                return Err(
                    "Review regions must be nonempty rectangles within the image.".to_owned(),
                );
            }
        }
        Ok(())
    }
    pub fn retarget(&mut self, target: &MaskFrame) -> Result<(), String> {
        if self.frame.source != target.source {
            return Err(
                "This recipe belongs to another source image or the file has changed.".to_owned(),
            );
        }
        let old_turns = self.frame.rotation.degrees() / 90;
        let new_turns = target.rotation.degrees() / 90;
        for region in &mut self.regions {
            for _ in 0..(new_turns + 4 - old_turns) % 4 {
                let [l, t, r, b] = region.bounds;
                region.bounds = [1.0 - b, l, 1.0 - t, r];
            }
        }
        self.frame = target.clone();
        for layer in &mut self.layers {
            if let LayerSource::Ca { configuration, .. } = &mut layer.source {
                configuration.rotation = target.rotation;
                configuration.downsampling = target.downsampling;
            }
        }
        self.validate()
    }
}

fn validate_ca_configuration(configuration: &ExperimentConfiguration) -> Result<(), String> {
    crate::types::validate_experiment_configuration(configuration)?;
    if configuration.cellular_automata.is_none() {
        return Err("Captured layer is missing its CA configuration.".to_owned());
    }
    Ok(())
}

pub struct MaskSession {
    pub document: MaskDocument,
    pub results: BTreeMap<u64, AutomataSimulationResult>,
}
impl MaskSession {
    pub fn activation_bytes(&self) -> usize {
        self.results
            .values()
            .map(|result| match result {
                AutomataSimulationResult::LegacyEvidenceGrowthV1(result) => {
                    result.state.len() * std::mem::size_of::<f32>()
                }
                _ => 0,
            })
            .sum()
    }

    pub fn new(frame: MaskFrame) -> Self {
        Self {
            document: MaskDocument::new(frame),
            results: BTreeMap::new(),
        }
    }
    pub fn add_layer(&mut self, name: String, source: LayerSource) -> Result<u64, String> {
        let id = self
            .document
            .layers
            .iter()
            .map(|l| l.id)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or("Layer ID limit reached.")?;
        self.document.layers.push(MaskLayer {
            id,
            name: name.trim().to_owned(),
            source,
        });
        if let Err(e) = self.document.validate() {
            self.document.layers.pop();
            return Err(e);
        }
        Ok(id)
    }
    pub fn capture(
        &mut self,
        frame: &MaskFrame,
        name: String,
        configuration: ExperimentConfiguration,
        result: CellularAutomataResult,
    ) -> Result<u64, String> {
        if frame != &self.document.frame || result.size != frame.size {
            return Err(
                "Rebuild layers for the current image coordinates before capturing another layer."
                    .to_owned(),
            );
        }
        let result = AutomataSimulationResult::LegacyEvidenceGrowthV1(result);
        crate::automata_projection::validate_simulation_result(&result)
            .map_err(|e| e.to_string())?;
        let id = self.add_layer(
            name,
            LayerSource::Ca {
                configuration,
                threshold: 0.1,
            },
        )?;
        self.results.insert(id, result);
        Ok(id)
    }
    pub fn remove(&mut self, id: u64) -> Result<(), String> {
        if self.document.layers.iter().any(|l| l.source.references(id)) {
            return Err("Remove combinations that reference this layer first.".to_owned());
        }
        self.document.layers.retain(|l| l.id != id);
        self.results.remove(&id);
        Ok(())
    }
    pub fn evaluate(&self) -> Result<BTreeMap<u64, Vec<u8>>, String> {
        self.document.validate()?;
        let mut masks: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        for layer in &self.document.layers {
            let mask = match &layer.source {
                LayerSource::Ca { threshold, .. } => {
                    let result = self
                        .results
                        .get(&layer.id)
                        .ok_or("Load the recipe with Rebuild layers to generate its masks.")?;
                    let mask = project_mask(
                        result,
                        &MaskProjectionSpec {
                            version: OUTPUT_PROJECTION_VERSION,
                            field: NumericProjectionSpec::normalized_final_state(),
                            minimum: *threshold,
                            maximum: 1.0,
                        },
                        ProjectionLimits::default(),
                    )
                    .map_err(|e| e.to_string())?;
                    if mask.size != self.document.frame.size {
                        return Err("Captured mask dimensions do not match the image.".to_owned());
                    }
                    mask.values
                }
                LayerSource::Combine {
                    left,
                    operation,
                    right,
                } => masks[left]
                    .iter()
                    .zip(&masks[right])
                    .map(|(a, b)| operation.apply(*a, *b))
                    .collect(),
                LayerSource::Invert { input } => masks[input].iter().map(|v| 1 - v).collect(),
            };
            masks.insert(layer.id, mask);
        }
        Ok(masks)
    }
}

/// All-or-nothing rebuild; callers install the new session only after success.
pub fn rebuild(
    document: MaskDocument,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<MaskSession, String> {
    document.validate()?;
    if SourceStamp::read(&document.frame.source.path)? != document.frame.source {
        return Err("Source image has changed since this recipe was captured.".to_owned());
    }
    let mut results = BTreeMap::new();
    for layer in &document.layers {
        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("Mask rebuild cancelled.".to_owned());
        }
        if let LayerSource::Ca { configuration, .. } = &layer.source {
            let (image, frame) = process_source(&document.frame.source.path, configuration)?;
            if frame != document.frame {
                return Err("Rebuilt mask coordinates differ from the recipe.".to_owned());
            }
            let result = image
                .cellular_automata_result
                .ok_or("CA did not produce a result.")?;
            results.insert(
                layer.id,
                AutomataSimulationResult::LegacyEvidenceGrowthV1(result),
            );
        }
    }
    Ok(MaskSession { document, results })
}

pub fn overlap_rgba(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter()
        .zip(b)
        .flat_map(|(a, b)| match (*a != 0, *b != 0) {
            (true, false) => [30, 210, 190, 170],
            (false, true) => [220, 90, 190, 170],
            (true, true) => [255, 185, 50, 210],
            _ => [0, 0, 0, 0],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::{CellularAutomataSettings, CellularAutomataSummary};
    fn frame() -> MaskFrame {
        MaskFrame {
            source: SourceStamp {
                path: PathBuf::from("/fixture.tif"),
                bytes: 10,
                modified_seconds: 1,
                modified_nanos: 0,
            },
            size: [4, 1],
            rotation: ImageRotation::None,
            downsampling: DownsamplingMethod::AreaAverage,
        }
    }
    fn configuration() -> ExperimentConfiguration {
        ExperimentConfiguration {
            cellular_automata: Some(CellularAutomataSettings::default()),
            ..Default::default()
        }
    }
    fn result(values: &[f32]) -> CellularAutomataResult {
        CellularAutomataResult {
            size: [values.len(), 1],
            state: values.to_vec(),
            summary: CellularAutomataSummary {
                iterations_run: 1,
                converged: false,
                mean_activation: 0.5,
                active_fraction: 0.5,
            },
        }
    }
    fn pair() -> (MaskSession, u64, u64) {
        let frame = frame();
        let mut session = MaskSession::new(frame.clone());
        let a = session
            .capture(
                &frame,
                "Fibrils".to_owned(),
                configuration(),
                result(&[0.0, 1.0, 0.0, 1.0]),
            )
            .unwrap();
        let b = session
            .capture(
                &frame,
                "Contamination".to_owned(),
                configuration(),
                result(&[0.0, 0.0, 1.0, 1.0]),
            )
            .unwrap();
        (session, a, b)
    }
    #[test]
    fn boolean_operations_cover_the_complete_truth_table() {
        let (mut session, a, b) = pair();
        for (operation, expected) in [
            (MaskOperation::And, vec![0, 0, 0, 1]),
            (MaskOperation::Or, vec![0, 1, 1, 1]),
            (MaskOperation::Xor, vec![0, 1, 1, 0]),
            (MaskOperation::Subtract, vec![0, 1, 0, 0]),
        ] {
            let id = session
                .add_layer(
                    operation.label().to_owned(),
                    LayerSource::Combine {
                        left: a,
                        operation,
                        right: b,
                    },
                )
                .unwrap();
            assert_eq!(session.evaluate().unwrap()[&id], expected);
        }
        let id = session
            .add_layer(
                "NOT contamination".to_owned(),
                LayerSource::Invert { input: b },
            )
            .unwrap();
        assert_eq!(session.evaluate().unwrap()[&id], [1, 1, 0, 0]);
        let reverse = session
            .add_layer(
                "Reverse subtraction".to_owned(),
                LayerSource::Combine {
                    left: b,
                    operation: MaskOperation::Subtract,
                    right: a,
                },
            )
            .unwrap();
        assert_eq!(session.evaluate().unwrap()[&reverse], [0, 0, 1, 0]);
    }
    #[test]
    fn threshold_changes_update_nested_combinations_without_rerunning_ca() {
        let (mut session, a, b) = pair();
        let c = session
            .capture(
                &frame(),
                "Blobs".to_owned(),
                configuration(),
                result(&[0.0, 0.4, 0.0, 0.0]),
            )
            .unwrap();
        let exclusion = session
            .add_layer(
                "Exclusion".to_owned(),
                LayerSource::Combine {
                    left: b,
                    operation: MaskOperation::Or,
                    right: c,
                },
            )
            .unwrap();
        let clean = session
            .add_layer(
                "Candidate fibrils".to_owned(),
                LayerSource::Combine {
                    left: a,
                    operation: MaskOperation::Subtract,
                    right: exclusion,
                },
            )
            .unwrap();
        assert_eq!(session.evaluate().unwrap()[&clean], [0, 0, 0, 0]);
        let LayerSource::Ca { threshold, .. } = &mut session.document.layers[2].source else {
            panic!()
        };
        *threshold = 0.5;
        assert_eq!(session.evaluate().unwrap()[&clean], [0, 1, 0, 0]);
        let AutomataSimulationResult::LegacyEvidenceGrowthV1(captured) = &session.results[&c]
        else {
            panic!()
        };
        assert_eq!(captured.state, [0.0, 0.4, 0.0, 0.0]);
    }
    #[test]
    fn threshold_boundaries_are_inclusive_and_zero_can_select_every_pixel() {
        let mut session = MaskSession::new(frame());
        let id = session
            .capture(
                &frame(),
                "Threshold test".to_owned(),
                configuration(),
                result(&[0.0, 0.1, 0.5, 1.0]),
            )
            .unwrap();
        assert_eq!(session.evaluate().unwrap()[&id], [0, 1, 1, 1]);
        for (threshold_value, expected) in [(0.0, vec![1, 1, 1, 1]), (1.0, vec![0, 0, 0, 1])] {
            let LayerSource::Ca { threshold, .. } = &mut session.document.layers[0].source else {
                panic!()
            };
            *threshold = threshold_value;
            assert_eq!(session.evaluate().unwrap()[&id], expected);
        }
    }
    #[test]
    fn rejects_misalignment_invalid_inputs_and_dangling_references_atomically() {
        let (mut session, a, b) = pair();
        let mut other = frame();
        other.rotation = ImageRotation::HalfTurn;
        assert!(
            session
                .capture(
                    &other,
                    "Wrong view".to_owned(),
                    configuration(),
                    result(&[0.0; 4])
                )
                .is_err()
        );
        other = frame();
        other.source.path = PathBuf::from("/another-image.tif");
        assert!(session.document.clone().retarget(&other).is_err());
        assert!(
            session
                .capture(
                    &frame(),
                    "NaN".to_owned(),
                    configuration(),
                    result(&[f32::NAN; 4])
                )
                .is_err()
        );
        assert!(
            session
                .add_layer("Cycle".to_owned(), LayerSource::Invert { input: 3 })
                .is_err()
        );
        assert!(
            session
                .add_layer(
                    "Missing".to_owned(),
                    LayerSource::Combine {
                        left: a,
                        operation: MaskOperation::And,
                        right: 99
                    }
                )
                .is_err()
        );
        assert_eq!(session.document.layers.len(), 2);
        let combination = session
            .add_layer(
                "A and B".to_owned(),
                LayerSource::Combine {
                    left: a,
                    operation: MaskOperation::And,
                    right: b,
                },
            )
            .unwrap();
        assert!(session.remove(a).is_err());
        session.remove(combination).unwrap();
        session.remove(a).unwrap();
        assert_eq!(session.document.layers.len(), 1);
        assert_eq!(session.evaluate().unwrap()[&b], [0, 0, 1, 1]);
    }
    #[test]
    fn recipes_round_trip_and_regions_follow_rotation() {
        let (mut session, a, b) = pair();
        session
            .add_layer(
                "A minus B".to_owned(),
                LayerSource::Combine {
                    left: a,
                    operation: MaskOperation::Subtract,
                    right: b,
                },
            )
            .unwrap();
        session.document.regions.push(ReviewRegion {
            class: ReviewClass::Fibril,
            bounds: [0.1, 0.2, 0.4, 0.7],
        });
        let json = serde_json::to_string(&session.document).unwrap();
        assert!(!json.contains("\"state\""));
        let mut decoded: MaskDocument = serde_json::from_str(&json).unwrap();
        decoded.validate().unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
        let mut rotated = frame();
        rotated.rotation = ImageRotation::Clockwise90;
        rotated.size = [1, 4];
        rotated.downsampling = DownsamplingMethod::Nearest;
        decoded.retarget(&rotated).unwrap();
        for (actual, expected) in decoded.regions[0].bounds.iter().zip([0.3, 0.1, 0.8, 0.4]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let LayerSource::Ca { configuration, .. } = &decoded.layers[0].source else {
            panic!()
        };
        assert_eq!(configuration.rotation, ImageRotation::Clockwise90);
        assert_eq!(configuration.downsampling, DownsamplingMethod::Nearest);
        decoded.retarget(&frame()).unwrap();
        for (a, b) in decoded.regions[0]
            .bounds
            .iter()
            .zip(session.document.regions[0].bounds)
        {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn recipe_validation_rejects_bad_geometry_thresholds_and_versions() {
        let (session, _, _) = pair();
        let mut bad = session.document.clone();
        bad.schema_version = 2;
        assert!(bad.validate().is_err());
        let mut bad = session.document.clone();
        bad.frame.size = [usize::MAX, 2];
        assert!(bad.validate().is_err());
        let mut bad = session.document.clone();
        bad.layers[1].id = bad.layers[0].id;
        assert!(bad.validate().is_err());
        let mut bad = session.document.clone();
        let LayerSource::Ca { threshold, .. } = &mut bad.layers[0].source else {
            panic!()
        };
        *threshold = f32::NAN;
        assert!(bad.validate().is_err());
        let mut bad = session.document.clone();
        bad.regions.push(ReviewRegion {
            class: ReviewClass::Uncertain,
            bounds: [0.5, 0.0, 0.3, 1.0],
        });
        assert!(bad.validate().is_err());
        let mut bad = session.document.clone();
        bad.layers[0].source = LayerSource::Invert { input: 2 };
        assert!(bad.validate().is_err());
        let mut bad = session.document;
        let LayerSource::Ca { configuration, .. } = &mut bad.layers[0].source else {
            panic!()
        };
        configuration
            .cellular_automata
            .as_mut()
            .unwrap()
            .kernel_size = 999;
        assert!(bad.validate().is_err());
    }
    #[test]
    fn overlap_colors_and_region_coverage_are_unambiguous() {
        assert_eq!(
            overlap_rgba(&[0, 1, 0, 1], &[0, 0, 1, 1]),
            [
                0, 0, 0, 0, 30, 210, 190, 170, 220, 90, 190, 170, 255, 185, 50, 210
            ]
        );
        let region = ReviewRegion {
            class: ReviewClass::Fibril,
            bounds: [0.0, 0.0, 0.5, 1.0],
        };
        assert_eq!(region.pixel_bounds([4, 2]), [0, 0, 2, 2]);
        assert_eq!(region.coverage(&[1, 1, 0, 0, 1, 0, 0, 0], [4, 2]), 0.75);
    }

    #[test]
    #[ignore = "requires the local RELION float TIFF fixture"]
    fn masks_on_real_micrograph_keep_numeric_results_and_coordinates() {
        let path = Path::new(
            "data/raw/motion-corrected/AAG-MC_mics/FoilHole_17243852_Data_17242991_17242993_20210817_174309_Fractions.tif",
        );
        assert!(path.exists(), "local micrograph fixture is missing");
        let configuration = ExperimentConfiguration {
            cellular_automata: Some(CellularAutomataSettings {
                kernel_size: 3,
                kernel: vec![1.0; 9],
                iterations: 2,
                ..Default::default()
            }),
            ..Default::default()
        };
        let (image, frame) = process_source(path, &configuration).unwrap();
        let mut session = MaskSession::new(frame.clone());
        let a = session
            .capture(
                &frame,
                "Candidate A".to_owned(),
                configuration.clone(),
                image.cellular_automata_result.unwrap(),
            )
            .unwrap();
        let mut other = configuration;
        other.gamma = Some(1.2);
        other.cellular_automata.as_mut().unwrap().response_center = 0.4;
        let (image, other_frame) = process_source(path, &other).unwrap();
        assert_eq!(frame, other_frame);
        let b = session
            .capture(
                &frame,
                "Candidate B".to_owned(),
                other,
                image.cellular_automata_result.unwrap(),
            )
            .unwrap();
        let difference = session
            .add_layer(
                "A without B".to_owned(),
                LayerSource::Combine {
                    left: a,
                    operation: MaskOperation::Subtract,
                    right: b,
                },
            )
            .unwrap();
        let masks = session.evaluate().unwrap();
        assert_eq!(masks[&difference].len(), 1024 * 1024);
        for ((a, b), selected) in masks[&a].iter().zip(&masks[&b]).zip(&masks[&difference]) {
            assert_eq!(*selected != 0, *a != 0 && *b == 0);
        }
        assert_eq!(overlap_rgba(&masks[&a], &masks[&b]).len(), 4 * 1024 * 1024);
    }
}
