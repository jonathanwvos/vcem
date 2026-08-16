use eframe::egui::{ColorImage, TextureHandle, Vec2};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub struct ImageFile {
    pub path: PathBuf,
    pub name: String,
}

pub struct DisplayedImage {
    pub image_index: usize,
    pub size: Vec2,
    pub texture: TextureHandle,
    pub bin_indices: Vec<usize>,
    pub bin_count: usize,
    pub highlight_bins: Vec<usize>,
    pub highlight_texture: Option<TextureHandle>,
    pub cellular_automata_texture: Option<TextureHandle>,
    pub cellular_automata_summary: Option<CellularAutomataSummary>,
}

pub struct ProcessedImage {
    pub size: Vec2,
    pub color_image: ColorImage,
    pub histogram: IntensityHistogram,
    pub bin_indices: Vec<usize>,
    pub bin_count: usize,
    pub cellular_automata_overlay: Option<ColorImage>,
    pub cellular_automata_summary: Option<CellularAutomataSummary>,
}

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

#[derive(Clone, Copy, Debug)]
pub struct CellularAutomataSummary {
    pub iterations_run: usize,
    pub converged: bool,
    pub mean_activation: f32,
    pub active_fraction: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenoiseMethod {
    None,
    Median,
    Gaussian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenoiseSettings {
    pub method: DenoiseMethod,
    pub kernel_size: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageRotation {
    #[default]
    None,
    Clockwise90,
    HalfTurn,
    Clockwise270,
}

impl ImageRotation {
    pub fn degrees(self) -> u16 {
        match self {
            Self::None => 0,
            Self::Clockwise90 => 90,
            Self::HalfTurn => 180,
            Self::Clockwise270 => 270,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperimentConfiguration {
    pub schema_version: u32,
    pub histogram_bin_count: usize,
    pub rotation: ImageRotation,
    pub denoise: Option<DenoiseSettings>,
    pub invert: bool,
    pub equalization: Option<f64>,
    pub gamma: Option<f64>,
    pub cellular_automata: Option<CellularAutomataSettings>,
}

impl Default for ExperimentConfiguration {
    fn default() -> Self {
        Self {
            schema_version: 1,
            histogram_bin_count: 256,
            rotation: ImageRotation::None,
            denoise: None,
            invert: false,
            equalization: None,
            gamma: None,
            cellular_automata: None,
        }
    }
}

#[derive(Clone)]
pub struct IntensityHistogram {
    pub bins: Vec<u64>,
    pub min_value: u32,
    pub max_value: u32,
    pub pixel_count: u64,
    pub source_description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn experiment_configuration_round_trips_through_json() {
        let configuration = ExperimentConfiguration {
            schema_version: 1,
            histogram_bin_count: 512,
            rotation: ImageRotation::HalfTurn,
            denoise: Some(DenoiseSettings {
                method: DenoiseMethod::Gaussian,
                kernel_size: 5,
            }),
            invert: true,
            equalization: Some(0.4),
            gamma: Some(1.2),
            cellular_automata: Some(CellularAutomataSettings::default()),
        };

        let json = serde_json::to_string_pretty(&configuration).unwrap();
        let decoded: ExperimentConfiguration = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, configuration);
    }

    #[test]
    fn low_pass_threshold_response_round_trips_through_json() {
        let settings = CellularAutomataSettings {
            response: CellularAutomataResponse::LowThreshold,
            ..CellularAutomataSettings::default()
        };

        let json = serde_json::to_string(&settings).unwrap();
        let decoded: CellularAutomataSettings = serde_json::from_str(&json).unwrap();

        assert!(json.contains("\"response\":\"low_threshold\""));
        assert_eq!(decoded.response, CellularAutomataResponse::LowThreshold);
    }

    #[test]
    fn cellular_automata_settings_default_missing_omni_directional_json() {
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
}
