use eframe::egui::{ColorImage, TextureHandle, Vec2};
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
}

pub struct ProcessedImage {
    pub size: Vec2,
    pub color_image: ColorImage,
    pub histogram: IntensityHistogram,
    pub bin_indices: Vec<usize>,
    pub bin_count: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethod {
    None,
    Median,
    Gaussian,
}

#[derive(Clone, Copy)]
pub struct DenoiseSettings {
    pub method: DenoiseMethod,
    pub kernel_size: usize,
}

#[derive(Clone)]
pub struct IntensityHistogram {
    pub bins: Vec<u64>,
    pub min_value: u32,
    pub max_value: u32,
    pub pixel_count: u64,
    pub source_description: String,
}
