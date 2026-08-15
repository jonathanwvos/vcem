use std::path::Path;

use eframe::egui::{ColorImage, Vec2};
use image::DynamicImage;

use crate::types::{IntensityHistogram, ProcessedImage};

pub fn intensity_histogram_from_path(
    path: &Path,
    bin_count: usize,
) -> std::result::Result<IntensityHistogram, String> {
    let bin_count = bin_count.clamp(2, 4096);
    let image = decode_image(path)?;

    match image {
        DynamicImage::ImageLuma8(image) => {
            let values = image.pixels().map(|pixel| pixel.0[0] as u32);
            Ok(histogram_from_values(
                values,
                bin_count,
                256,
                "8-bit grayscale",
            ))
        }
        DynamicImage::ImageLumaA8(image) => {
            let values = image.pixels().map(|pixel| pixel.0[0] as u32);
            Ok(histogram_from_values(
                values,
                bin_count,
                256,
                "8-bit grayscale + alpha",
            ))
        }
        DynamicImage::ImageLuma16(image) => {
            let values = image.pixels().map(|pixel| pixel.0[0] as u32);
            Ok(histogram_from_values(
                values,
                bin_count,
                65_536,
                "16-bit grayscale",
            ))
        }
        DynamicImage::ImageLumaA16(image) => {
            let values = image.pixels().map(|pixel| pixel.0[0] as u32);
            Ok(histogram_from_values(
                values,
                bin_count,
                65_536,
                "16-bit grayscale + alpha",
            ))
        }
        other => {
            let image = other.to_luma16();
            let values = image.pixels().map(|pixel| pixel.0[0] as u32);
            Ok(histogram_from_values(
                values,
                bin_count,
                65_536,
                "converted to 16-bit grayscale",
            ))
        }
    }
}

pub fn gamma_adjusted_image_from_path(
    path: &Path,
    gamma: f64,
    bin_count: usize,
) -> std::result::Result<ProcessedImage, String> {
    let gamma = gamma.clamp(0.001, 2.0);
    let bin_count = bin_count.clamp(2, 4096);
    let image = decode_image(path)?;

    match image {
        DynamicImage::ImageLuma8(image) => {
            let size = [image.width() as usize, image.height() as usize];
            let values = image
                .pixels()
                .map(|pixel| (apply_gamma(pixel.0[0] as u32, 255, gamma), 255));
            Ok(process_adjusted_values(
                values,
                size,
                bin_count,
                256,
                255,
                "gamma-adjusted 8-bit grayscale",
            ))
        }
        DynamicImage::ImageLumaA8(image) => {
            let size = [image.width() as usize, image.height() as usize];
            let values = image
                .pixels()
                .map(|pixel| (apply_gamma(pixel.0[0] as u32, 255, gamma), pixel.0[1]));
            Ok(process_adjusted_values(
                values,
                size,
                bin_count,
                256,
                255,
                "gamma-adjusted 8-bit grayscale + alpha",
            ))
        }
        DynamicImage::ImageLuma16(image) => {
            let size = [image.width() as usize, image.height() as usize];
            let values = image
                .pixels()
                .map(|pixel| (apply_gamma(pixel.0[0] as u32, 65_535, gamma), 255));
            Ok(process_adjusted_values(
                values,
                size,
                bin_count,
                65_536,
                65_535,
                "gamma-adjusted 16-bit grayscale",
            ))
        }
        DynamicImage::ImageLumaA16(image) => {
            let size = [image.width() as usize, image.height() as usize];
            let values = image.pixels().map(|pixel| {
                (
                    apply_gamma(pixel.0[0] as u32, 65_535, gamma),
                    (pixel.0[1] >> 8) as u8,
                )
            });
            Ok(process_adjusted_values(
                values,
                size,
                bin_count,
                65_536,
                65_535,
                "gamma-adjusted 16-bit grayscale + alpha",
            ))
        }
        other => {
            let image = other.to_luma16();
            let size = [image.width() as usize, image.height() as usize];
            let values = image
                .pixels()
                .map(|pixel| (apply_gamma(pixel.0[0] as u32, 65_535, gamma), 255));
            Ok(process_adjusted_values(
                values,
                size,
                bin_count,
                65_536,
                65_535,
                "gamma-adjusted image converted to 16-bit grayscale",
            ))
        }
    }
}

fn decode_image(path: &Path) -> std::result::Result<DynamicImage, String> {
    image::ImageReader::open(path)
        .map_err(|error| format!("Could not open {}: {}", path.display(), error))?
        .with_guessed_format()
        .map_err(|error| format!("Could not inspect {}: {}", path.display(), error))?
        .decode()
        .map_err(|error| format!("Could not decode {}: {}", path.display(), error))
}

fn apply_gamma(value: u32, max_value: u32, gamma: f64) -> u32 {
    let normalized = value as f64 / max_value as f64;
    (normalized.powf(gamma) * max_value as f64).round() as u32
}

fn histogram_from_values(
    values: impl Iterator<Item = u32>,
    bin_count: usize,
    value_range: u64,
    source_description: &str,
) -> IntensityHistogram {
    let mut min_value = u32::MAX;
    let mut max_value = u32::MIN;
    let mut pixel_count = 0;
    let mut bins = vec![0; bin_count];

    for value in values {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
        pixel_count += 1;

        let bin = (value as u64 * bin_count as u64 / value_range).min(bin_count as u64 - 1);
        bins[bin as usize] += 1;
    }

    if pixel_count == 0 {
        min_value = 0;
        max_value = 0;
    }

    IntensityHistogram {
        bins,
        min_value,
        max_value,
        pixel_count,
        source_description: source_description.to_owned(),
    }
}

fn process_adjusted_values(
    values: impl Iterator<Item = (u32, u8)>,
    size: [usize; 2],
    bin_count: usize,
    value_range: u64,
    display_max: u32,
    source_description: &str,
) -> ProcessedImage {
    let mut min_value = u32::MAX;
    let mut max_value = u32::MIN;
    let mut pixel_count = 0;
    let mut bins = vec![0; bin_count];
    let mut rgba = Vec::with_capacity(size[0] * size[1] * 4);

    for (value, alpha) in values {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
        pixel_count += 1;

        let bin = (value as u64 * bin_count as u64 / value_range).min(bin_count as u64 - 1);
        bins[bin as usize] += 1;

        let display_value = (value * 255 / display_max.max(1)) as u8;
        rgba.extend_from_slice(&[display_value, display_value, display_value, alpha]);
    }

    if pixel_count == 0 {
        min_value = 0;
        max_value = 0;
    }

    ProcessedImage {
        size: Vec2::new(size[0] as f32, size[1] as f32),
        color_image: ColorImage::from_rgba_unmultiplied(size, &rgba),
        histogram: IntensityHistogram {
            bins,
            min_value,
            max_value,
            pixel_count,
            source_description: source_description.to_owned(),
        },
    }
}
