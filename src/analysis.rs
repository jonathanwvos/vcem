use std::{fs::File, io::BufReader, path::Path};

use eframe::egui::{ColorImage, Vec2};
use image::{DynamicImage, imageops::FilterType};
use tiff::{
    ColorType,
    decoder::{Decoder, DecodingResult},
};

use crate::types::{DenoiseMethod, DenoiseSettings, IntensityHistogram, ProcessedImage};

const MAX_PREVIEW_PIXELS: u64 = 1_048_576;

pub fn intensity_histogram_from_path(
    path: &Path,
    bin_count: usize,
) -> std::result::Result<IntensityHistogram, String> {
    let plane = decode_gray_plane(path)?;
    Ok(histogram_from_values(
        plane.values.iter().copied(),
        bin_count.clamp(2, 4096),
        plane.value_range,
        &plane.source_description,
    ))
}

pub fn gamma_adjusted_image_from_path(
    path: &Path,
    gamma: f64,
    bin_count: usize,
) -> std::result::Result<ProcessedImage, String> {
    processed_image_from_path(path, None, false, None, Some(gamma), bin_count)
}

pub fn processed_image_from_path(
    path: &Path,
    denoise: Option<DenoiseSettings>,
    invert: bool,
    equalization: Option<f64>,
    gamma: Option<f64>,
    bin_count: usize,
) -> std::result::Result<ProcessedImage, String> {
    let mut plane = decode_gray_plane(path)?;
    let mut description = plane.source_description.clone();

    if let Some(settings) = denoise.filter(|settings| settings.method != DenoiseMethod::None) {
        apply_denoise(&mut plane.values, plane.width, plane.height, settings)?;
        description = format!("{} + {}", description, settings.label());
    }

    if invert {
        invert_values(&mut plane.values, plane.max_value);
        description = format!("{} + inverted values", description);
    }

    if let Some(strength) = equalization {
        let strength = strength.clamp(0.0, 1.0);
        if strength > 0.0 {
            equalize_histogram_values(
                &mut plane.values,
                plane.max_value,
                plane.value_range,
                strength,
            );
            description = format!("{} + equalized {:.0}%", description, strength * 100.0);
        }
    }

    if let Some(gamma) = gamma {
        let gamma = gamma.clamp(0.001, 2.0);
        apply_gamma_values(&mut plane.values, plane.max_value, gamma);
        description = format!("{} + gamma {:.3}", description, gamma);
    }

    Ok(process_values(
        plane
            .values
            .iter()
            .copied()
            .zip(plane.alpha.iter().copied()),
        [plane.width, plane.height],
        bin_count.clamp(2, 4096),
        plane.value_range,
        plane.max_value,
        &description,
    ))
}

struct GrayPlane {
    width: usize,
    height: usize,
    values: Vec<u32>,
    alpha: Vec<u8>,
    value_range: u64,
    max_value: u32,
    source_description: String,
}

fn decode_gray_plane(path: &Path) -> std::result::Result<GrayPlane, String> {
    match decode_gray_plane_with_image(path) {
        Ok(plane) => Ok(plane),
        Err(image_error) if is_tiff_path(path) => decode_float_tiff_gray_plane(path)
            .map_err(|tiff_error| format!("{image_error}; TIFF fallback failed: {tiff_error}")),
        Err(image_error) => Err(image_error),
    }
}

fn decode_gray_plane_with_image(path: &Path) -> std::result::Result<GrayPlane, String> {
    let image = image::ImageReader::open(path)
        .map_err(|error| format!("Could not open {}: {}", path.display(), error))?
        .with_guessed_format()
        .map_err(|error| format!("Could not inspect {}: {}", path.display(), error))?
        .decode()
        .map_err(|error| format!("Could not decode {}: {}", path.display(), error))?;

    Ok(match image {
        DynamicImage::ImageLuma8(image) => {
            let source_width = image.width();
            let source_height = image.height();
            let (preview_width, preview_height, resized) =
                preview_dimensions(source_width, source_height);
            let image = if resized {
                image::imageops::resize(&image, preview_width, preview_height, FilterType::Triangle)
            } else {
                image
            };
            let width = image.width() as usize;
            let height = image.height() as usize;
            GrayPlane {
                width,
                height,
                values: image.pixels().map(|pixel| pixel.0[0] as u32).collect(),
                alpha: vec![255; width * height],
                value_range: 256,
                max_value: 255,
                source_description: preview_description(
                    "8-bit grayscale",
                    source_width,
                    source_height,
                    image.width(),
                    image.height(),
                ),
            }
        }
        DynamicImage::ImageLumaA8(image) => {
            let source_width = image.width();
            let source_height = image.height();
            let (preview_width, preview_height, resized) =
                preview_dimensions(source_width, source_height);
            let image = if resized {
                image::imageops::resize(&image, preview_width, preview_height, FilterType::Triangle)
            } else {
                image
            };
            let width = image.width() as usize;
            let height = image.height() as usize;
            let mut values = Vec::with_capacity(width * height);
            let mut alpha = Vec::with_capacity(width * height);
            for pixel in image.pixels() {
                values.push(pixel.0[0] as u32);
                alpha.push(pixel.0[1]);
            }
            GrayPlane {
                width,
                height,
                values,
                alpha,
                value_range: 256,
                max_value: 255,
                source_description: preview_description(
                    "8-bit grayscale + alpha",
                    source_width,
                    source_height,
                    image.width(),
                    image.height(),
                ),
            }
        }
        DynamicImage::ImageLuma16(image) => {
            let source_width = image.width();
            let source_height = image.height();
            let (preview_width, preview_height, resized) =
                preview_dimensions(source_width, source_height);
            let image = if resized {
                image::imageops::resize(&image, preview_width, preview_height, FilterType::Triangle)
            } else {
                image
            };
            let width = image.width() as usize;
            let height = image.height() as usize;
            GrayPlane {
                width,
                height,
                values: image.pixels().map(|pixel| pixel.0[0] as u32).collect(),
                alpha: vec![255; width * height],
                value_range: 65_536,
                max_value: 65_535,
                source_description: preview_description(
                    "16-bit grayscale",
                    source_width,
                    source_height,
                    image.width(),
                    image.height(),
                ),
            }
        }
        DynamicImage::ImageLumaA16(image) => {
            let source_width = image.width();
            let source_height = image.height();
            let (preview_width, preview_height, resized) =
                preview_dimensions(source_width, source_height);
            let image = if resized {
                image::imageops::resize(&image, preview_width, preview_height, FilterType::Triangle)
            } else {
                image
            };
            let width = image.width() as usize;
            let height = image.height() as usize;
            let mut values = Vec::with_capacity(width * height);
            let mut alpha = Vec::with_capacity(width * height);
            for pixel in image.pixels() {
                values.push(pixel.0[0] as u32);
                alpha.push((pixel.0[1] >> 8) as u8);
            }
            GrayPlane {
                width,
                height,
                values,
                alpha,
                value_range: 65_536,
                max_value: 65_535,
                source_description: preview_description(
                    "16-bit grayscale + alpha",
                    source_width,
                    source_height,
                    image.width(),
                    image.height(),
                ),
            }
        }
        other => {
            let image = other.to_luma16();
            let source_width = image.width();
            let source_height = image.height();
            let (preview_width, preview_height, resized) =
                preview_dimensions(source_width, source_height);
            let image = if resized {
                image::imageops::resize(&image, preview_width, preview_height, FilterType::Triangle)
            } else {
                image
            };
            let width = image.width() as usize;
            let height = image.height() as usize;
            GrayPlane {
                width,
                height,
                values: image.pixels().map(|pixel| pixel.0[0] as u32).collect(),
                alpha: vec![255; width * height],
                value_range: 65_536,
                max_value: 65_535,
                source_description: preview_description(
                    "converted to 16-bit grayscale",
                    source_width,
                    source_height,
                    image.width(),
                    image.height(),
                ),
            }
        }
    })
}

fn decode_float_tiff_gray_plane(path: &Path) -> std::result::Result<GrayPlane, String> {
    let file = File::open(path)
        .map_err(|error| format!("Could not open {}: {}", path.display(), error))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .map_err(|error| format!("Could not inspect {}: {}", path.display(), error))?;
    let color_type = decoder
        .colortype()
        .map_err(|error| format!("Could not inspect TIFF color type: {error}"))?;
    let (source_width, source_height) = decoder
        .dimensions()
        .map_err(|error| format!("Could not inspect TIFF dimensions: {error}"))?;
    let image = decoder
        .read_image()
        .map_err(|error| format!("Could not decode TIFF image: {error}"))?;

    match (color_type, image) {
        (ColorType::Gray(32), DecodingResult::F32(values)) => float_values_to_gray_plane(
            values,
            source_width,
            source_height,
            "32-bit float grayscale",
        ),
        (ColorType::Gray(64), DecodingResult::F64(values)) => float_values_to_gray_plane(
            values.into_iter().map(|value| value as f32).collect(),
            source_width,
            source_height,
            "64-bit float grayscale",
        ),
        (color_type, image) => Err(format!(
            "Unsupported TIFF fallback type: color {color_type:?}, data {}",
            decoding_result_label(&image)
        )),
    }
}

fn float_values_to_gray_plane(
    values: Vec<f32>,
    source_width: u32,
    source_height: u32,
    base_description: &str,
) -> std::result::Result<GrayPlane, String> {
    let expected_len = source_width as usize * source_height as usize;
    if values.len() != expected_len {
        return Err(format!(
            "TIFF pixel count mismatch: expected {expected_len}, decoded {}",
            values.len()
        ));
    }

    let mut finite_min = f32::INFINITY;
    let mut finite_max = f32::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        finite_min = finite_min.min(value);
        finite_max = finite_max.max(value);
    }

    if !finite_min.is_finite() || !finite_max.is_finite() {
        return Err("TIFF float image contains no finite values".to_owned());
    }

    let scale = if finite_max > finite_min {
        65_535.0 / (finite_max - finite_min)
    } else {
        1.0
    };
    let mut normalized_values = values
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                ((value - finite_min) * scale).round().clamp(0.0, 65_535.0) as u32
            } else {
                0
            }
        })
        .collect::<Vec<_>>();

    let (preview_width, preview_height, resized) = preview_dimensions(source_width, source_height);
    if resized {
        normalized_values = resize_gray_nearest(
            &normalized_values,
            source_width as usize,
            source_height as usize,
            preview_width as usize,
            preview_height as usize,
        );
    }

    let width = preview_width as usize;
    let height = preview_height as usize;
    Ok(GrayPlane {
        width,
        height,
        values: normalized_values,
        alpha: vec![255; width * height],
        value_range: 65_536,
        max_value: 65_535,
        source_description: format!(
            "{} normalized {:.3}..{:.3}",
            preview_description(
                base_description,
                source_width,
                source_height,
                preview_width,
                preview_height,
            ),
            finite_min,
            finite_max
        ),
    })
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32, bool) {
    let pixel_count = width as u64 * height as u64;
    if pixel_count <= MAX_PREVIEW_PIXELS {
        return (width, height, false);
    }

    let scale = (MAX_PREVIEW_PIXELS as f64 / pixel_count as f64).sqrt();
    let preview_width = ((width as f64 * scale).floor() as u32).max(1);
    let preview_height = ((height as f64 * scale).floor() as u32).max(1);

    (preview_width, preview_height, true)
}

fn preview_description(
    base_description: &str,
    source_width: u32,
    source_height: u32,
    preview_width: u32,
    preview_height: u32,
) -> String {
    if source_width == preview_width && source_height == preview_height {
        base_description.to_owned()
    } else {
        format!(
            "{base_description} preview {preview_width} x {preview_height} from {source_width} x {source_height}"
        )
    }
}

fn resize_gray_nearest(
    values: &[u32],
    source_width: usize,
    source_height: usize,
    preview_width: usize,
    preview_height: usize,
) -> Vec<u32> {
    let mut output = Vec::with_capacity(preview_width * preview_height);
    for y in 0..preview_height {
        let source_y = (y * source_height / preview_height).min(source_height - 1);
        for x in 0..preview_width {
            let source_x = (x * source_width / preview_width).min(source_width - 1);
            output.push(values[source_y * source_width + source_x]);
        }
    }
    output
}

fn is_tiff_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| matches!(extension.to_lowercase().as_str(), "tif" | "tiff"))
}

fn decoding_result_label(result: &DecodingResult) -> &'static str {
    match result {
        DecodingResult::U8(_) => "u8",
        DecodingResult::U16(_) => "u16",
        DecodingResult::U32(_) => "u32",
        DecodingResult::U64(_) => "u64",
        DecodingResult::F16(_) => "f16",
        DecodingResult::F32(_) => "f32",
        DecodingResult::F64(_) => "f64",
        DecodingResult::I8(_) => "i8",
        DecodingResult::I16(_) => "i16",
        DecodingResult::I32(_) => "i32",
        DecodingResult::I64(_) => "i64",
    }
}

fn apply_denoise(
    values: &mut Vec<u32>,
    width: usize,
    height: usize,
    settings: DenoiseSettings,
) -> std::result::Result<(), String> {
    match settings.method {
        DenoiseMethod::None => Ok(()),
        DenoiseMethod::Median => {
            *values = median_filter(values, width, height, settings.kernel_size);
            Ok(())
        }
        DenoiseMethod::Gaussian => {
            *values = gaussian_filter(values, width, height, settings.kernel_size);
            Ok(())
        }
    }
}

fn median_filter(values: &[u32], width: usize, height: usize, kernel_size: usize) -> Vec<u32> {
    if values.is_empty() || width == 0 || height == 0 {
        return values.to_vec();
    }

    let radius = kernel_radius(kernel_size);
    let mut output = vec![0; values.len()];
    let mut window = Vec::with_capacity(kernel_size * kernel_size);

    for y in 0..height {
        for x in 0..width {
            window.clear();
            for yy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                    window.push(values[yy * width + xx]);
                }
            }
            window.sort_unstable();
            output[y * width + x] = window[window.len() / 2];
        }
    }

    output
}

fn gaussian_filter(values: &[u32], width: usize, height: usize, kernel_size: usize) -> Vec<u32> {
    if values.is_empty() || width == 0 || height == 0 {
        return values.to_vec();
    }

    let kernel_size = normalized_kernel_size(kernel_size);
    let radius = kernel_radius(kernel_size);
    let kernel = gaussian_1d_kernel(kernel_size);
    let mut horizontal = vec![0.0; values.len()];
    let mut output = vec![0; values.len()];

    for y in 0..height {
        for x in 0..width {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (kx, weight) in kernel.iter().enumerate() {
                let sample_x = x as isize + kx as isize - radius as isize;
                if !(0..width as isize).contains(&sample_x) {
                    continue;
                }

                weighted_sum += values[y * width + sample_x as usize] as f64 * weight;
                weight_total += weight;
            }

            horizontal[y * width + x] = weighted_sum / weight_total.max(f64::EPSILON);
        }
    }

    for y in 0..height {
        for x in 0..width {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for (ky, weight) in kernel.iter().enumerate() {
                let sample_y = y as isize + ky as isize - radius as isize;
                if !(0..height as isize).contains(&sample_y) {
                    continue;
                }

                weighted_sum += horizontal[sample_y as usize * width + x] * weight;
                weight_total += weight;
            }

            output[y * width + x] = (weighted_sum / weight_total.max(f64::EPSILON)).round() as u32;
        }
    }

    output
}

fn kernel_radius(kernel_size: usize) -> usize {
    normalized_kernel_size(kernel_size) / 2
}

fn normalized_kernel_size(kernel_size: usize) -> usize {
    match kernel_size {
        5 => 5,
        7 => 7,
        _ => 3,
    }
}

fn gaussian_1d_kernel(kernel_size: usize) -> Vec<f64> {
    let kernel_size = normalized_kernel_size(kernel_size);
    let sigma = kernel_size as f64 / 3.0;
    let radius = kernel_size / 2;
    let mut kernel = Vec::with_capacity(kernel_size);

    for x in 0..kernel_size {
        let dx = x as f64 - radius as f64;
        kernel.push((-(dx * dx) / (2.0 * sigma * sigma)).exp());
    }

    kernel
}

fn apply_gamma_values(values: &mut [u32], max_value: u32, gamma: f64) {
    for value in values {
        let normalized = *value as f64 / max_value as f64;
        *value = (normalized.powf(gamma) * max_value as f64).round() as u32;
    }
}

fn invert_values(values: &mut [u32], max_value: u32) {
    for value in values {
        *value = max_value.saturating_sub(*value);
    }
}

fn equalize_histogram_values(values: &mut [u32], max_value: u32, value_range: u64, strength: f64) {
    if values.is_empty() || max_value == 0 || strength <= 0.0 {
        return;
    }

    let histogram_len = value_range.min(max_value as u64 + 1).max(1) as usize;
    let mut histogram = vec![0_u64; histogram_len];
    for value in values.iter().copied() {
        let index = (value as usize).min(histogram_len - 1);
        histogram[index] += 1;
    }

    let Some(cdf_min) = histogram.iter().copied().find(|count| *count > 0) else {
        return;
    };
    let total = values.len() as u64;
    if total <= cdf_min {
        return;
    }

    let denominator = (total - cdf_min) as f64;
    let mut cumulative = 0_u64;
    let mut lookup = Vec::with_capacity(histogram_len);
    for count in histogram {
        cumulative += count;
        let equalized = ((cumulative.saturating_sub(cdf_min)) as f64 / denominator
            * max_value as f64)
            .round()
            .clamp(0.0, max_value as f64) as u32;
        lookup.push(equalized);
    }

    let original_weight = 1.0 - strength;
    for value in values {
        let equalized = lookup[(*value as usize).min(lookup.len() - 1)];
        *value = (*value as f64 * original_weight + equalized as f64 * strength)
            .round()
            .clamp(0.0, max_value as f64) as u32;
    }
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

fn process_values(
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
    let mut bin_indices = Vec::with_capacity(size[0] * size[1]);
    let mut rgba = Vec::with_capacity(size[0] * size[1] * 4);

    for (value, alpha) in values {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
        pixel_count += 1;

        let bin = (value as u64 * bin_count as u64 / value_range).min(bin_count as u64 - 1);
        let bin = bin as usize;
        bins[bin] += 1;
        bin_indices.push(bin);

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
        bin_indices,
        bin_count,
    }
}

impl DenoiseSettings {
    fn label(self) -> &'static str {
        match self.method {
            DenoiseMethod::None => "no denoising",
            DenoiseMethod::Median => "median denoise",
            DenoiseMethod::Gaussian => "gaussian denoise",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equalizes_values_across_available_range() {
        let mut values = vec![10, 10, 20, 30];

        equalize_histogram_values(&mut values, 65_535, 65_536, 1.0);

        assert_eq!(values, vec![0, 0, 32_768, 65_535]);
    }

    #[test]
    fn blends_equalized_values_with_original_values() {
        let mut values = vec![10, 10, 20, 30];

        equalize_histogram_values(&mut values, 65_535, 65_536, 0.5);

        assert_eq!(values, vec![5, 5, 16_394, 32_783]);
    }

    #[test]
    #[ignore = "requires the local RELION float TIFF fixture"]
    fn decodes_relion_float_tiff_fixture() {
        let path = Path::new(
            "data/raw/motion-corrected/AAG-MC_mics/FoilHole_17243830_Data_17242991_17242993_20210817_164930_Fractions.tif",
        );
        if !path.exists() {
            return;
        }

        let plane = decode_gray_plane(path).expect("float TIFF should decode through fallback");
        assert_eq!((plane.width, plane.height), (1024, 1024));
        assert_eq!(plane.value_range, 65_536);
        assert_eq!(plane.max_value, 65_535);
        assert!(plane.source_description.contains("32-bit float grayscale"));
    }
}
