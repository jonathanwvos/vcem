use std::{fs::File, io::BufReader, path::Path};

use eframe::egui::{ColorImage, Vec2};
use image::{DynamicImage, imageops::FilterType};
use tiff::{
    ColorType,
    decoder::{Decoder, DecodingResult},
};

use crate::automata::{CellularAutomataResult, simulate_legacy};
use crate::quantization::{
    CompiledHistogramStateInitializer, HistogramStateInitializer, IntensityCalibration,
    QuantizedHistogram, QuantizedIntensities,
};
use crate::types::{
    CellularAutomataSettings, DenoiseMethod, DenoiseSettings, ExperimentConfiguration,
    ImageRotation, IntensityHistogram, ProcessedImage,
};

const MAX_PREVIEW_PIXELS: u64 = 1_048_576;

pub fn intensity_histogram_from_path(
    path: &Path,
    bin_count: usize,
) -> std::result::Result<IntensityHistogram, String> {
    let plane = decode_gray_plane(path)?;
    let quantizer = compile_normal_histogram_quantizer(plane.calibration, bin_count)?;
    let histogram = quantizer.histogram(&plane.values)?;
    Ok(histogram_from_counts(histogram, &plane.source_description))
}

pub fn gamma_adjusted_image_from_path(
    path: &Path,
    gamma: f64,
    bin_count: usize,
) -> std::result::Result<ProcessedImage, String> {
    let configuration = ExperimentConfiguration {
        histogram_bin_count: bin_count,
        gamma: Some(gamma),
        ..ExperimentConfiguration::default()
    };
    processed_image_from_path(path, &configuration)
}

pub fn processed_image_from_path(
    path: &Path,
    configuration: &ExperimentConfiguration,
) -> std::result::Result<ProcessedImage, String> {
    let mut plane = decode_gray_plane(path)?;
    if plane.calibration.minimum != 0 {
        return Err(
            "The current image-processing pipeline requires zero-based encoded intensities."
                .to_owned(),
        );
    }
    let mut description = plane.source_description.clone();

    if configuration.rotation != ImageRotation::None {
        rotate_gray_plane(&mut plane, configuration.rotation);
        description = format!(
            "{} + rotated {}",
            description,
            configuration.rotation.degrees()
        );
    }

    if let Some(settings) = configuration
        .denoise
        .filter(|settings| settings.method != DenoiseMethod::None)
    {
        apply_denoise(&mut plane.values, plane.width, plane.height, settings)?;
        description = format!("{} + {}", description, settings.label());
    }

    if configuration.invert {
        invert_values(&mut plane.values, plane.calibration.maximum_value());
        description = format!("{} + inverted values", description);
    }

    if let Some(strength) = configuration.equalization {
        let strength = strength.clamp(0.0, 1.0);
        if strength > 0.0 {
            equalize_histogram_values(
                &mut plane.values,
                plane.calibration.maximum_value(),
                plane.calibration.level_count(),
                strength,
            );
            description = format!("{} + equalized {:.0}%", description, strength * 100.0);
        }
    }

    if let Some(gamma) = configuration.gamma {
        let gamma = gamma.clamp(0.001, 4.0);
        apply_gamma_values(&mut plane.values, plane.calibration.maximum_value(), gamma);
        description = format!("{} + gamma {:.3}", description, gamma);
    }

    let cellular_automata = configuration
        .cellular_automata
        .as_ref()
        .map(|settings| {
            cellular_automata_overlay(&plane.values, plane.width, plane.height, settings)
        })
        .transpose()?;

    // Keep the per-pixel bin buffer out of the legacy CA's peak working set.
    // New discrete models will consume these exact identifiers as generation
    // zero instead of allocating a second quantized image.
    let quantizer =
        compile_normal_histogram_quantizer(plane.calibration, configuration.histogram_bin_count)?;
    let quantized = quantizer.quantize(&plane.values)?;

    process_values(
        &plane.values,
        &plane.alpha,
        [plane.width, plane.height],
        plane.calibration.maximum_value(),
        &description,
        quantized,
        cellular_automata,
    )
}

struct GrayPlane {
    width: usize,
    height: usize,
    values: Vec<u32>,
    alpha: Vec<u8>,
    calibration: IntensityCalibration,
    source_description: String,
}

fn rotate_gray_plane(plane: &mut GrayPlane, rotation: ImageRotation) {
    if rotation == ImageRotation::None || plane.values.is_empty() {
        return;
    }

    let source_width = plane.width;
    let source_height = plane.height;
    let (target_width, target_height) = match rotation {
        ImageRotation::None | ImageRotation::HalfTurn => (source_width, source_height),
        ImageRotation::Clockwise90 | ImageRotation::Clockwise270 => (source_height, source_width),
    };
    let mut values = vec![0; plane.values.len()];
    let mut alpha = vec![0; plane.alpha.len()];

    for source_y in 0..source_height {
        for source_x in 0..source_width {
            let (target_x, target_y) = match rotation {
                ImageRotation::None => (source_x, source_y),
                ImageRotation::Clockwise90 => (source_height - 1 - source_y, source_x),
                ImageRotation::HalfTurn => {
                    (source_width - 1 - source_x, source_height - 1 - source_y)
                }
                ImageRotation::Clockwise270 => (source_y, source_width - 1 - source_x),
            };
            let source_index = source_y * source_width + source_x;
            let target_index = target_y * target_width + target_x;
            values[target_index] = plane.values[source_index];
            alpha[target_index] = plane.alpha[source_index];
        }
    }

    plane.width = target_width;
    plane.height = target_height;
    plane.values = values;
    plane.alpha = alpha;
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
                calibration: IntensityCalibration::full_scale(256)
                    .expect("8-bit calibration is valid"),
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
                calibration: IntensityCalibration::full_scale(256)
                    .expect("8-bit calibration is valid"),
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
                calibration: IntensityCalibration::full_scale(65_536)
                    .expect("16-bit calibration is valid"),
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
                calibration: IntensityCalibration::full_scale(65_536)
                    .expect("16-bit calibration is valid"),
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
                calibration: IntensityCalibration::full_scale(65_536)
                    .expect("16-bit calibration is valid"),
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
        calibration: IntensityCalibration::full_scale(65_536).expect("16-bit calibration is valid"),
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

fn cellular_automata_overlay(
    values: &[u32],
    width: usize,
    height: usize,
    settings: &CellularAutomataSettings,
) -> std::result::Result<(ColorImage, CellularAutomataResult), String> {
    let result = simulate_legacy(values, width, height, settings)?;
    let mut rgba = Vec::with_capacity(result.state.len() * 4);

    for activation in &result.state {
        let alpha = (activation.sqrt() * 190.0).round().clamp(0.0, 190.0) as u8;
        rgba.extend_from_slice(&[30, 210, 190, alpha]);
    }

    Ok((
        ColorImage::from_rgba_unmultiplied([width, height], &rgba),
        result,
    ))
}

fn compile_normal_histogram_quantizer(
    calibration: IntensityCalibration,
    bin_count: usize,
) -> Result<CompiledHistogramStateInitializer, String> {
    if !(2..=4096).contains(&bin_count) {
        return Err("Histogram bin count must be between 2 and 4096.".to_owned());
    }
    let bin_count = u32::try_from(bin_count)
        .map_err(|_| "Histogram bin count cannot be represented as u32.".to_owned())?;
    HistogramStateInitializer::identity(calibration, bin_count).compile()
}

fn histogram_from_counts(
    histogram: QuantizedHistogram,
    source_description: &str,
) -> IntensityHistogram {
    let (min_value, max_value) = histogram.observed_range.unwrap_or((0, 0));

    IntensityHistogram {
        bins: histogram.counts,
        min_value,
        max_value,
        pixel_count: histogram.pixel_count,
        calibration: histogram.calibration,
        source_description: source_description.to_owned(),
    }
}

fn process_values(
    values: &[u32],
    alpha: &[u8],
    size: [usize; 2],
    display_max: u32,
    source_description: &str,
    quantized: QuantizedIntensities,
    cellular_automata: Option<(ColorImage, CellularAutomataResult)>,
) -> Result<ProcessedImage, String> {
    let pixel_count = size[0]
        .checked_mul(size[1])
        .ok_or_else(|| "Processed image dimensions overflowed usize.".to_owned())?;
    if values.len() != pixel_count
        || alpha.len() != pixel_count
        || quantized.bin_indices.len() != pixel_count
    {
        return Err(
            "Processed image values, alpha, and histogram bins must match its dimensions."
                .to_owned(),
        );
    }

    let rgba_len = pixel_count
        .checked_mul(4)
        .ok_or_else(|| "Processed image RGBA allocation size overflowed usize.".to_owned())?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(rgba_len)
        .map_err(|_| "Could not allocate the processed image RGBA buffer.".to_owned())?;
    for (value, alpha) in values.iter().zip(alpha) {
        let display_value = (*value as u64 * 255 / display_max.max(1) as u64).min(255) as u8;
        rgba.extend_from_slice(&[display_value, display_value, display_value, *alpha]);
    }

    let QuantizedIntensities {
        bin_indices,
        counts,
        observed_range,
        calibration,
        bin_count,
    } = quantized;
    let bin_count = usize::try_from(bin_count)
        .map_err(|_| "Histogram bin count is not addressable on this platform.".to_owned())?;
    let (min_value, max_value) = observed_range.unwrap_or((0, 0));
    let histogram = IntensityHistogram {
        bins: counts,
        min_value,
        max_value,
        pixel_count: bin_indices.len() as u64,
        calibration,
        source_description: source_description.to_owned(),
    };

    let (cellular_automata_overlay, cellular_automata_summary, cellular_automata_result) =
        cellular_automata
            .map(|(overlay, result)| (Some(overlay), Some(result.summary), Some(result)))
            .unwrap_or((None, None, None));

    Ok(ProcessedImage {
        size: Vec2::new(size[0] as f32, size[1] as f32),
        color_image: ColorImage::from_rgba_unmultiplied(size, &rgba),
        histogram,
        bin_indices,
        bin_count,
        cellular_automata_overlay,
        cellular_automata_summary,
        cellular_automata_result,
    })
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
    fn processed_image_retains_numeric_cellular_automata_result() {
        let values = vec![0, 32, 64, 96, 128, 160, 192, 224, 255];
        let (overlay, result) =
            cellular_automata_overlay(&values, 3, 3, &CellularAutomataSettings::default()).unwrap();
        let expected_state = result.state.clone();
        let expected_summary = result.summary;

        let quantizer =
            compile_normal_histogram_quantizer(IntensityCalibration::full_scale(256).unwrap(), 256)
                .unwrap();
        let quantized = quantizer.quantize(&values).unwrap();
        let processed = process_values(
            &values,
            &[255; 9],
            [3, 3],
            255,
            "test",
            quantized,
            Some((overlay, result)),
        )
        .unwrap();

        assert_eq!(processed.cellular_automata_summary, Some(expected_summary));
        let retained = processed.cellular_automata_result.unwrap();
        assert_eq!(retained.size, [3, 3]);
        assert_eq!(retained.state, expected_state);
        assert_eq!(retained.summary, expected_summary);
    }

    #[test]
    fn rotates_gray_plane_without_changing_pixel_values() {
        let make_plane = || GrayPlane {
            width: 3,
            height: 2,
            values: vec![1, 2, 3, 4, 5, 6],
            alpha: vec![11, 12, 13, 14, 15, 16],
            calibration: IntensityCalibration::full_scale(256).unwrap(),
            source_description: "test".to_owned(),
        };

        let mut clockwise = make_plane();
        rotate_gray_plane(&mut clockwise, ImageRotation::Clockwise90);
        assert_eq!((clockwise.width, clockwise.height), (2, 3));
        assert_eq!(clockwise.values, vec![4, 1, 5, 2, 6, 3]);
        assert_eq!(clockwise.alpha, vec![14, 11, 15, 12, 16, 13]);

        let mut half_turn = make_plane();
        rotate_gray_plane(&mut half_turn, ImageRotation::HalfTurn);
        assert_eq!((half_turn.width, half_turn.height), (3, 2));
        assert_eq!(half_turn.values, vec![6, 5, 4, 3, 2, 1]);

        let mut counter_clockwise = make_plane();
        rotate_gray_plane(&mut counter_clockwise, ImageRotation::Clockwise270);
        assert_eq!((counter_clockwise.width, counter_clockwise.height), (2, 3));
        assert_eq!(counter_clockwise.values, vec![3, 6, 2, 5, 1, 4]);
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
        assert_eq!(plane.calibration.level_count(), 65_536);
        assert_eq!(plane.calibration.maximum_value(), 65_535);
        assert!(plane.source_description.contains("32-bit float grayscale"));
    }

    #[test]
    #[ignore = "requires the local RELION float TIFF fixture"]
    fn evolves_cellular_automata_on_relion_float_tiff_fixture() {
        let path = Path::new(
            "data/raw/motion-corrected/AAG-MC_mics/FoilHole_17243830_Data_17242991_17242993_20210817_164930_Fractions.tif",
        );
        if !path.exists() {
            return;
        }

        let plane = decode_gray_plane(path).expect("float TIFF should decode through fallback");
        let result = simulate_legacy(
            &plane.values,
            plane.width,
            plane.height,
            &CellularAutomataSettings::default(),
        )
        .expect("default cellular automata settings should evolve");

        assert_eq!(result.state.len(), plane.width * plane.height);
        assert!(result.summary.iterations_run > 0);
        assert!(
            result.summary.active_fraction > 0.0 && result.summary.active_fraction < 1.0,
            "default feature selection should be non-empty and selective: {:?}",
            result.summary
        );
        assert!(result.state.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}
