use std::path::Path;

use image::DynamicImage;

use crate::types::IntensityHistogram;

pub fn intensity_histogram_from_path(
    path: &Path,
    bin_count: usize,
) -> std::result::Result<IntensityHistogram, String> {
    let bin_count = bin_count.clamp(2, 4096);
    let image = image::ImageReader::open(path)
        .map_err(|error| format!("Could not open {}: {}", path.display(), error))?
        .with_guessed_format()
        .map_err(|error| format!("Could not inspect {}: {}", path.display(), error))?
        .decode()
        .map_err(|error| format!("Could not decode {}: {}", path.display(), error))?;

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
        return IntensityHistogram {
            bins,
            min_value: 0,
            max_value: 0,
            pixel_count: 0,
            source_description: source_description.to_owned(),
        };
    }

    IntensityHistogram {
        bins,
        min_value,
        max_value,
        pixel_count,
        source_description: source_description.to_owned(),
    }
}
