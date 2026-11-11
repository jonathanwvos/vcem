use image::{ImageBuffer, Pixel, Primitive, imageops::FilterType};

use crate::types::DownsamplingMethod;

/// Integer channels used by the decoded grayscale and grayscale-alpha images.
pub(crate) trait Sample: Primitive + Into<f64> {
    fn from_average(value: f64) -> Self;
}

impl Sample for u8 {
    fn from_average(value: f64) -> Self {
        value.round().clamp(0.0, u8::MAX as f64) as u8
    }
}

impl Sample for u16 {
    fn from_average(value: f64) -> Self {
        value.round().clamp(0.0, u16::MAX as f64) as u16
    }
}

/// Resample decoded channels before augmentations, histograms, or CA processing.
/// The caller resolves legacy mode according to the decoder that succeeded.
pub(crate) fn resize_preview<P>(
    source: &ImageBuffer<P, Vec<P::Subpixel>>,
    width: u32,
    height: u32,
    method: DownsamplingMethod,
) -> ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
    P::Subpixel: Sample,
{
    if source.dimensions() == (width, height) {
        return source.clone();
    }
    match method {
        DownsamplingMethod::Legacy => unreachable!("resolve legacy mode before resizing"),
        DownsamplingMethod::Triangle => {
            image::imageops::resize(source, width, height, FilterType::Triangle)
        }
        DownsamplingMethod::Nearest => ImageBuffer::from_fn(width, height, |x, y| {
            // Preserve the original float TIFF floor-coordinate convention.
            let sx = (u64::from(x) * u64::from(source.width()) / u64::from(width)) as u32;
            let sy = (u64::from(y) * u64::from(source.height()) / u64::from(height)) as u32;
            *source.get_pixel(sx, sy)
        }),
        DownsamplingMethod::AreaAverage => {
            let scale_x = f64::from(source.width()) / f64::from(width);
            let scale_y = f64::from(source.height()) / f64::from(height);
            ImageBuffer::from_fn(width, height, |x, y| {
                let left = f64::from(x) * scale_x;
                let right = f64::from(x + 1) * scale_x;
                let top = f64::from(y) * scale_y;
                let bottom = f64::from(y + 1) * scale_y;
                let mut pixel = *source.get_pixel(left.floor() as u32, top.floor() as u32);
                for (channel, output) in pixel.channels_mut().iter_mut().enumerate() {
                    let mut sum = 0.0;
                    let mut area = 0.0;
                    for sy in top.floor() as u32..(bottom.ceil() as u32).min(source.height()) {
                        let wy = bottom.min(f64::from(sy + 1)) - top.max(f64::from(sy));
                        for sx in left.floor() as u32..(right.ceil() as u32).min(source.width()) {
                            let wx = right.min(f64::from(sx + 1)) - left.max(f64::from(sx));
                            let weight = wx * wy;
                            sum += source.get_pixel(sx, sy).channels()[channel].into() * weight;
                            area += weight;
                        }
                    }
                    *output = P::Subpixel::from_average(sum / area);
                }
                pixel
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, LumaA};

    #[test]
    fn nearest_preserves_legacy_floor_samples() {
        let source = ImageBuffer::from_fn(5, 3, |x, y| Luma([(y * 5 + x) as u16]));
        assert_eq!(
            resize_preview(&source, 3, 2, DownsamplingMethod::Nearest).into_raw(),
            [0, 1, 3, 5, 6, 8]
        );
    }

    #[test]
    fn area_averages_checkerboard_instead_of_aliasing() {
        let source = ImageBuffer::from_fn(8, 8, |x, y| Luma([((x + y) % 2 * 255) as u8]));
        assert_eq!(
            resize_preview(&source, 2, 2, DownsamplingMethod::AreaAverage).into_raw(),
            [128; 4]
        );
        assert_eq!(
            resize_preview(&source, 2, 2, DownsamplingMethod::Nearest).into_raw(),
            [0; 4]
        );
    }

    #[test]
    fn area_weights_fractional_footprints_and_alpha_at_boundaries() {
        let source =
            ImageBuffer::<LumaA<u16>, _>::from_raw(3, 1, vec![0, 600, 30, 300, 90, 0]).unwrap();
        let output = resize_preview(&source, 2, 1, DownsamplingMethod::AreaAverage);
        assert_eq!(output.into_raw(), [10, 500, 70, 100]);
        let vertical = ImageBuffer::<Luma<u8>, _>::from_raw(1, 3, vec![0, 30, 90]).unwrap();
        assert_eq!(
            resize_preview(&vertical, 1, 2, DownsamplingMethod::AreaAverage).into_raw(),
            [10, 70]
        );
    }

    #[test]
    fn area_preserves_constants_and_full_intensity_range() {
        for value in [0, 12345, u16::MAX] {
            let source = ImageBuffer::from_pixel(7, 5, Luma([value]));
            assert_eq!(
                resize_preview(&source, 3, 2, DownsamplingMethod::AreaAverage).into_raw(),
                vec![value; 6]
            );
        }
    }

    #[test]
    fn triangle_matches_existing_library_resampler() {
        let source = ImageBuffer::from_fn(7, 5, |x, y| LumaA([(x * y * 13) as u8, (x * 30) as u8]));
        assert_eq!(
            resize_preview(&source, 3, 2, DownsamplingMethod::Triangle),
            image::imageops::resize(&source, 3, 2, FilterType::Triangle)
        );
    }

    #[test]
    fn unchanged_dimensions_preserve_every_channel() {
        let source =
            ImageBuffer::from_fn(3, 5, |x, y| LumaA([(x * y * 13) as u16, (x * 3000) as u16]));
        for method in [
            DownsamplingMethod::Nearest,
            DownsamplingMethod::AreaAverage,
            DownsamplingMethod::Triangle,
        ] {
            assert_eq!(resize_preview(&source, 3, 5, method), source);
        }
    }
}
