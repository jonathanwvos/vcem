//! Region-guided, one-generation discrete CA proposals. ROI membership is a
//! weak example label; contrast scores are not segmentation accuracy.
use crate::{
    automata_lab::DiscreteLabSettings,
    discrete_automata::{
        AggregateReducer, BoundaryPolicy, DiscreteRuleKind, KernelOffset, SamplePolarity,
        aggregate_sum_at,
    },
    lab_configuration::random_kernel,
    quantization::{HistogramStateInitializer, IntensityCalibration},
    types::CA_MASK_RGB,
};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct CandidateRequest {
    pub values: Vec<u32>,
    pub calibration: IntensityCalibration,
    pub size: [usize; 2],
    pub region: [usize; 4],
    pub kernel_size: u32,
    pub state_count: u32,
    pub seed: u64,
    pub current_kernel: Vec<KernelOffset>,
}
#[derive(Clone)]
pub struct DiscreteCandidate {
    pub name: String,
    pub settings: DiscreteLabSettings,
    pub region_coverage: f32,
    pub background_coverage: f32,
    pub score: f32,
    pub preview_size: [usize; 2],
    pub preview_rgba: Vec<u8>,
}

fn inside([x, y]: [usize; 2], [left, top, right, bottom]: [usize; 4]) -> bool {
    x >= left && x < right && y >= top && y < bottom
}
fn samples(bounds: [usize; 4], exclude: Option<[usize; 4]>) -> Vec<[usize; 2]> {
    let [l, t, r, b] = bounds;
    let area = (r - l) * (b - t);
    let stride = area.div_ceil(8192).max(1);
    (0..area)
        .step_by(stride)
        .map(|i| [l + i % (r - l), t + i / (r - l)])
        .filter(|p| exclude.is_none_or(|roi| !inside(*p, roi)))
        .collect()
}
fn cancelled(flag: &AtomicBool) -> Result<(), String> {
    if flag.load(Ordering::Relaxed) {
        Err("Candidate generation cancelled.".into())
    } else {
        Ok(())
    }
}
pub fn generate(
    request: CandidateRequest,
    cancel: &AtomicBool,
) -> Result<Vec<DiscreteCandidate>, String> {
    let CandidateRequest {
        values,
        calibration,
        size,
        region,
        kernel_size,
        state_count,
        seed,
        current_kernel,
    } = request;
    if size[0].checked_mul(size[1]) != Some(values.len())
        || values.is_empty()
        || values.len() > 1_048_576
    {
        return Err("Invalid prepared image size.".into());
    }
    let [l, t, r, b] = region;
    if l >= r || t >= b || r > size[0] || b > size[1] || (r - l) * (b - t) < 4 {
        return Err("Draw a region containing at least four pixels.".into());
    }
    if !(2..=64).contains(&state_count) || !(3..=15).contains(&kernel_size) || kernel_size % 2 == 0
    {
        return Err("Choose 2–64 states and an odd kernel size between 3 and 15.".into());
    }
    cancelled(cancel)?;
    let quantizer = HistogramStateInitializer::identity(calibration, state_count).compile()?;
    let states: Vec<u32> = values
        .iter()
        .map(|v| quantizer.state(*v))
        .collect::<Result<_, _>>()?;
    let padding = (kernel_size as usize * 2)
        .max(((r - l).max(b - t)) / 2)
        .max(8);
    let context = [
        l.saturating_sub(padding),
        t.saturating_sub(padding),
        (r + padding).min(size[0]),
        (b + padding).min(size[1]),
    ];
    let positive = samples(region, None);
    let negative = samples(context, Some(region));
    if positive.len() < 4 || negative.len() < 4 {
        return Err(
            "Select a smaller region with surrounding image content for comparison.".into(),
        );
    }
    let radius = (kernel_size / 2) as i32;
    let mut kernels = Vec::new();
    if !current_kernel.is_empty()
        && current_kernel.len() <= 225
        && current_kernel
            .iter()
            .all(|k| k.dx.abs_diff(0) <= 7 && k.dy.abs_diff(0) <= 7)
    {
        kernels.push(("Current kernel".to_owned(), current_kernel));
    }
    for (name, dx, dy) in [
        ("Horizontal", 1, 0),
        ("Vertical", 0, 1),
        ("Diagonal ↘", 1, 1),
        ("Diagonal ↗", 1, -1),
    ] {
        let mut kernel = Vec::new();
        for a in -radius..=radius {
            kernel.push(KernelOffset::new(a * dx, a * dy));
        }
        // Contrast with samples beside the line, inside the chosen kernel grid.
        for a in -radius..=radius {
            for side in [-1, 1] {
                let x = a * dx - side * dy;
                let y = a * dy + side * dx;
                if x.abs() <= radius && y.abs() <= radius {
                    kernel.push(KernelOffset {
                        dx: x,
                        dy: y,
                        polarity: SamplePolarity::Negative,
                    });
                }
            }
        }
        kernel.sort_by_key(|k| (k.dy, k.dx));
        kernel.dedup_by_key(|k| (k.dx, k.dy));
        kernels.push((name.to_owned(), kernel));
    }
    kernels.push(("Local intensity".into(), vec![KernelOffset::new(0, 0)]));
    kernels.push((
        "Seeded contrast".into(),
        random_kernel(kernel_size, seed, 0.25, 0.5, 25),
    ));
    let mut candidates = Vec::new();
    for (name, kernel) in kernels {
        cancelled(cancel)?;
        let bins = kernel.len() * (state_count as usize - 1) + 1;
        let signature = |position| {
            aggregate_sum_at(
                &states,
                size,
                position,
                state_count,
                &kernel,
                BoundaryPolicy::Clamp,
            )
        };
        let mut pos_counts = vec![0usize; bins];
        let mut neg_counts = vec![0usize; bins];
        let mut pos_indices = Vec::new();
        let mut neg_indices = Vec::new();
        for (i, p) in positive.iter().enumerate() {
            if i % 128 == 0 {
                cancelled(cancel)?;
            }
            let v = signature(*p);
            pos_counts[v] += 1;
            pos_indices.push(v);
        }
        for (i, p) in negative.iter().enumerate() {
            if i % 128 == 0 {
                cancelled(cancel)?;
            }
            let v = signature(*p);
            neg_counts[v] += 1;
            neg_indices.push(v);
        }
        let table: Vec<u32> = (0..bins)
            .map(|i| {
                let pos = pos_counts[i] as f64 / positive.len() as f64;
                let neg = neg_counts[i] as f64 / negative.len() as f64;
                // Require observed support, and a margin over the local background.
                if pos_counts[i] >= 2 && pos > neg * 1.25 + 0.002 {
                    state_count - 1
                } else {
                    0
                }
            })
            .collect();
        let region_coverage =
            pos_indices.iter().filter(|i| table[**i] > 0).count() as f32 / positive.len() as f32;
        let background_coverage =
            neg_indices.iter().filter(|i| table[**i] > 0).count() as f32 / negative.len() as f32;
        if region_coverage <= 0.0 {
            continue;
        }
        let [cl, ct, cr, cb] = context;
        let scale = (128.0 / (cr - cl).max(cb - ct) as f32).min(1.0);
        let preview_size = [
            (((cr - cl) as f32 * scale).round() as usize).max(1),
            (((cb - ct) as f32 * scale).round() as usize).max(1),
        ];
        let mut preview_rgba = Vec::with_capacity(preview_size[0] * preview_size[1] * 4);
        for py in 0..preview_size[1] {
            cancelled(cancel)?;
            for px in 0..preview_size[0] {
                let x = cl + px * (cr - cl) / preview_size[0];
                let y = ct + py * (cb - ct) / preview_size[1];
                let gray = (((values[y * size[0] + x] as u64 - calibration.minimum as u64) as f64
                    / (calibration.level_count() - 1) as f64)
                    * 255.0)
                    .round() as u8;
                let selected = table[signature([x, y])] > 0;
                let mut rgb = [gray; 3];
                if selected {
                    for c in 0..3 {
                        rgb[c] = ((gray as u16 + CA_MASK_RGB[c] as u16 * 2) / 3) as u8;
                    }
                }
                let on_border = inside([x, y], region)
                    && (x - l < ((cr - cl) / preview_size[0]).max(1)
                        || r - x <= ((cr - cl) / preview_size[0]).max(1)
                        || y - t < ((cb - ct) / preview_size[1]).max(1)
                        || b - y <= ((cb - ct) / preview_size[1]).max(1));
                if on_border {
                    rgb = [255, 184, 54];
                }
                preview_rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }
        candidates.push(DiscreteCandidate {
            name,
            settings: DiscreteLabSettings {
                state_count,
                kernel,
                boundary: BoundaryPolicy::Clamp,
                rule_kind: DiscreteRuleKind::Aggregate,
                aggregate_reducer: AggregateReducer::Sum,
                aggregate_table: table,
                omni_directional: false,
                ..Default::default()
            },
            region_coverage,
            background_coverage,
            score: region_coverage - background_coverage,
            preview_size,
            preview_rgba,
        });
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    candidates.truncate(4);
    if candidates.is_empty() {
        return Err("No distinct candidates found. Try a tighter region, another state count, or different image preparation.".into());
    }
    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        automata::AutomataSimulationResult,
        lab_configuration::{LabExperimentConfiguration, LabModelSettings},
        types::ExperimentConfiguration,
    };
    fn request() -> CandidateRequest {
        let size = [32, 24];
        let region = [8, 6, 24, 18];
        CandidateRequest {
            values: (0..size[0] * size[1])
                .map(|i| {
                    if inside([i % size[0], i / size[0]], region) && i % size[0] % 4 < 2 {
                        255
                    } else {
                        0
                    }
                })
                .collect(),
            calibration: IntensityCalibration::full_scale(256).unwrap(),
            size,
            region,
            kernel_size: 5,
            state_count: 4,
            seed: 7,
            current_kernel: Vec::new(),
        }
    }
    #[test]
    fn candidates_reproduce_their_region_masks_in_the_real_engine() {
        let candidates = generate(request(), &AtomicBool::new(false)).unwrap();
        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 4);
        for candidate in &candidates {
            let input = request();
            let experiment = LabExperimentConfiguration {
                schema_version: 1,
                preprocessing: ExperimentConfiguration::default(),
                automaton: LabModelSettings::Discrete(candidate.settings.clone()),
                generations: 1,
            };
            experiment.validate().unwrap();
            let result = experiment
                .run(
                    &input.values,
                    input.calibration,
                    input.size,
                    &AtomicBool::new(false),
                )
                .unwrap();
            let AutomataSimulationResult::DiscreteMapV1(result) = result else {
                panic!()
            };
            let points = samples(input.region, None);
            let coverage = points
                .iter()
                .filter(|p| result.final_state.get(p[1] * input.size[0] + p[0]).unwrap() > 0)
                .count() as f32
                / points.len() as f32;
            assert!((coverage - candidate.region_coverage).abs() < 1e-6);
            assert!(candidate.score > 0.0);
            assert_eq!(
                candidate.preview_rgba.len(),
                candidate.preview_size[0] * candidate.preview_size[1] * 4
            );
        }
        let repeated = generate(request(), &AtomicBool::new(false)).unwrap();
        for (a, b) in candidates.iter().zip(repeated) {
            assert_eq!(a.settings, b.settings);
            assert_eq!(a.preview_rgba, b.preview_rgba);
        }
    }
    #[test]
    fn uniform_images_invalid_regions_and_cancelled_jobs_do_not_offer_false_candidates() {
        let mut input = request();
        input.values.fill(128);
        assert!(generate(input, &AtomicBool::new(false)).is_err());
        let mut input = request();
        input.region = [0, 0, 32, 24];
        assert!(generate(input, &AtomicBool::new(false)).is_err());
        let mut input = request();
        input.region = [0, 0, 1, 1];
        assert!(generate(input, &AtomicBool::new(false)).is_err());
        assert!(generate(request(), &AtomicBool::new(true)).is_err());
    }
}
