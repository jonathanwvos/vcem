use super::*;
use crate::lab_configuration::{MAX_LAB_CONFIG_BYTES, random_kernel, randomize_rule};

impl VCEMApp {
    pub(super) fn poll_lab_replay(&mut self, ctx: &egui::Context) {
        if self.automata_lab_job.is_none() && self.processing_job.is_none() {
            if let Some(cycles) = self.lab_replay_generations.take() {
                self.step_automata_lab(ctx, cycles);
            }
        }
    }

    pub(super) fn lab_draft_model(&self) -> LabModelSettings {
        match self.automata_lab_model_family {
            LabModelFamily::Discrete => {
                LabModelSettings::Discrete(self.automata_lab_discrete_settings.clone())
            }
            LabModelFamily::Continuous => {
                LabModelSettings::Continuous(self.automata_lab_settings.clone())
            }
        }
    }
    pub(super) fn set_lab_draft_model(&mut self, model: LabModelSettings) {
        match model {
            LabModelSettings::Discrete(settings) => {
                self.automata_lab_model_family = LabModelFamily::Discrete;
                self.lab_kernel_preset = CellularKernelPreset::Custom;
                let radius = settings
                    .kernel
                    .iter()
                    .map(|k| k.dx.abs().max(k.dy.abs()) as u32)
                    .max()
                    .unwrap_or(1);
                self.automata_lab_kernel_grid_size = (radius * 2 + 1).clamp(3, 15);
                self.automata_lab_discrete_settings = settings;
            }
            LabModelSettings::Continuous(settings) => {
                self.automata_lab_model_family = LabModelFamily::Continuous;
                self.automata_lab_settings = settings;
            }
        }
    }
    pub(super) fn applied_lab_configuration(&self) -> Result<LabExperimentConfiguration, String> {
        if self.processing_job.is_some()
            || self.automata_lab_job.is_some()
            || self.automata_lab_playing
        {
            return Err(
                "Pause the Lab and wait for processing to finish before saving or capturing."
                    .into(),
            );
        }
        let session = self
            .automata_lab_session
            .as_ref()
            .ok_or("Initialize the Lab first.")?;
        let automaton = self
            .lab_applied_model
            .clone()
            .ok_or("Initialize the Lab first.")?;
        if automaton.program_document(session.calibration())? != *session.program_document() {
            return Err("Applied Lab configuration does not match the current session.".into());
        }
        let mut preprocessing = self.applied_experiment_configuration();
        preprocessing.cellular_automata = None;
        if let Some(displayed) = &self.displayed_image {
            preprocessing.histogram_bin_count = displayed.bin_count;
        }
        let experiment = LabExperimentConfiguration {
            schema_version: 1,
            preprocessing,
            automaton,
            generations: session.generation(),
        };
        experiment.validate()?;
        Ok(experiment)
    }
    fn save_lab_configuration(&self) -> Result<(), String> {
        let configuration = self.applied_lab_configuration()?;
        let path = Path::new(self.lab_configuration_path.trim());
        if path.extension().is_none_or(|ext| ext != "json") {
            return Err("Choose a .json file for the Lab configuration.".into());
        }
        let json = serde_json::to_vec_pretty(&configuration).map_err(|e| e.to_string())?;
        if json.len() as u64 > MAX_LAB_CONFIG_BYTES {
            return Err("Lab configuration is too large.".into());
        }
        fs::write(path, json).map_err(|e| format!("Could not save Lab configuration: {e}"))
    }
    fn load_lab_configuration(&mut self) -> Result<(), String> {
        use std::io::Read;
        let file = fs::File::open(self.lab_configuration_path.trim()).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take(MAX_LAB_CONFIG_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_LAB_CONFIG_BYTES {
            return Err("Lab configuration is too large.".into());
        }
        let configuration: LabExperimentConfiguration =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        configuration.validate()?;
        self.sync_controls_from_configuration(&configuration.preprocessing);
        self.set_lab_draft_model(configuration.automaton.clone());
        self.lab_saved_configuration = Some(configuration);
        Ok(())
    }
    pub(super) fn apply_lab_configuration(&mut self, ctx: &egui::Context) {
        let Some(configuration) = self.lab_saved_configuration.clone() else {
            return;
        };
        let Some((index, path, name)) = self.selected_image.and_then(|i| {
            self.images
                .get(i)
                .map(|image| (i, image.path.clone(), image.name.clone()))
        }) else {
            return;
        };
        self.sync_controls_from_configuration(&configuration.preprocessing);
        self.start_processing_job(
            ctx,
            index,
            path,
            name,
            ProcessingKind::Configuration(configuration.preprocessing.clone()),
        );
        self.lab_pending_configuration = Some(configuration);
    }
    pub(super) fn show_lab_configuration_controls(&mut self, ui: &mut Ui) {
        ui.heading("Lab Configuration");
        ui.text_edit_singleline(&mut self.lab_configuration_path);
        let busy = self.processing_job.is_some()
            || self.automata_lab_job.is_some()
            || self.automata_lab_playing
            || self.masks.is_busy();
        ui.horizontal_wrapped(|ui| {
            if ui.add_enabled(!busy && self.automata_lab_session.is_some(), Button::new("Save Applied")).clicked() {
                self.automata_lab_notice = Some(match self.save_lab_configuration() { Ok(()) => "Saved applied image settings, automaton, and generation count.".into(), Err(e) => e });
            }
            if ui.add_enabled(!busy, Button::new("Load")).clicked() {
                self.automata_lab_notice = Some(match self.load_lab_configuration() { Ok(()) => "Loaded settings. Apply Saved rebuilds this experiment on the selected image.".into(), Err(e) => e });
            }
            if ui.add_enabled(!busy && self.lab_saved_configuration.is_some() && self.selected_image.is_some(), Button::new("Apply Saved")).clicked() {
                self.apply_lab_configuration(ui.ctx());
            }
        });
        if let Some(configuration) = &self.lab_saved_configuration {
            ui.small(format!(
                "Saved run: {} generations",
                configuration.generations
            ));
        }
        ui.small("Save Applied records the current run, including its image preparation. Pause before saving or capturing a mask.");
    }
    pub(super) fn show_discrete_kernel_grid(&mut self, ui: &mut Ui) {
        let size = self.automata_lab_kernel_grid_size as usize;
        let half = (size / 2) as i32;
        let original_kernel = self.automata_lab_discrete_settings.kernel.clone();
        let cells: Vec<f32> = (0..size * size)
            .map(|i| {
                original_kernel
                    .iter()
                    .find(|k| k.dx == (i % size) as i32 - half && k.dy == (i / size) as i32 - half)
                    .map(|k| {
                        if k.polarity == SamplePolarity::Positive {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .unwrap_or(0.0)
            })
            .collect();
        let mut editor = CellularAutomataSettings {
            kernel_size: size,
            kernel: cells.clone(),
            omni_directional: self.automata_lab_discrete_settings.omni_directional,
            ..Default::default()
        };
        kernel_editor::show(
            ui,
            &mut editor,
            &mut self.lab_kernel_preset,
            &mut self.lab_kernel_angle,
            &mut self.automata_lab_kernel_brush,
            &mut self.automata_lab_kernel_drag_brush,
        );
        self.automata_lab_kernel_grid_size = editor.kernel_size as u32;
        self.automata_lab_discrete_settings.omni_directional = editor.omni_directional;
        if editor.kernel != cells || editor.kernel_size != size {
            let half = (editor.kernel_size / 2) as i32;
            self.automata_lab_discrete_settings.kernel = editor
                .kernel
                .iter()
                .enumerate()
                .filter_map(|(i, w)| {
                    (*w != 0.0).then_some(KernelOffset {
                        dx: (i % editor.kernel_size) as i32 - half,
                        dy: (i / editor.kernel_size) as i32 - half,
                        polarity: if *w > 0.0 {
                            SamplePolarity::Positive
                        } else {
                            SamplePolarity::Negative
                        },
                    })
                })
                .collect();
        }
        ui.small("+ reads state; − reads q − 1 − state. Omni-directional evaluates four rotations and keeps the highest output state.");
        if self.automata_lab_discrete_settings.rule_kind == DiscreteRuleKind::Direct {
            ui.small("Direct mapping ignores the kernel. Choose a neighbourhood rule to use these cells.");
        }
    }
    pub(super) fn show_discrete_generators(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new("Random generators").default_open(true).show(ui, |ui| {
            ui.add(DragValue::new(&mut self.lab_random_seed).prefix("Seed: "));
            if ui.button("Next seed").clicked() { self.lab_random_seed = self.lab_random_seed.wrapping_add(1); }
            ui.add(egui::Slider::new(&mut self.lab_random_density, 0.01..=1.0).text("Kernel density"));
            ui.add(egui::Slider::new(&mut self.lab_random_negative, 0.0..=1.0).text("Negative fraction"));
            ui.horizontal_wrapped(|ui| {
                if ui.button("Random kernel").clicked() {
                    let s = &self.automata_lab_discrete_settings;
                    let mut max_samples = 225;
                    if s.rule_kind == DiscreteRuleKind::OrderedNeighborhood {
                        let mut entries = 1usize; max_samples = 0;
                        while entries <= MAX_AUTHORING_TABLE_LEN / s.state_count as usize { entries *= s.state_count as usize; max_samples += 1; }
                    }
                    self.lab_kernel_preset = CellularKernelPreset::Custom;
                    self.automata_lab_discrete_settings.kernel = random_kernel(self.automata_lab_kernel_grid_size, self.lab_random_seed, self.lab_random_density, self.lab_random_negative, max_samples);
                    self.sync_discrete_table_lengths();
                }
                if ui.button("Random rule mapping").clicked() {
                    if let Err(e) = randomize_rule(&mut self.automata_lab_discrete_settings, self.lab_random_seed) { self.automata_lab_error = Some(e); }
                }
            });
            ui.small("The same seed and settings reproduce the same generator result. Ordered kernels are limited to fit 65,536 rules. Apply Settings & Reset to run changes.");
        });
    }
    pub(super) fn show_discrete_visualizations(&mut self, ui: &mut Ui) {
        let s = &self.automata_lab_discrete_settings;
        let count = s.kernel.len();
        let negatives = s
            .kernel
            .iter()
            .filter(|k| k.polarity == SamplePolarity::Negative)
            .count();
        ui.label(format!(
            "{} states · {} positive / {} negative samples",
            s.state_count,
            count - negatives,
            negatives
        ));
        let (table, x_label, explanation) = match s.rule_kind {
            DiscreteRuleKind::Direct => (
                &s.direct_table,
                "Current cell state",
                "Each cell looks up its own state and takes the mapped output.",
            ),
            DiscreteRuleKind::Aggregate => (
                &s.aggregate_table,
                "Sensed aggregate",
                "Reverse negative samples, then sum or round their mean. Use that value as the table index.",
            ),
            DiscreteRuleKind::OrderedNeighborhood => (
                &s.ordered_table,
                "Neighbourhood index",
                "Reverse negative samples, then encode samples in kernel order: index = s0 + q·s1 + q²·s2 + … . Editing the kernel orders samples row by row; hover cells for offsets.",
            ),
            DiscreteRuleKind::PresenceTriggered => (
                &s.presence_fallback,
                "Current cell state (fallback)",
                "Look for trigger states among sensed samples after reversal. Resolve simultaneous triggers using the selected reduction. Use the fallback only when no trigger matches.",
            ),
        };
        ui.label(explanation);
        match s.expected_table_len() {
            Ok(len) if len <= MAX_AUTHORING_TABLE_LEN => {
                ui.small(format!("Required mapping entries: {len}"));
            }
            _ => {
                ui.colored_label(ui.visuals().error_fg_color, "Rule table exceeds 65,536 entries. Reduce state count or kernel samples before initializing or saving.");
                return;
            }
        }
        let points: Vec<[f64; 2]> = table
            .iter()
            .enumerate()
            .step_by(table.len().div_ceil(512).max(1))
            .map(|(i, v)| [i as f64, *v as f64])
            .collect();
        Plot::new("discrete mapping graph")
            .height(150.0)
            .x_axis_label(x_label)
            .y_axis_label("Next state")
            .legend(egui_plot::Legend::default())
            .include_y(0.0)
            .include_y((s.state_count - 1) as f64)
            .show(ui, |plot| {
                plot.points(egui_plot::Points::new("Rule output", points).radius(2.5));
                if s.rule_kind == DiscreteRuleKind::PresenceTriggered {
                    plot.points(
                        egui_plot::Points::new(
                            "Trigger → output",
                            s.presence_transitions
                                .iter()
                                .map(|t| [t.trigger_state as f64, t.output_state as f64])
                                .collect::<Vec<_>>(),
                        )
                        .color(Color32::YELLOW)
                        .radius(4.0),
                    );
                }
            });
        if table.len() > 512 {
            ui.small("Graph samples the table at regular intervals (up to 512 entries).");
        }
        if let Some(displayed) = &self.displayed_image {
            let q = s.state_count as usize;
            let mut bins = vec![0usize; q];
            let minimum = displayed.calibration.minimum as u64;
            let levels = displayed.calibration.level_count();
            for v in &displayed.prepared_values {
                let index = ((*v as u64 - minimum) * q as u64 / levels) as usize;
                bins[index.min(q - 1)] += 1;
            }
            Plot::new("lab initial state histogram")
                .height(110.0)
                .x_axis_label("Initial state after quantization")
                .y_axis_label("Pixels")
                .show(ui, |plot| {
                    plot.bar_chart(BarChart::new(
                        "Generation zero",
                        bins.iter()
                            .enumerate()
                            .map(|(i, n)| Bar::new(i as f64, *n as f64).width(0.85))
                            .collect(),
                    ));
                });
        }
    }
    pub(super) fn show_continuous_visualizations(&self, ui: &mut Ui) {
        let s = &self.automata_lab_settings;
        let calibration = crate::quantization::IntensityCalibration::full_scale(256).unwrap();
        if let Ok(document) = s.program_document(calibration)
            && let AutomataModelDefinition::ContinuousLenia(definition) = document.transition_model
        {
            let kernel = definition.program.kernel;
            let sum: f32 = kernel.weights.iter().sum();
            let max = kernel
                .weights
                .iter()
                .copied()
                .fold(0.0f32, f32::max)
                .max(f32::EPSILON);
            ui.label("Sensing kernel (brighter = stronger weight)");
            let side = ui.available_width().min(180.0);
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(side), Sense::hover());
            let cell = side / kernel.width as f32;
            for (i, weight) in kernel.weights.iter().enumerate() {
                let x = i % kernel.width as usize;
                let y = i / kernel.width as usize;
                let r = Rect::from_min_size(
                    rect.min + Vec2::new(x as f32 * cell, y as f32 * cell),
                    Vec2::splat(cell),
                );
                ui.painter().rect_filled(
                    r,
                    0.0,
                    Color32::from_rgb(
                        0,
                        (weight / max * 210.0) as u8,
                        (weight / max * 190.0) as u8,
                    ),
                );
                if response.hover_pos().is_some_and(|p| r.contains(p)) {
                    response.clone().on_hover_text(format!(
                        "Offset ({}, {}): normalized weight {:.5}",
                        x as i32 - kernel.anchor_x as i32,
                        y as i32 - kernel.anchor_y as i32,
                        weight / sum
                    ));
                }
            }
            ui.small("Weights sum to 1. The image initializes state in [0, 1].");
        }
        let points: Vec<_> = (0..=256)
            .map(|i| {
                let x = i as f64 / 256.0;
                let z = (x - s.growth_center as f64) / s.growth_width as f64;
                [x, 2.0 * (-0.5 * z * z).exp() - 1.0]
            })
            .collect();
        Plot::new("lab growth response")
            .height(150.0)
            .x_axis_label("Weighted neighbourhood state")
            .y_axis_label("Growth response")
            .include_y(-1.0)
            .include_y(1.0)
            .show(ui, |plot| {
                plot.line(Line::new("Growth", points));
            });
        ui.small("Above zero: grow. Below zero: decay. Next state = clamp(state + time step × response, 0, 1).");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn finish(app: &mut VCEMApp, ctx: &egui::Context) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while app.processing_job.is_some()
            || app.automata_lab_job.is_some()
            || app.lab_replay_generations.is_some()
        {
            assert!(Instant::now() < deadline, "Lab workflow timed out");
            app.poll_processing_job(ctx);
            app.poll_automata_lab_job(ctx);
            app.poll_lab_replay(ctx);
            thread::sleep(Duration::from_millis(1));
        }
        assert!(app.viewer_error.is_none(), "{:?}", app.viewer_error);
        assert!(
            app.automata_lab_error.is_none(),
            "{:?}",
            app.automata_lab_error
        );
    }
    #[test]
    fn lab_image_controls_save_applied_and_load_replay_the_same_generation() {
        let ctx = egui::Context::default();
        let dir = std::env::temp_dir().join(format!("vcem-lab-ui-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("image.png");
        image::GrayImage::from_fn(8, 4, |x, y| image::Luma([(x * 31 + y * 7) as u8]))
            .save(&path)
            .unwrap();
        let mut app = VCEMApp::initial_state();
        app.view_mode = ViewMode::AutomataLab;
        app.add_image_paths(vec![path]);
        app.ensure_selected_image_loaded(&ctx);
        finish(&mut app, &ctx);
        app.gamma_value = 2.0;
        app.apply_gamma_adjustment(&ctx);
        finish(&mut app, &ctx);
        app.automata_lab_model_family = LabModelFamily::Discrete;
        app.automata_lab_discrete_settings.direct_table = vec![1, 0];
        app.initialize_automata_lab(&ctx);
        app.step_automata_lab(&ctx, 3);
        finish(&mut app, &ctx);
        let before =
            crate::lab_configuration::snapshot_result(app.automata_lab_session.as_ref().unwrap())
                .unwrap();
        app.lab_configuration_path = dir.join("lab.json").display().to_string();
        // Unapplied edits must not change the configuration of a captured run.
        app.gamma_value = 3.0;
        app.automata_lab_discrete_settings.direct_table = vec![0, 0];
        app.save_lab_configuration().unwrap();
        let saved: LabExperimentConfiguration =
            serde_json::from_slice(&fs::read(&app.lab_configuration_path).unwrap()).unwrap();
        assert_eq!(saved.generations, 3);
        assert_eq!(saved.preprocessing.gamma, Some(2.0));
        let LabModelSettings::Discrete(s) = &saved.automaton else {
            panic!()
        };
        assert_eq!(s.direct_table, vec![1, 0]);
        app.load_lab_configuration().unwrap();
        app.apply_lab_configuration(&ctx);
        finish(&mut app, &ctx);
        let after =
            crate::lab_configuration::snapshot_result(app.automata_lab_session.as_ref().unwrap())
                .unwrap();
        assert_eq!(before, after);
        assert_eq!(app.applied_gamma, Some(2.0));
        // Both model panels and their graphs render with the shared image controls.
        for family in [LabModelFamily::Discrete, LabModelFamily::Continuous] {
            app.automata_lab_model_family = family;
            let output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 6000.0))),
                    ..Default::default()
                },
                |ui| {
                    app.show_right_panel(ui);
                    app.show_central_panel(ui);
                },
            );
            assert!(!output.shapes.is_empty());
        }
        app.gamma_value = 1.0;
        app.apply_gamma_adjustment(&ctx);
        assert!(app.automata_lab_session.is_none());
        finish(&mut app, &ctx);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn oversized_ordered_rule_does_not_allocate_or_initialize() {
        let mut app = VCEMApp::initial_state();
        app.automata_lab_discrete_settings.rule_kind = DiscreteRuleKind::OrderedNeighborhood;
        app.automata_lab_discrete_settings.state_count = 64;
        app.automata_lab_discrete_settings.kernel = random_kernel(15, 1, 1.0, 0.5, 225);
        let old = app.automata_lab_discrete_settings.ordered_table.clone();
        app.sync_discrete_table_lengths();
        assert_eq!(old, app.automata_lab_discrete_settings.ordered_table);
        assert!(
            LabModelSettings::Discrete(app.automata_lab_discrete_settings)
                .validate()
                .is_err()
        );
    }
}

impl VCEMApp {
    /// Fixed rows keep progress updates from changing the scroll extent.
    pub(super) fn show_lab_run_controls(&mut self, ui: &mut Ui) -> Rect {
        let authored = self
            .displayed_image
            .as_ref()
            .and_then(|d| self.lab_draft_model().program_document(d.calibration).ok());
        let dirty = self
            .automata_lab_session
            .as_ref()
            .is_some_and(|s| authored.as_ref() != Some(s.program_document()));
        if dirty {
            self.automata_lab_playing = false;
            self.cancel_automata_lab_job();
        }
        ui.group(|ui| {
            let width = ui.available_width();
            let line = |ui: &mut Ui, text: String, color: Color32| {
                let label = egui::Label::new(
                    egui::RichText::new(if text.is_empty() {
                        " ".to_owned()
                    } else {
                        text.clone()
                    })
                    .color(color)
                    .monospace(),
                )
                .truncate();
                ui.add_sized([width, 20.0], label).on_hover_text(text);
            };
            line(
                ui,
                if dirty {
                    "Settings changed — apply and reset".into()
                } else {
                    "Simulation".into()
                },
                if dirty {
                    Color32::YELLOW
                } else {
                    ui.visuals().text_color()
                },
            );
            let has_session = self.automata_lab_session.is_some();
            let stepping = self.automata_lab_job.is_some();
            let label = if has_session {
                "Apply Settings & Reset"
            } else {
                "Initialize Lab"
            };
            if ui
                .add_enabled(
                    self.displayed_image.is_some(),
                    Button::new(label).min_size(Vec2::new(width, 24.0)),
                )
                .clicked()
            {
                self.initialize_automata_lab(ui.ctx());
            }
            ui.horizontal(|ui| {
                let button_width = ((width - 2.0 * ui.spacing().item_spacing.x) / 3.0).max(30.0);
                for (action, label) in [
                    (0, "Reset"),
                    (1, "Step"),
                    (
                        2,
                        if self.automata_lab_playing {
                            "Pause"
                        } else {
                            "Play"
                        },
                    ),
                ] {
                    let enabled = has_session && !dirty && (action != 1 || !stepping);
                    if ui
                        .add_enabled(
                            enabled,
                            Button::new(label).min_size(Vec2::new(button_width, 24.0)),
                        )
                        .clicked()
                    {
                        match action {
                            0 => self.reset_automata_lab(ui.ctx()),
                            1 => {
                                self.step_automata_lab(ui.ctx(), self.automata_lab_cycles_per_step)
                            }
                            _ => {
                                self.automata_lab_playing = !self.automata_lab_playing;
                                self.automata_lab_last_step = Instant::now();
                                ui.ctx().request_repaint();
                            }
                        }
                    }
                }
            });
            let cancelling = self
                .automata_lab_job
                .as_ref()
                .is_some_and(|job| job.discard_result);
            let status = if cancelling {
                "Cancelling…"
            } else if stepping {
                "Computing…"
            } else if self.automata_lab_playing {
                "Playing"
            } else {
                "Paused"
            };
            line(ui, status.into(), ui.visuals().text_color());
            let (fraction, text) = if let Some((n, total)) = self.automata_lab_step_progress {
                (n as f32 / total.max(1) as f32, format!("{n} / {total}"))
            } else {
                (0.0, if stepping { "Running" } else { "Ready" }.into())
            };
            ui.add_sized([width, 18.0], egui::ProgressBar::new(fraction).text(text));
            let session = self.automata_lab_session.as_ref();
            line(
                ui,
                format!(
                    "Generation {}",
                    session.map(|s| s.generation()).unwrap_or(0)
                ),
                ui.visuals().text_color(),
            );
            let mut rows = [
                "Mean / nonzero: —".to_owned(),
                "Change: —".to_owned(),
                "Cycle / fixed point: —".to_owned(),
            ];
            match session.and_then(|s| s.last_diagnostics()) {
                Some(LabDiagnostics::Continuous(d)) => {
                    rows[0] = format!("Mean {:.4} · mass {:.2}", d.final_mean, d.final_mass);
                    rows[1] = format!(
                        "Max change: {}",
                        d.last_transition_max_delta
                            .map(|v| format!("{v:.6}"))
                            .unwrap_or_else(|| "—".into())
                    );
                    if let Some(g) = d.exact_fixed_point_generation {
                        rows[2] = format!("Fixed point from {g}");
                    }
                }
                Some(LabDiagnostics::Discrete(d)) => {
                    rows[0] = format!("Nonzero: {}", d.final_nonzero_cells);
                    rows[1] = format!("Changed: {}", d.last_transition_changed_cells);
                    if let Some(c) = d.cycle {
                        rows[2] = format!("Period {} from {}", c.period, c.start_generation);
                    }
                }
                None => {}
            }
            for row in rows {
                line(ui, row, ui.visuals().text_color());
            }
            line(
                ui,
                format!(
                    "History: {:.1} MiB",
                    session.map(|s| s.retained_history_bytes()).unwrap_or(0) as f64
                        / (1024.0 * 1024.0)
                ),
                ui.visuals().text_color(),
            );
            let message = self
                .automata_lab_error
                .as_ref()
                .or(self.automata_lab_notice.as_ref())
                .cloned()
                .unwrap_or_default();
            line(
                ui,
                message,
                if self.automata_lab_error.is_some() {
                    ui.visuals().error_fg_color
                } else {
                    ui.visuals().text_color()
                },
            );
        })
        .response
        .rect
    }
}

#[cfg(test)]
mod run_layout_tests {
    use super::*;
    #[test]
    fn simulation_run_panel_has_fixed_bounds_across_execution_states() {
        let ctx = egui::Context::default();
        let mut app = VCEMApp::initial_state();
        let calibration = crate::quantization::IntensityCalibration::full_scale(256).unwrap();
        let texture = ctx.load_texture(
            "layout fixture",
            egui::ColorImage::new([2, 2], vec![Color32::BLACK; 4]),
            TextureOptions::NEAREST,
        );
        app.displayed_image = Some(DisplayedImage {
            image_index: 0,
            size: Vec2::new(2.0, 2.0),
            texture,
            prepared_values: vec![0, 64, 128, 255],
            calibration,
            bin_indices: vec![0, 64, 128, 255],
            bin_count: 256,
            highlight_bins: Vec::new(),
            highlight_texture: None,
            cellular_automata_texture: None,
            cellular_automata_summary: None,
            cellular_automata_result: None,
        });
        let render = |app: &mut VCEMApp| {
            let mut rect = Rect::NOTHING;
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(600.0, 700.0))),
                    ..Default::default()
                },
                |ui| {
                    ui.set_width(260.0);
                    rect = app.show_lab_run_controls(ui);
                },
            );
            rect
        };
        let idle = render(&mut app);
        app.initialize_automata_lab(&ctx);
        assert_eq!(render(&mut app), idle);
        let (_sender, receiver) = mpsc::channel();
        app.automata_lab_job = Some(AutomataLabJob {
            receiver,
            cancellation: Arc::new(AtomicBool::new(false)),
            discard_result: false,
        });
        assert_eq!(render(&mut app), idle);
        app.automata_lab_step_progress = Some((3, 32));
        app.automata_lab_playing = true;
        assert_eq!(render(&mut app), idle);
        app.automata_lab_session.as_mut().unwrap().step(1).unwrap();
        assert_eq!(render(&mut app), idle);
        app.automata_lab_job = None;
        app.automata_lab_playing = false;
        app.automata_lab_error=Some("A long diagnostic that would normally wrap over several rows and move the simulation controls while the scroll area is anchored at its bottom".into());
        assert_eq!(render(&mut app), idle);
    }
}
