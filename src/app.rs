use eframe::{
    App, CreationContext, Frame,
    egui::{
        self, Button, CentralPanel, Color32, DragValue, Id, Modal, Panel, PointerButton, Pos2,
        Rect, ScrollArea, Sense, Stroke, TextureOptions, Ui, Vec2,
    },
};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, PlotUi};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread,
};

use crate::analysis::{intensity_histogram_from_path, processed_image_from_path};
use crate::types::{
    CellularAutomataResponse, CellularAutomataSeedMode, CellularAutomataSettings, DenoiseMethod,
    DenoiseSettings, DisplayedImage, ExperimentConfiguration, ImageFile, ImageRotation,
    IntensityHistogram, ProcessedImage,
};

pub struct VCEMApp {
    upper_panel_expanded: bool,
    view_mode: ViewMode,
    images: Vec<ImageFile>,
    selected_image: Option<usize>,
    displayed_image: Option<DisplayedImage>,
    viewer_error: Option<String>,
    show_image_picker: bool,
    image_picker_dir: String,
    image_candidates: Vec<ImageCandidate>,
    image_picker_error: Option<String>,
    viewer_zoom: f32,
    viewer_pan: Vec2,
    viewer_needs_fit: bool,
    picker_drag_selection: Option<bool>,
    histogram_bin_count: usize,
    histogram: Option<IntensityHistogram>,
    histogram_error: Option<String>,
    reset_histogram_view: bool,
    reset_cellular_response_view: bool,
    selected_histogram_bins: BTreeSet<usize>,
    image_selection_start: Option<Pos2>,
    histogram_selection_start_bin: Option<usize>,
    rotation: ImageRotation,
    applied_rotation: ImageRotation,
    denoise_method: DenoiseMethod,
    denoise_kernel_size: usize,
    applied_denoise: Option<DenoiseSettings>,
    invert_values: bool,
    applied_invert: bool,
    equalization_strength: f64,
    applied_equalization: Option<f64>,
    gamma_value: f64,
    applied_gamma: Option<f64>,
    cellular_automata_settings: CellularAutomataSettings,
    applied_cellular_automata: Option<CellularAutomataSettings>,
    cellular_kernel_preset: CellularKernelPreset,
    cellular_kernel_angle: f32,
    cellular_kernel_brush: CellularKernelBrush,
    cellular_kernel_drag_brush: Option<CellularKernelBrush>,
    configuration_path: String,
    saved_configuration: Option<ExperimentConfiguration>,
    configuration_notice: Option<String>,
    configuration_error: Option<String>,
    processing_job: Option<ProcessingJob>,
    next_processing_job_id: u64,
}

struct ImageCandidate {
    path: PathBuf,
    name: String,
    selected: bool,
    kind: ImagePickerEntryKind,
}

struct ProcessingJob {
    id: u64,
    image_index: usize,
    receiver: Receiver<ProcessingResult>,
    kind: ProcessingKind,
}

struct ProcessingResult {
    id: u64,
    image_index: usize,
    image_name: String,
    kind: ProcessingKind,
    result: std::result::Result<ProcessedImage, String>,
}

#[derive(Clone)]
enum ProcessingKind {
    Preview(ExperimentConfiguration),
    Rotation(ExperimentConfiguration),
    Denoise(ExperimentConfiguration),
    Invert(ExperimentConfiguration),
    Equalize(ExperimentConfiguration),
    Gamma(ExperimentConfiguration),
    CellularAutomata(ExperimentConfiguration),
    Configuration(ExperimentConfiguration),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CellularKernelPreset {
    Custom,
    Disk,
    Ring,
    Line,
    Cross,
    Crescent,
    Clear,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CellularKernelBrush {
    Empty,
    Positive,
    Negative,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImagePickerEntryKind {
    Directory,
    SupportedImage,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    Inspect,
    Segment,
    Measure,
}

impl VCEMApp {
    pub fn new(_: &CreationContext<'_>) -> Self {
        let current_dir = std::env::current_dir().unwrap_or_default();
        let image_picker_dir = current_dir.display().to_string();
        let configuration_path = current_dir
            .join("vcem-experiment.json")
            .display()
            .to_string();

        Self {
            upper_panel_expanded: true,
            view_mode: ViewMode::Inspect,
            images: Vec::new(),
            selected_image: None,
            displayed_image: None,
            viewer_error: None,
            show_image_picker: false,
            image_picker_dir,
            image_candidates: Vec::new(),
            image_picker_error: None,
            viewer_zoom: 1.0,
            viewer_pan: Vec2::ZERO,
            viewer_needs_fit: true,
            picker_drag_selection: None,
            histogram_bin_count: 256,
            histogram: None,
            histogram_error: None,
            reset_histogram_view: false,
            reset_cellular_response_view: false,
            selected_histogram_bins: BTreeSet::new(),
            image_selection_start: None,
            histogram_selection_start_bin: None,
            rotation: ImageRotation::None,
            applied_rotation: ImageRotation::None,
            denoise_method: DenoiseMethod::None,
            denoise_kernel_size: 3,
            applied_denoise: None,
            invert_values: false,
            applied_invert: false,
            equalization_strength: 0.0,
            applied_equalization: None,
            gamma_value: 1.0,
            applied_gamma: None,
            cellular_automata_settings: CellularAutomataSettings::default(),
            applied_cellular_automata: None,
            cellular_kernel_preset: CellularKernelPreset::Disk,
            cellular_kernel_angle: 0.0,
            cellular_kernel_brush: CellularKernelBrush::Positive,
            cellular_kernel_drag_brush: None,
            configuration_path,
            saved_configuration: None,
            configuration_notice: None,
            configuration_error: None,
            processing_job: None,
            next_processing_job_id: 1,
        }
    }

    fn show_upper_panel(&mut self, ui: &mut Ui) {
        let mut upper_panel_expanded = self.upper_panel_expanded;
        let mut toggle_upper_panel = false;
        let mut view_mode = self.view_mode;

        Panel::show_switched(
            ui,
            &mut upper_panel_expanded,
            Panel::top("upper_panel_collapsed")
                .resizable(true)
                .exact_size(28.0),
            Panel::top("upper_panel_expanded")
                .resizable(true)
                .default_size(116.0)
                .min_size(72.0)
                .max_size(180.0),
            |ui, expanded| {
                if expanded {
                    ui.horizontal(|ui| {
                        ui.heading("VCEM Analyser");
                        ui.separator();
                        if ui.button("Collapse upper panel").clicked() {
                            toggle_upper_panel = true;
                        }
                    });

                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Project:");
                        ui.strong("Cryo-EM TEM Image Experiments");
                        ui.separator();
                        ui.label("Mode:");
                        if ui
                            .selectable_label(view_mode == ViewMode::Inspect, "Inspect")
                            .clicked()
                        {
                            view_mode = ViewMode::Inspect;
                        }
                        if ui
                            .selectable_label(view_mode == ViewMode::Segment, "Segment")
                            .clicked()
                        {
                            view_mode = ViewMode::Segment;
                        }
                        if ui
                            .selectable_label(view_mode == ViewMode::Measure, "Measure")
                            .clicked()
                        {
                            view_mode = ViewMode::Measure;
                        }
                    });
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("Expand").clicked() {
                            toggle_upper_panel = true;
                        }
                        ui.separator();
                        ui.label("VCEM Analyser");
                    });
                }
            },
        );

        if toggle_upper_panel {
            upper_panel_expanded = !upper_panel_expanded;
        }
        self.upper_panel_expanded = upper_panel_expanded;
        self.view_mode = view_mode;
    }

    fn show_left_panel(&mut self, ui: &mut Ui) {
        Panel::left("left_panel")
            .resizable(true)
            .default_size(260.0)
            .min_size(160.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Frames");
                    ui.add_space(8.0);
                    if ui.button("Add Image").clicked() {
                        self.show_image_picker = true;
                        if self.image_candidates.is_empty() {
                            self.scan_image_directory();
                        }
                    }
                });

                ui.separator();

                if self.images.is_empty() {
                    ui.label("No images added.");
                } else {
                    let mut selected_image = self.selected_image;
                    ScrollArea::vertical().show(ui, |ui| {
                        for (index, image) in self.images.iter().enumerate() {
                            if ui
                                .selectable_label(selected_image == Some(index), &image.name)
                                .on_hover_text(image.path.display().to_string())
                                .clicked()
                            {
                                selected_image = Some(index);
                            }
                        }
                    });

                    if selected_image != self.selected_image {
                        self.selected_image = selected_image;
                        self.displayed_image = None;
                        self.viewer_error = None;
                        self.viewer_needs_fit = true;
                        self.histogram = None;
                        self.histogram_error = None;
                        self.reset_histogram_view = true;
                        self.clear_histogram_highlight();
                        self.applied_denoise = None;
                        self.invert_values = false;
                        self.applied_invert = false;
                        self.equalization_strength = 0.0;
                        self.applied_equalization = None;
                        self.applied_gamma = None;
                        self.rotation = ImageRotation::None;
                        self.applied_rotation = ImageRotation::None;
                        self.cellular_automata_settings = CellularAutomataSettings::default();
                        self.applied_cellular_automata = None;
                        self.cellular_kernel_preset = CellularKernelPreset::Disk;
                        self.cellular_kernel_angle = 0.0;
                        self.processing_job = None;
                    }
                }
            });
    }

    fn show_right_panel(&mut self, ui: &mut Ui) {
        Panel::right("right_panel")
            .resizable(true)
            .default_size(240.0)
            .min_size(160.0)
            .show(ui, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Analysis");
                    ui.separator();

                    if let Some(image) =
                        self.selected_image.and_then(|index| self.images.get(index))
                    {
                        let is_processing = self.processing_job.is_some();

                        ui.label(format!("File: {}", image.name));
                        if let Some(displayed) = self
                            .displayed_image
                            .as_ref()
                            .filter(|displayed| Some(displayed.image_index) == self.selected_image)
                        {
                            ui.label(format!(
                                "Size: {} x {}",
                                displayed.size.x as usize, displayed.size.y as usize
                            ));
                        } else {
                            ui.label("Size: not loaded");
                        }
                        if let Some(job) = &self.processing_job {
                            ui.label(format!("Processing {}...", job.kind.label()));
                        }

                        ui.add_space(5.0);
                        ui.heading("Image Augmentations");
                        ui.separator();
                        if ui
                            .add_enabled(
                                self.has_image_augmentation_state() || is_processing,
                                Button::new("Reset All Augmentations"),
                            )
                            .on_hover_text("Restores the current image to its unaugmented state.")
                            .clicked()
                        {
                            self.reset_image_augmentations(ui.ctx());
                        }

                        ui.add_space(8.0);
                        self.show_rotation_controls(ui, is_processing);

                        ui.add_space(8.0);
                        self.show_configuration_controls(ui, is_processing);

                        ui.add_space(5.0);
                        ui.heading("Intensity Histogram");
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Bin count");
                            let response = ui.add(
                                DragValue::new(&mut self.histogram_bin_count)
                                    .range(2..=4096)
                                    .speed(1),
                            )
                            .on_hover_text("Controls how finely intensities are divided into bins.");
                            if response.changed() {
                                self.displayed_image = None;
                                self.histogram = None;
                                self.histogram_error = None;
                                self.clear_histogram_highlight();
                                self.processing_job = None;
                                self.reset_histogram_view = true;
                            }
                        });

                        if ui
                            .add_enabled(!is_processing, Button::new("Create Histogram"))
                            .on_hover_text("Builds the intensity distribution for the current processed image.")
                            .clicked()
                        {
                            self.create_intensity_histogram();
                        }

                        if let Some(error) = &self.histogram_error {
                            ui.colored_label(ui.visuals().error_fg_color, error);
                        }

                        if let Some(histogram) = self.histogram.clone() {
                            ui.label(format!(
                                "{} pixels, {}, range {}..{}",
                                histogram.pixel_count,
                                histogram.source_description,
                                histogram.min_value,
                                histogram.max_value
                            ));
                            self.show_histogram_zoom_controls(ui, &histogram);
                            self.show_histogram_plot(ui, &histogram);
                        }

                        ui.add_space(8.0);
                        ui.heading("Denoising");
                        ui.separator();
                        egui::ComboBox::from_label("Method")
                            .selected_text(self.denoise_method.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.denoise_method,
                                    DenoiseMethod::None,
                                    DenoiseMethod::None.label(),
                                );
                                ui.selectable_value(
                                    &mut self.denoise_method,
                                    DenoiseMethod::Median,
                                    DenoiseMethod::Median.label(),
                                );
                                ui.selectable_value(
                                    &mut self.denoise_method,
                                    DenoiseMethod::Gaussian,
                                    DenoiseMethod::Gaussian.label(),
                                );
                            })
                            .response
                            .on_hover_text("Chooses the local denoising operation to apply before later augmentations.");

                        match self.denoise_method {
                            DenoiseMethod::None => {
                                ui.label("No denoising will be applied.");
                            }
                            DenoiseMethod::Median | DenoiseMethod::Gaussian => {
                                ui.horizontal(|ui| {
                                    ui.label("Kernel");
                                    ui.selectable_value(&mut self.denoise_kernel_size, 3, "3")
                                        .on_hover_text("Smallest denoise neighbourhood; preserves finer details.");
                                    ui.selectable_value(&mut self.denoise_kernel_size, 5, "5")
                                        .on_hover_text("Moderate denoise neighbourhood.");
                                    ui.selectable_value(&mut self.denoise_kernel_size, 7, "7")
                                        .on_hover_text("Largest denoise neighbourhood; smooths more aggressively.");
                                });
                            }
                        }

                        if ui
                            .add_enabled(!is_processing, Button::new("Apply Denoising"))
                            .on_hover_text("Applies the selected denoise method to the current image.")
                            .clicked()
                        {
                            self.apply_denoising(ui.ctx());
                        }

                        ui.add_space(8.0);
                        ui.heading("Inversion");
                        ui.separator();
                        ui.checkbox(&mut self.invert_values, "Invert values")
                            .on_hover_text("Swaps low and high intensities before later processing.");
                        if ui
                            .add_enabled(
                                !is_processing && self.invert_values != self.applied_invert,
                                Button::new("Apply Inversion"),
                            )
                            .on_hover_text("Applies or removes intensity inversion.")
                            .clicked()
                        {
                            self.apply_inversion(ui.ctx());
                        }

                        ui.add_space(8.0);
                        ui.heading("Distribution Equalization");
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Strength");
                            ui.add(
                                egui::Slider::new(&mut self.equalization_strength, 0.0..=1.0)
                                    .show_value(true),
                            )
                            .on_hover_text("Spreads occupied intensity values across more of the available range.");
                        });
                        if ui
                            .add_enabled(
                                !is_processing
                                    && Some(self.equalization_strength)
                                        != self.applied_equalization
                                    && (self.equalization_strength > 0.0
                                        || self.applied_equalization.is_some()),
                                Button::new("Apply Equalization"),
                            )
                            .on_hover_text("Applies the selected distribution equalization strength.")
                            .clicked()
                        {
                            self.apply_equalization(ui.ctx());
                        }

                        ui.add_space(8.0);
                        ui.heading("Gamma Adjustment");
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Gamma");
                            ui.add(
                                DragValue::new(&mut self.gamma_value)
                                    .range(0.001..=4.0)
                                    .speed(0.01),
                            )
                            .on_hover_text("Adjusts contrast nonlinearly while preserving intensity ordering.");
                        });

                        if ui
                            .add_enabled(!is_processing, Button::new("Apply Gamma Adjustment"))
                            .on_hover_text("Applies the selected gamma value to the current image.")
                            .clicked()
                        {
                            self.apply_gamma_adjustment(ui.ctx());
                        }

                        ui.add_space(8.0);
                        self.show_cellular_automata_controls(ui, is_processing);
                    } else {
                        ui.label("Analytical results live here.");
                    }
                });
            });
    }

    fn show_rotation_controls(&mut self, ui: &mut Ui, is_processing: bool) {
        ui.heading("Rotation");
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            for rotation in ImageRotation::ALL {
                ui.selectable_value(&mut self.rotation, rotation, rotation.label())
                    .on_hover_text(
                        "Chooses the image orientation used by all downstream analysis.",
                    );
            }
        });
        if ui
            .add_enabled(
                !is_processing && self.rotation != self.applied_rotation,
                Button::new("Apply Rotation"),
            )
            .on_hover_text("Rotates the image and rebuilds derived analysis data.")
            .clicked()
        {
            self.apply_rotation(ui.ctx());
        }
    }

    fn show_configuration_controls(&mut self, ui: &mut Ui, is_processing: bool) {
        ui.heading("Experiment Configuration");
        ui.separator();
        ui.add(
            egui::TextEdit::singleline(&mut self.configuration_path)
                .hint_text("Configuration path"),
        )
        .on_hover_text("Path used when saving or loading the experiment configuration.");
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(
                    !is_processing && self.displayed_image.is_some(),
                    Button::new("Save Applied"),
                )
                .on_hover_text("Saves the currently applied augmentation and CA settings.")
                .clicked()
            {
                self.save_applied_configuration();
            }
            if ui
                .add_enabled(!is_processing, Button::new("Load"))
                .on_hover_text("Loads an experiment configuration from the configured path.")
                .clicked()
            {
                self.load_configuration();
            }
            if ui
                .add_enabled(
                    !is_processing
                        && self.selected_image.is_some()
                        && self.saved_configuration.is_some(),
                    Button::new("Apply Saved"),
                )
                .on_hover_text("Runs the loaded configuration on the selected image.")
                .clicked()
            {
                self.apply_saved_configuration(ui.ctx());
            }
        });
        if let Some(notice) = &self.configuration_notice {
            ui.label(notice);
        }
        if let Some(error) = &self.configuration_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
    }

    fn show_cellular_automata_controls(&mut self, ui: &mut Ui, is_processing: bool) {
        ui.heading("Cellular Feature Filter");
        ui.separator();

        egui::ComboBox::from_label("Seed polarity")
            .selected_text(self.cellular_automata_settings.seed_mode.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.cellular_automata_settings.seed_mode,
                    CellularAutomataSeedMode::Dark,
                    CellularAutomataSeedMode::Dark.label(),
                );
                ui.selectable_value(
                    &mut self.cellular_automata_settings.seed_mode,
                    CellularAutomataSeedMode::Bright,
                    CellularAutomataSeedMode::Bright.label(),
                );
            })
            .response
            .on_hover_text(
                "Selects which intensity polarity becomes the initial feature evidence.",
            );

        ui.collapsing("Sensing Kernel", |ui| {
            let old_size = self.cellular_automata_settings.kernel_size;
            let mut new_size = old_size;
            egui::ComboBox::from_label("Size")
                .selected_text(format!("{old_size} x {old_size}"))
                .show_ui(ui, |ui| {
                    for size in (3..=15).step_by(2) {
                        ui.selectable_value(&mut new_size, size, format!("{size} x {size}"));
                    }
                })
                .response
                .on_hover_text("Sets the local neighbourhood size sampled around each pixel.");
            if new_size != old_size {
                resize_cellular_kernel(&mut self.cellular_automata_settings, new_size);
                self.cellular_kernel_preset = CellularKernelPreset::Custom;
            }

            let previous_preset = self.cellular_kernel_preset;
            egui::ComboBox::from_label("Preset")
                .selected_text(self.cellular_kernel_preset.label())
                .show_ui(ui, |ui| {
                    for preset in CellularKernelPreset::ALL {
                        ui.selectable_value(
                            &mut self.cellular_kernel_preset,
                            preset,
                            preset.label(),
                        );
                    }
                })
                .response
                .on_hover_text("Seeds the kernel editor with common sensing shapes.");
            if self.cellular_kernel_preset != previous_preset {
                apply_cellular_kernel_preset(
                    &mut self.cellular_automata_settings,
                    self.cellular_kernel_preset,
                    self.cellular_kernel_angle,
                );
            }
            if matches!(
                self.cellular_kernel_preset,
                CellularKernelPreset::Line | CellularKernelPreset::Crescent
            ) {
                let angle_changed = ui
                    .add(
                        egui::Slider::new(&mut self.cellular_kernel_angle, 0.0..=180.0)
                            .text("Angle")
                            .suffix(" degrees"),
                    )
                    .on_hover_text("Rotates directional presets before applying the filter.")
                    .changed();
                if angle_changed {
                    apply_cellular_kernel_preset(
                        &mut self.cellular_automata_settings,
                        self.cellular_kernel_preset,
                        self.cellular_kernel_angle,
                    );
                }
            }
            ui.checkbox(
                &mut self.cellular_automata_settings.omni_directional,
                "Omni-directional pass",
            )
            .on_hover_text(
                "Tests cardinal rotations of asymmetric kernels in each iteration and keeps the strongest response.",
            );

            ui.add_space(4.0);
            let size = self.cellular_automata_settings.kernel_size;
            let palette_width = ui.spacing().interact_size.x + 8.0;
            let cell_size =
                ((ui.available_width() - palette_width).max(0.0) / size as f32 - 2.0)
                    .clamp(12.0, 24.0);
            let primary_down = ui.input(|input| input.pointer.button_down(PointerButton::Primary));
            if !primary_down {
                self.cellular_kernel_drag_brush = None;
            }
            let pointer_pos = ui.input(|input| input.pointer.latest_pos());

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    for y in 0..size {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 2.0;
                            for x in 0..size {
                                let index = y * size + x;
                                let weight = self.cellular_automata_settings.kernel[index];
                                let response = ui
                                    .add_sized(
                                        [cell_size, cell_size],
                                        kernel_weight_button(weight).sense(Sense::click_and_drag()),
                                    )
                                    .on_hover_text(
                                        "Click toggles the selected sign. Drag toggles from the starting cell and paints or erases with it. Right click cycles in reverse.",
                                    );

                                if response.clicked_by(PointerButton::Primary) {
                                    self.cellular_automata_settings.kernel[index] =
                                        toggle_kernel_weight(weight, self.cellular_kernel_brush);
                                    self.cellular_kernel_preset = CellularKernelPreset::Custom;
                                } else if response.clicked_by(PointerButton::Secondary) {
                                    self.cellular_automata_settings.kernel[index] =
                                        next_kernel_weight(weight, true);
                                    self.cellular_kernel_preset = CellularKernelPreset::Custom;
                                }

                                if response.drag_started_by(PointerButton::Primary) {
                                    let drag_brush = toggled_kernel_brush(
                                        weight,
                                        self.cellular_kernel_brush,
                                    );
                                    self.cellular_kernel_drag_brush = Some(drag_brush);
                                    self.cellular_automata_settings.kernel[index] =
                                        drag_brush.weight();
                                    self.cellular_kernel_preset = CellularKernelPreset::Custom;
                                }

                                if primary_down
                                    && pointer_pos.is_some_and(|pos| response.rect.contains(pos))
                                    && let Some(brush) = self.cellular_kernel_drag_brush
                                {
                                    self.cellular_automata_settings.kernel[index] = brush.weight();
                                    self.cellular_kernel_preset = CellularKernelPreset::Custom;
                                }
                            }
                        });
                    }
                });

                ui.add_space(6.0);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    for brush in CellularKernelBrush::ALL {
                        let response = ui
                            .add_sized(
                                [cell_size, cell_size],
                                kernel_brush_button(brush, self.cellular_kernel_brush == brush),
                            )
                            .on_hover_text(brush.hover_text());
                        if response.clicked() {
                            self.cellular_kernel_brush = brush;
                        }
                    }
                });
            });
        });

        ui.collapsing("Growth Response", |ui| {
            egui::ComboBox::from_label("Response")
                .selected_text(self.cellular_automata_settings.response.label())
                .show_ui(ui, |ui| {
                    for response in CellularAutomataResponse::ALL {
                        ui.selectable_value(
                            &mut self.cellular_automata_settings.response,
                            response,
                            response.label(),
                        );
                    }
                })
                .response
                .on_hover_text("Maps the sensed kernel value to growth or suppression.");
            ui.add(
                egui::Slider::new(
                    &mut self.cellular_automata_settings.response_center,
                    -1.0..=1.0,
                )
                .text("Centre"),
            )
            .on_hover_text("Moves the target sensed value along the response curve.");
            ui.add(
                egui::Slider::new(
                    &mut self.cellular_automata_settings.response_width,
                    0.01..=1.0,
                )
                .text("Width"),
            )
            .on_hover_text("Controls how broad or selective the response curve is.");
            if ui
                .button("Reset")
                .on_hover_text("Restores the response plot view bounds.")
                .clicked()
            {
                self.reset_cellular_response_view = true;
            }

            let response_points = PlotPoints::from_iter((-100..=100).map(|sample| {
                let sensed = sample as f64 / 100.0;
                [
                    sensed,
                    cellular_response_preview(sensed as f32, &self.cellular_automata_settings)
                        as f64,
                ]
            }));
            let mut plot = Plot::new("cellular_growth_response")
                .height(110.0)
                .include_x(-1.0)
                .include_x(1.0)
                .include_y(-1.0)
                .include_y(1.0)
                .allow_drag(true)
                .pan_pointer_button(PointerButton::Secondary)
                .allow_scroll(false)
                .allow_zoom(true)
                .allow_boxed_zoom(false)
                .allow_axis_zoom_drag(false);

            if self.reset_cellular_response_view {
                plot = plot.reset();
                self.reset_cellular_response_view = false;
            }

            plot.show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Growth", response_points).color(Color32::from_rgb(30, 210, 190)),
                );
                zoom_hovered_plot_with_scroll(plot_ui);
            });
        });

        ui.collapsing("Evolution", |ui| {
            ui.add(
                egui::Slider::new(&mut self.cellular_automata_settings.growth_rate, 0.0..=2.0)
                    .text("Growth"),
            )
            .on_hover_text("Scales positive response into activation growth.");
            ui.add(
                egui::Slider::new(&mut self.cellular_automata_settings.decay_rate, 0.0..=2.0)
                    .text("Decay"),
            )
            .on_hover_text("Controls how quickly unsupported activation fades.");
            ui.add(
                egui::Slider::new(&mut self.cellular_automata_settings.time_step, 0.01..=0.5)
                    .text("Time step"),
            )
            .on_hover_text("Sets the amount of CA state change applied per step.");
            ui.add(
                egui::Slider::new(&mut self.cellular_automata_settings.iterations, 1..=128)
                    .text("Duration"),
            )
            .on_hover_text("Sets the maximum number of synchronous CA steps to run.");
        });

        if let Some(summary) = self
            .displayed_image
            .as_ref()
            .and_then(|displayed| displayed.cellular_automata_summary)
        {
            let state = if summary.converged {
                "converged"
            } else {
                "duration reached"
            };
            ui.label(format!(
                "{:.1}% active, mean {:.3}, {} after {} steps",
                summary.active_fraction * 100.0,
                summary.mean_activation,
                state,
                summary.iterations_run
            ));
        }

        ui.horizontal_wrapped(|ui| {
            let settings_changed =
                self.applied_cellular_automata.as_ref() != Some(&self.cellular_automata_settings);
            if ui
                .add_enabled(
                    !is_processing && settings_changed,
                    Button::new("Apply Feature Filter"),
                )
                .on_hover_text("Runs the CA feature filter and renders its overlay.")
                .clicked()
            {
                self.apply_cellular_automata(ui.ctx(), true);
            }
            if ui
                .add_enabled(
                    !is_processing && self.applied_cellular_automata.is_some(),
                    Button::new("Remove Overlay"),
                )
                .on_hover_text("Removes the CA overlay while keeping other applied augmentations.")
                .clicked()
            {
                self.apply_cellular_automata(ui.ctx(), false);
            }
        });
    }

    fn show_bottom_panel(&mut self, ui: &mut Ui) {
        Panel::bottom("bottom_panel")
            .resizable(true)
            .default_size(48.0)
            .min_size(32.0)
            .max_size(120.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if let Some(image) =
                        self.selected_image.and_then(|index| self.images.get(index))
                    {
                        ui.label(&image.name);
                    } else {
                        ui.label("No image loaded");
                    }

                    ui.separator();
                    ui.label(format!("Zoom {:.0}%", self.viewer_zoom * 100.0));
                });
            });
    }

    fn show_central_panel(&mut self, ui: &mut Ui) {
        CentralPanel::default_margins().show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Image Viewer");
                if self.selected_image.is_some() {
                    ui.separator();
                    if ui.button("Fit").clicked() {
                        self.viewer_needs_fit = true;
                    }
                    if !self.selected_histogram_bins.is_empty()
                        && ui.button("Clear Selection").clicked()
                    {
                        self.clear_histogram_highlight();
                    }
                }
            });
            ui.separator();

            let available_size = ui.available_size().max(Vec2::splat(1.0));
            let (canvas_rect, response) =
                ui.allocate_exact_size(available_size, Sense::click_and_drag());
            let painter = ui.painter_at(canvas_rect);
            painter.rect_filled(canvas_rect, 0.0, ui.visuals().extreme_bg_color);

            self.ensure_selected_image_loaded(ui.ctx());
            self.ensure_histogram_highlight_texture(ui.ctx());

            if let Some(error) = &self.viewer_error {
                painter.text(
                    canvas_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    error,
                    egui::TextStyle::Body.resolve(ui.style()),
                    ui.visuals().error_fg_color,
                );
                return;
            }

            let Some((texture_id, image_size, cellular_texture_id, highlight_texture_id)) =
                self.displayed_image.as_ref().map(|displayed| {
                    (
                        displayed.texture.id(),
                        displayed.size,
                        displayed
                            .cellular_automata_texture
                            .as_ref()
                            .map(|texture| texture.id()),
                        displayed
                            .highlight_texture
                            .as_ref()
                            .map(|texture| texture.id()),
                    )
                })
            else {
                painter.text(
                    canvas_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Add an image to begin",
                    egui::TextStyle::Heading.resolve(ui.style()),
                    ui.visuals().weak_text_color(),
                );
                return;
            };

            if self.viewer_needs_fit {
                let fit_zoom = (canvas_rect.width() / image_size.x)
                    .min(canvas_rect.height() / image_size.y)
                    .clamp(0.05, 16.0);
                self.viewer_zoom = fit_zoom;
                self.viewer_pan = Vec2::ZERO;
                self.viewer_needs_fit = false;
            }

            if response.dragged_by(PointerButton::Secondary) {
                self.viewer_pan += response.drag_delta();
            }

            if response.hovered() {
                let (scroll_delta, zoom_delta, pointer_pos) = ui.input(|input| {
                    (
                        input.smooth_scroll_delta().y,
                        input.zoom_delta(),
                        input.pointer.latest_pos(),
                    )
                });

                let scroll_zoom_delta = (scroll_delta * 0.002).exp();
                let combined_zoom_delta = scroll_zoom_delta * zoom_delta;
                if combined_zoom_delta != 1.0 {
                    self.zoom_around_pointer(
                        combined_zoom_delta,
                        pointer_pos.unwrap_or(canvas_rect.center()),
                        canvas_rect,
                    );
                }
            }

            let draw_size = image_size * self.viewer_zoom;
            let image_rect =
                Rect::from_center_size(canvas_rect.center() + self.viewer_pan, draw_size);

            if response.drag_started_by(PointerButton::Primary) {
                self.image_selection_start = response
                    .interact_pointer_pos()
                    .filter(|pointer| image_rect.contains(*pointer));
            }

            let image_selection_rect = if let (Some(start), Some(pointer)) =
                (self.image_selection_start, response.interact_pointer_pos())
            {
                let selection_rect = Rect::from_two_pos(start, pointer).intersect(image_rect);
                if (response.dragged_by(PointerButton::Primary)
                    || response.drag_stopped_by(PointerButton::Primary))
                    && let Some(displayed) = self.displayed_image.as_ref()
                {
                    let bins = displayed.bins_in_pointer_rect(image_rect, selection_rect);
                    if !bins.is_empty() {
                        self.select_histogram_bins(bins);
                        ui.ctx().request_repaint();
                    }
                }
                Some(selection_rect)
            } else {
                None
            };

            if response.drag_stopped_by(PointerButton::Primary) {
                self.image_selection_start = None;
            }

            let clicked_image_bin = if response.clicked_by(PointerButton::Primary) {
                response.interact_pointer_pos().and_then(|pointer| {
                    self.displayed_image
                        .as_ref()
                        .and_then(|displayed| displayed.bin_at_pointer(image_rect, pointer))
                })
            } else {
                None
            };
            if let Some(bin) = clicked_image_bin {
                self.select_single_histogram_bin(bin);
                ui.ctx().request_repaint();
            } else if response.clicked_by(PointerButton::Primary)
                && response
                    .interact_pointer_pos()
                    .is_some_and(|pointer| !image_rect.contains(pointer))
            {
                self.clear_histogram_highlight();
                ui.ctx().request_repaint();
            }

            painter.image(
                texture_id,
                image_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );

            if let Some(cellular_texture_id) = cellular_texture_id {
                painter.image(
                    cellular_texture_id,
                    image_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }

            if let Some(highlight_texture_id) = highlight_texture_id {
                painter.image(
                    highlight_texture_id,
                    image_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }

            if let Some(selection_rect) = image_selection_rect {
                painter.rect_filled(
                    selection_rect,
                    0.0,
                    Color32::from_rgba_unmultiplied(255, 180, 60, 28),
                );
                painter.rect_stroke(
                    selection_rect,
                    0.0,
                    egui::Stroke::new(1.0, Color32::from_rgb(255, 180, 60)),
                    egui::StrokeKind::Inside,
                );
            }

            painter.rect_stroke(
                canvas_rect,
                0.0,
                ui.visuals().widgets.noninteractive.bg_stroke,
                egui::StrokeKind::Inside,
            );
        });
    }

    fn zoom_around_pointer(&mut self, zoom_delta: f32, pointer: Pos2, canvas_rect: Rect) {
        let old_zoom = self.viewer_zoom;
        let new_zoom = (old_zoom * zoom_delta).clamp(0.02, 64.0);
        if (new_zoom - old_zoom).abs() <= f32::EPSILON {
            return;
        }

        let pointer_delta = pointer - canvas_rect.center();
        let zoom_ratio = new_zoom / old_zoom;
        self.viewer_pan = pointer_delta - (pointer_delta - self.viewer_pan) * zoom_ratio;
        self.viewer_zoom = new_zoom;
    }

    fn ensure_histogram_highlight_texture(&mut self, ctx: &egui::Context) {
        if self.selected_histogram_bins.is_empty() {
            if let Some(displayed) = self.displayed_image.as_mut() {
                displayed.highlight_bins.clear();
                displayed.highlight_texture = None;
            }
            return;
        }

        let selected_bins = self.selected_bins_vec();
        let selected_bin_set = self.selected_histogram_bins.clone();

        let Some(displayed) = self.displayed_image.as_mut() else {
            return;
        };

        if selected_bin_set
            .iter()
            .any(|bin| *bin >= displayed.bin_count)
            || displayed.bin_indices.len() != displayed.pixel_count()
        {
            displayed.highlight_bins.clear();
            displayed.highlight_texture = None;
            self.selected_histogram_bins.clear();
            return;
        }

        if displayed.highlight_bins == selected_bins && displayed.highlight_texture.is_some() {
            return;
        }

        let rgba = histogram_highlight_rgba(&displayed.bin_indices, &selected_bin_set);
        let size = [displayed.size.x as usize, displayed.size.y as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &rgba);
        displayed.highlight_texture = Some(ctx.load_texture(
            format!(
                "image {} histogram {} bins highlight",
                displayed.image_index,
                selected_bins.len()
            ),
            color_image,
            TextureOptions::NEAREST,
        ));
        displayed.highlight_bins = selected_bins;
    }

    fn toggle_histogram_highlight(&mut self, bin: usize) {
        if self.selected_histogram_bins.len() == 1 && self.selected_histogram_bins.contains(&bin) {
            self.clear_histogram_highlight();
            return;
        }

        self.select_single_histogram_bin(bin);
    }

    fn select_single_histogram_bin(&mut self, bin: usize) {
        self.select_histogram_bins([bin]);
    }

    fn select_histogram_bin_range(&mut self, start: usize, end: usize) {
        let low = start.min(end);
        let high = start.max(end);
        self.select_histogram_bins(low..=high);
    }

    fn clear_histogram_highlight(&mut self) {
        self.selected_histogram_bins.clear();
        self.image_selection_start = None;
        self.histogram_selection_start_bin = None;
        if let Some(displayed) = self.displayed_image.as_mut() {
            displayed.highlight_bins.clear();
            displayed.highlight_texture = None;
        }
    }

    fn select_histogram_bins(&mut self, bins: impl IntoIterator<Item = usize>) {
        let next_bins = bins.into_iter().collect::<BTreeSet<_>>();
        if self.selected_histogram_bins == next_bins {
            return;
        }

        self.selected_histogram_bins = next_bins;
        if let Some(displayed) = self.displayed_image.as_mut() {
            displayed.highlight_bins.clear();
            displayed.highlight_texture = None;
        }
    }

    fn selected_bins_vec(&self) -> Vec<usize> {
        self.selected_histogram_bins.iter().copied().collect()
    }

    fn show_image_picker_modal(&mut self, ctx: &egui::Context) {
        if !self.show_image_picker {
            return;
        }

        let mut close_modal = false;
        let response = Modal::new(Id::new("add_image_modal")).show(ctx, |ui| {
            ui.set_min_width(520.0);
            ui.heading("Add Images");
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                if ui.button("Up").clicked() {
                    self.navigate_to_parent_directory();
                }
                ui.label("Location");
                ui.text_edit_singleline(&mut self.image_picker_dir);
                if ui.button("Go").clicked() {
                    let directory = PathBuf::from(self.image_picker_dir.trim());
                    self.navigate_to_directory(directory);
                }
            });

            if let Some(error) = &self.image_picker_error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }

            ui.add_space(8.0);
            let primary_down = ui.input(|input| input.pointer.button_down(PointerButton::Primary));
            if !primary_down {
                self.picker_drag_selection = None;
            }

            ui.horizontal(|ui| {
                if ui.button("Select all").clicked() {
                    for candidate in &mut self.image_candidates {
                        if candidate.kind == ImagePickerEntryKind::SupportedImage {
                            candidate.selected = true;
                        }
                    }
                }
                if ui.button("Clear").clicked() {
                    for candidate in &mut self.image_candidates {
                        candidate.selected = false;
                    }
                }
            });

            ui.separator();
            ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                if self.image_candidates.is_empty() {
                    ui.label("No folders or supported images found in this folder.");
                } else {
                    let mut directory_to_open = None;
                    let pointer_pos = ui.input(|input| input.pointer.latest_pos());

                    for candidate in &mut self.image_candidates {
                        match candidate.kind {
                            ImagePickerEntryKind::Directory => {
                                let response = ui
                                    .button(format!("[dir] {}", candidate.name))
                                    .on_hover_text(candidate.path.display().to_string());
                                if response.clicked() && self.picker_drag_selection.is_none() {
                                    directory_to_open = Some(candidate.path.clone());
                                }
                            }
                            ImagePickerEntryKind::SupportedImage => {
                                let row_width = ui.available_width();
                                let row_height = ui.spacing().interact_size.y;
                                let response = ui
                                    .add(
                                        Button::selectable(
                                            candidate.selected,
                                            format!("[img] {}", candidate.name),
                                        )
                                        .sense(Sense::click_and_drag())
                                        .min_size(Vec2::new(row_width, row_height)),
                                    )
                                    .on_hover_text(candidate.path.display().to_string());

                                if response.clicked() {
                                    candidate.selected = !candidate.selected;
                                }

                                if response.drag_started_by(PointerButton::Primary) {
                                    let target = !candidate.selected;
                                    candidate.selected = target;
                                    self.picker_drag_selection = Some(target);
                                }

                                if primary_down
                                    && pointer_pos.is_some_and(|pos| response.rect.contains(pos))
                                    && let Some(target) = self.picker_drag_selection
                                {
                                    candidate.selected = target;
                                }
                            }
                        }
                    }

                    if let Some(directory) = directory_to_open {
                        self.navigate_to_directory(directory);
                    }
                }
            });

            ui.separator();
            ui.horizontal(|ui| {
                let has_selection = self.image_candidates.iter().any(|candidate| {
                    candidate.kind == ImagePickerEntryKind::SupportedImage && candidate.selected
                });
                if ui
                    .add_enabled(has_selection, Button::new("Add selected"))
                    .clicked()
                {
                    let paths = self
                        .image_candidates
                        .iter()
                        .filter(|candidate| {
                            candidate.kind == ImagePickerEntryKind::SupportedImage
                                && candidate.selected
                        })
                        .map(|candidate| candidate.path.clone())
                        .collect();
                    self.add_image_paths(paths);
                    close_modal = true;
                }

                if ui.button("Cancel").clicked() {
                    close_modal = true;
                }
            });
        });

        if response.should_close() || close_modal {
            self.show_image_picker = false;
            self.picker_drag_selection = None;
        }
    }

    fn scan_image_directory(&mut self) {
        self.image_picker_error = None;
        self.image_candidates.clear();

        let directory = PathBuf::from(self.image_picker_dir.trim());
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) => {
                self.image_picker_error = Some(format!("Could not read folder: {}", error));
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Untitled folder")
                    .to_owned();

                self.image_candidates.push(ImageCandidate {
                    path,
                    name,
                    selected: false,
                    kind: ImagePickerEntryKind::Directory,
                });
                continue;
            }

            if path.is_file() && Self::is_supported_image_path(&path) {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Untitled image")
                    .to_owned();

                self.image_candidates.push(ImageCandidate {
                    path,
                    name,
                    selected: false,
                    kind: ImagePickerEntryKind::SupportedImage,
                });
            }
        }

        self.image_candidates.sort_by(|left, right| {
            left.kind
                .sort_order()
                .cmp(&right.kind.sort_order())
                .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
        });
    }

    fn navigate_to_directory(&mut self, directory: PathBuf) {
        if !directory.is_dir() {
            self.image_picker_error = Some(format!("Not a folder: {}", directory.display()));
            return;
        }

        self.image_picker_dir = directory
            .canonicalize()
            .unwrap_or(directory)
            .display()
            .to_string();
        self.scan_image_directory();
    }

    fn navigate_to_parent_directory(&mut self) {
        let directory = PathBuf::from(self.image_picker_dir.trim());
        if let Some(parent) = directory.parent() {
            self.navigate_to_directory(parent.to_path_buf());
        }
    }

    fn add_image_paths(&mut self, paths: Vec<PathBuf>) {
        self.image_picker_error = None;
        let mut first_added = None;

        for path in paths {
            if let Some(existing_index) = self.images.iter().position(|image| image.path == path) {
                first_added.get_or_insert(existing_index);
                continue;
            }

            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Untitled image")
                .to_owned();
            let index = self.images.len();
            self.images.push(ImageFile { path, name });
            first_added.get_or_insert(index);
        }

        if let Some(index) = first_added {
            self.selected_image = Some(index);
            self.displayed_image = None;
            self.viewer_error = None;
            self.viewer_needs_fit = true;
            self.histogram = None;
            self.histogram_error = None;
            self.reset_histogram_view = true;
            self.clear_histogram_highlight();
            self.applied_denoise = None;
            self.invert_values = false;
            self.applied_invert = false;
            self.equalization_strength = 0.0;
            self.applied_equalization = None;
            self.applied_gamma = None;
            self.rotation = ImageRotation::None;
            self.applied_rotation = ImageRotation::None;
            self.cellular_automata_settings = CellularAutomataSettings::default();
            self.applied_cellular_automata = None;
            self.cellular_kernel_preset = CellularKernelPreset::Disk;
            self.cellular_kernel_angle = 0.0;
            self.processing_job = None;
        }
    }

    fn create_intensity_histogram(&mut self) {
        self.histogram = None;
        self.histogram_error = None;

        let Some(path) = self
            .selected_image
            .and_then(|index| self.images.get(index))
            .map(|image| image.path.clone())
        else {
            self.histogram_error = Some("Select an image before creating a histogram.".to_owned());
            return;
        };

        if self.applied_rotation != ImageRotation::None
            || self.applied_denoise.is_some()
            || self.applied_invert
            || self.applied_equalization.is_some()
            || self.applied_gamma.is_some()
        {
            let mut configuration = self.applied_experiment_configuration();
            configuration.cellular_automata = None;
            match processed_image_from_path(&path, &configuration) {
                Ok(processed) => {
                    self.histogram = Some(processed.histogram);
                    self.reset_histogram_view = true;
                }
                Err(error) => {
                    self.histogram_error = Some(error);
                    self.reset_histogram_view = true;
                }
            }
        } else {
            match intensity_histogram_from_path(&path, self.histogram_bin_count) {
                Ok(histogram) => {
                    self.histogram = Some(histogram);
                    self.reset_histogram_view = true;
                }
                Err(error) => {
                    self.histogram_error = Some(error);
                    self.reset_histogram_view = true;
                }
            }
        }
    }

    fn current_denoise_settings(&self) -> Option<DenoiseSettings> {
        (self.denoise_method != DenoiseMethod::None).then_some(DenoiseSettings {
            method: self.denoise_method,
            kernel_size: self.denoise_kernel_size,
        })
    }

    fn applied_experiment_configuration(&self) -> ExperimentConfiguration {
        ExperimentConfiguration {
            schema_version: 1,
            histogram_bin_count: self.histogram_bin_count,
            rotation: self.applied_rotation,
            denoise: self.applied_denoise,
            invert: self.applied_invert,
            equalization: self.applied_equalization,
            gamma: self.applied_gamma,
            cellular_automata: self.applied_cellular_automata.clone(),
        }
    }

    fn has_image_augmentation_state(&self) -> bool {
        self.applied_denoise.is_some()
            || self.applied_rotation != ImageRotation::None
            || self.rotation != ImageRotation::None
            || self.applied_invert
            || self.invert_values
            || self.applied_equalization.is_some()
            || self.equalization_strength != 0.0
            || self.applied_gamma.is_some()
            || self.applied_cellular_automata.is_some()
            || self.cellular_automata_settings != CellularAutomataSettings::default()
            || self.denoise_method != DenoiseMethod::None
            || self.denoise_kernel_size != 3
            || self.gamma_value != 1.0
    }

    fn apply_denoising(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram_error = None;

        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.histogram_error = Some("Select an image before applying denoising.".to_owned());
            return;
        };

        let mut parameters = self.applied_experiment_configuration();
        parameters.denoise = self.current_denoise_settings();
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Denoise(parameters),
        );
    }

    fn apply_inversion(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram_error = None;

        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.histogram_error = Some("Select an image before applying inversion.".to_owned());
            return;
        };

        let mut parameters = self.applied_experiment_configuration();
        parameters.invert = self.invert_values;
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Invert(parameters),
        );
    }

    fn apply_equalization(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram_error = None;

        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.histogram_error = Some("Select an image before applying equalization.".to_owned());
            return;
        };

        let equalization = (self.equalization_strength > 0.0).then_some(self.equalization_strength);
        let mut parameters = self.applied_experiment_configuration();
        parameters.equalization = equalization;
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Equalize(parameters),
        );
    }

    fn apply_gamma_adjustment(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram_error = None;

        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.histogram_error = Some("Select an image before applying gamma.".to_owned());
            return;
        };

        let mut parameters = self.applied_experiment_configuration();
        parameters.gamma = Some(self.gamma_value);
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Gamma(parameters),
        );
    }

    fn apply_cellular_automata(&mut self, ctx: &egui::Context, enabled: bool) {
        self.viewer_error = None;
        self.histogram_error = None;

        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.histogram_error =
                Some("Select an image before applying the feature filter.".to_owned());
            return;
        };

        let cellular_automata = enabled.then(|| self.cellular_automata_settings.clone());
        let mut parameters = self.applied_experiment_configuration();
        parameters.cellular_automata = cellular_automata;
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::CellularAutomata(parameters),
        );
    }

    fn apply_rotation(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram_error = None;

        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.histogram_error = Some("Select an image before applying rotation.".to_owned());
            return;
        };

        let mut parameters = self.applied_experiment_configuration();
        parameters.rotation = self.rotation;
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Rotation(parameters),
        );
    }

    fn save_applied_configuration(&mut self) {
        self.configuration_notice = None;
        self.configuration_error = None;
        let configuration = self.applied_experiment_configuration();
        let path = PathBuf::from(self.configuration_path.trim());
        if path.as_os_str().is_empty() {
            self.configuration_error = Some("Enter a configuration path.".to_owned());
            return;
        }

        let result = serde_json::to_string_pretty(&configuration)
            .map(|json| format!("{json}\n"))
            .map_err(|error| format!("Could not encode configuration: {error}"))
            .and_then(|json| {
                fs::write(&path, json)
                    .map_err(|error| format!("Could not save {}: {error}", path.display()))
            });

        match result {
            Ok(()) => {
                self.saved_configuration = Some(configuration);
                self.configuration_notice = Some(format!("Saved {}", path.display()));
            }
            Err(error) => self.configuration_error = Some(error),
        }
    }

    fn load_configuration(&mut self) {
        self.configuration_notice = None;
        self.configuration_error = None;
        let path = PathBuf::from(self.configuration_path.trim());
        if path.as_os_str().is_empty() {
            self.configuration_error = Some("Enter a configuration path.".to_owned());
            return;
        }

        let configuration = fs::read_to_string(&path)
            .map_err(|error| format!("Could not read {}: {error}", path.display()))
            .and_then(|json| {
                serde_json::from_str::<ExperimentConfiguration>(&json)
                    .map_err(|error| format!("Could not decode {}: {error}", path.display()))
            })
            .and_then(|configuration| {
                validate_experiment_configuration(&configuration)?;
                Ok(configuration)
            });

        match configuration {
            Ok(configuration) => {
                self.sync_controls_from_configuration(&configuration);
                self.saved_configuration = Some(configuration);
                self.configuration_notice = Some(format!("Loaded {}", path.display()));
            }
            Err(error) => self.configuration_error = Some(error),
        }
    }

    fn apply_saved_configuration(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram_error = None;
        self.configuration_error = None;

        let Some(configuration) = self.saved_configuration.clone() else {
            self.configuration_error = Some("Load or save a configuration first.".to_owned());
            return;
        };
        let Some((image_index, path, name)) = self.selected_image.and_then(|index| {
            self.images
                .get(index)
                .map(|image| (index, image.path.clone(), image.name.clone()))
        }) else {
            self.configuration_error =
                Some("Select an image before applying a configuration.".to_owned());
            return;
        };

        self.sync_controls_from_configuration(&configuration);
        let parameters = configuration;
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Configuration(parameters),
        );
    }

    fn sync_controls_from_configuration(&mut self, configuration: &ExperimentConfiguration) {
        self.histogram_bin_count = configuration.histogram_bin_count;
        self.reset_cellular_response_view = true;
        self.rotation = configuration.rotation;
        self.denoise_method = configuration
            .denoise
            .map(|settings| settings.method)
            .unwrap_or(DenoiseMethod::None);
        self.denoise_kernel_size = configuration
            .denoise
            .map(|settings| settings.kernel_size)
            .unwrap_or(3);
        self.invert_values = configuration.invert;
        self.equalization_strength = configuration.equalization.unwrap_or(0.0);
        self.gamma_value = configuration.gamma.unwrap_or(1.0);
        self.cellular_automata_settings =
            configuration.cellular_automata.clone().unwrap_or_default();
        self.cellular_kernel_preset = CellularKernelPreset::Custom;
        self.cellular_kernel_angle = 0.0;
    }

    fn reset_image_augmentations(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram = None;
        self.histogram_error = None;
        self.reset_histogram_view = true;
        self.reset_cellular_response_view = true;
        self.clear_histogram_highlight();
        self.displayed_image = None;
        self.viewer_needs_fit = true;
        self.applied_denoise = None;
        self.invert_values = false;
        self.applied_invert = false;
        self.equalization_strength = 0.0;
        self.applied_equalization = None;
        self.applied_gamma = None;
        self.rotation = ImageRotation::None;
        self.applied_rotation = ImageRotation::None;
        self.cellular_automata_settings = CellularAutomataSettings::default();
        self.applied_cellular_automata = None;
        self.cellular_kernel_preset = CellularKernelPreset::Disk;
        self.cellular_kernel_angle = 0.0;
        self.denoise_method = DenoiseMethod::None;
        self.denoise_kernel_size = 3;
        self.gamma_value = 1.0;
        self.processing_job = None;
        ctx.request_repaint();
    }

    fn start_processing_job(
        &mut self,
        ctx: &egui::Context,
        image_index: usize,
        path: PathBuf,
        image_name: String,
        kind: ProcessingKind,
    ) {
        if self.processing_job.is_some() {
            return;
        }

        let id = self.next_processing_job_id;
        self.next_processing_job_id += 1;
        let (sender, receiver) = mpsc::channel();
        let repaint_ctx = ctx.clone();
        let job_kind = kind.clone();

        thread::spawn(move || {
            let result = processed_image_from_path(&path, kind.parameters());
            let _ = sender.send(ProcessingResult {
                id,
                image_index,
                image_name,
                kind,
                result,
            });
            repaint_ctx.request_repaint();
        });

        self.processing_job = Some(ProcessingJob {
            id,
            image_index,
            receiver,
            kind: job_kind,
        });
    }

    fn poll_processing_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.processing_job.as_ref() else {
            return;
        };

        let Ok(result) = job.receiver.try_recv() else {
            ctx.request_repaint();
            return;
        };

        let active_job = self.processing_job.take();
        if active_job
            .as_ref()
            .is_none_or(|job| job.id != result.id || job.image_index != result.image_index)
        {
            return;
        }

        if self.selected_image != Some(result.image_index) {
            return;
        }

        match result.result {
            Ok(processed) => {
                let texture_name = result.kind.texture_name(&result.image_name);
                let size = processed.size;
                let histogram = processed.histogram;
                let bin_indices = processed.bin_indices;
                let bin_count = processed.bin_count;
                let cellular_automata_summary = processed.cellular_automata_summary;
                let cellular_automata_texture =
                    processed.cellular_automata_overlay.map(|overlay| {
                        ctx.load_texture(
                            format!("{texture_name} cellular feature overlay"),
                            overlay,
                            TextureOptions::LINEAR,
                        )
                    });
                let texture =
                    ctx.load_texture(texture_name, processed.color_image, TextureOptions::LINEAR);
                let should_fit = self.displayed_image.is_none();
                self.displayed_image = Some(DisplayedImage {
                    image_index: result.image_index,
                    size,
                    texture,
                    bin_indices,
                    bin_count,
                    highlight_bins: Vec::new(),
                    highlight_texture: None,
                    cellular_automata_texture,
                    cellular_automata_summary,
                });
                if self
                    .selected_histogram_bins
                    .iter()
                    .any(|bin| *bin >= bin_count)
                {
                    self.clear_histogram_highlight();
                }
                self.histogram = Some(histogram);
                self.reset_histogram_view = true;
                result.kind.apply_to(self);
                if should_fit {
                    self.viewer_needs_fit = true;
                }
            }
            Err(error) => {
                self.viewer_error = Some(error.clone());
                self.histogram_error = Some(error);
            }
        }
    }

    fn show_histogram_zoom_controls(&mut self, ui: &mut Ui, _histogram: &IntensityHistogram) {
        ui.horizontal(|ui| {
            if ui
                .button("Reset")
                .on_hover_text("Restores the histogram plot bounds.")
                .clicked()
            {
                self.reset_histogram_view = true;
            }
            if !self.selected_histogram_bins.is_empty()
                && ui
                    .button("Clear Selection")
                    .on_hover_text("Clears selected intensity bins and image overlays.")
                    .clicked()
            {
                self.clear_histogram_highlight();
            }
        });

        match self.selected_bins_vec().as_slice() {
            [] => {}
            [bin] => {
                ui.label(format!("Highlighted bin {bin}"));
            }
            bins => {
                ui.label(format!("Highlighted {} bins", bins.len()));
            }
        }
    }

    fn show_histogram_plot(&mut self, ui: &mut Ui, histogram: &IntensityHistogram) {
        let peak_count = histogram.bins.iter().copied().max().unwrap_or(0);
        let bars = histogram
            .bins
            .iter()
            .enumerate()
            .map(|(index, count)| {
                let fill = if self.selected_histogram_bins.contains(&index) {
                    Color32::from_rgb(255, 180, 60)
                } else {
                    Color32::from_rgb(120, 170, 220)
                };
                Bar::new(index as f64, *count as f64).width(1.0).fill(fill)
            })
            .collect();
        let highlight_extensions = self
            .selected_histogram_bins
            .iter()
            .filter_map(|bin| {
                let count = histogram.bins.get(*bin).copied()?;
                (peak_count > count).then(|| {
                    Bar::new(*bin as f64, (peak_count - count) as f64)
                        .base_offset(count as f64)
                        .width(1.0)
                        .fill(Color32::from_rgba_unmultiplied(255, 180, 60, 70))
                })
            })
            .collect::<Vec<_>>();

        let mut plot = Plot::new("intensity_histogram")
            .height(180.0)
            .allow_drag(true)
            .pan_pointer_button(PointerButton::Secondary)
            .allow_scroll(false)
            .allow_zoom(true)
            .allow_boxed_zoom(false)
            .allow_axis_zoom_drag(false);

        if self.reset_histogram_view {
            plot = plot.reset();
            self.reset_histogram_view = false;
        }

        let response = plot.show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("Intensity", bars).width(1.0));
            if !highlight_extensions.is_empty() {
                plot_ui.bar_chart(
                    BarChart::new("Selected bin extent", highlight_extensions).width(1.0),
                );
            }
            let pointer_bin = plot_ui.pointer_coordinate().and_then(|point| {
                let bin = (point.x + 0.5).floor() as isize;
                (0..histogram.bins.len() as isize)
                    .contains(&bin)
                    .then_some(bin as usize)
            });
            let clicked_bin = if plot_ui.response().clicked() {
                pointer_bin
            } else {
                None
            };
            let drag_start_bin = plot_ui
                .response()
                .drag_started_by(PointerButton::Primary)
                .then_some(pointer_bin)
                .flatten();
            let drag_current_bin = plot_ui
                .response()
                .dragged_by(PointerButton::Primary)
                .then_some(pointer_bin)
                .flatten();
            let drag_stopped = plot_ui.response().drag_stopped_by(PointerButton::Primary);

            zoom_hovered_plot_with_scroll(plot_ui);

            (clicked_bin, drag_start_bin, drag_current_bin, drag_stopped)
        });

        let (clicked_bin, drag_start_bin, drag_current_bin, drag_stopped) = response.inner;
        if let Some(bin) = drag_start_bin {
            self.histogram_selection_start_bin = Some(bin);
        }
        if let (Some(start), Some(current)) = (self.histogram_selection_start_bin, drag_current_bin)
        {
            self.select_histogram_bin_range(start, current);
        }
        if drag_stopped {
            self.histogram_selection_start_bin = None;
        } else if let Some(bin) = clicked_bin {
            self.toggle_histogram_highlight(bin);
        }
    }

    fn ensure_selected_image_loaded(&mut self, ctx: &egui::Context) {
        let Some(image_index) = self.selected_image else {
            self.displayed_image = None;
            self.viewer_error = None;
            return;
        };

        if self
            .displayed_image
            .as_ref()
            .is_some_and(|displayed| displayed.image_index == image_index)
        {
            return;
        }

        let Some(path) = self.images.get(image_index).map(|image| image.path.clone()) else {
            self.displayed_image = None;
            self.viewer_error = None;
            return;
        };

        self.displayed_image = None;
        self.viewer_error = None;

        if self.processing_job.is_some() {
            return;
        }

        let Some(name) = self.images.get(image_index).map(|image| image.name.clone()) else {
            self.viewer_error = Some("Could not find selected image.".to_owned());
            return;
        };
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Preview(self.applied_experiment_configuration()),
        );
    }

    fn is_supported_image_path(path: &Path) -> bool {
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| {
                matches!(
                    extension.to_lowercase().as_str(),
                    "png" | "jpg" | "jpeg" | "tif" | "tiff"
                )
            })
            .unwrap_or(false)
    }
}

impl ImagePickerEntryKind {
    fn sort_order(self) -> usize {
        match self {
            Self::Directory => 0,
            Self::SupportedImage => 1,
        }
    }
}

impl ProcessingKind {
    fn label(&self) -> &'static str {
        match self {
            Self::Preview(_) => "preview",
            Self::Rotation(_) => "rotation",
            Self::Denoise(_) => "denoising",
            Self::Invert(_) => "inversion",
            Self::Equalize(_) => "equalization",
            Self::Gamma(_) => "gamma adjustment",
            Self::CellularAutomata(_) => "cellular feature filter",
            Self::Configuration(_) => "saved configuration",
        }
    }

    fn texture_name(&self, image_name: &str) -> String {
        match self {
            Self::Preview(_) => format!("{image_name} preview"),
            Self::Rotation(parameters) => {
                format!("{image_name} rotated {}", parameters.rotation.degrees())
            }
            Self::Denoise(parameters) => {
                let stage_name = parameters
                    .denoise
                    .map(|settings| settings.method.label())
                    .unwrap_or("raw");
                format!("{image_name} {stage_name} denoise")
            }
            Self::Invert(parameters) => {
                let stage_name = if parameters.invert {
                    "inverted"
                } else {
                    "uninverted"
                };
                format!("{image_name} {stage_name} values")
            }
            Self::Equalize(parameters) => {
                let stage_name = parameters
                    .equalization
                    .map(|strength| format!("equalized {:.0}%", strength * 100.0))
                    .unwrap_or_else(|| "unequalized".to_owned());
                format!("{image_name} {stage_name}")
            }
            Self::Gamma(parameters) => {
                format!("{image_name} gamma {:.3}", parameters.gamma.unwrap_or(1.0))
            }
            Self::CellularAutomata(parameters) => {
                let stage_name = if parameters.cellular_automata.is_some() {
                    "cellular feature filter"
                } else {
                    "without cellular feature filter"
                };
                format!("{image_name} {stage_name}")
            }
            Self::Configuration(_) => format!("{image_name} saved configuration"),
        }
    }

    fn parameters(&self) -> &ExperimentConfiguration {
        match self {
            Self::Preview(parameters)
            | Self::Rotation(parameters)
            | Self::Denoise(parameters)
            | Self::Invert(parameters)
            | Self::Equalize(parameters)
            | Self::Gamma(parameters)
            | Self::CellularAutomata(parameters)
            | Self::Configuration(parameters) => parameters,
        }
    }

    fn apply_to(self, app: &mut VCEMApp) {
        match self {
            Self::Preview(_) => {}
            Self::Rotation(parameters) => {
                app.applied_rotation = parameters.rotation;
                app.rotation = parameters.rotation;
                app.viewer_needs_fit = true;
            }
            Self::Denoise(parameters) => {
                app.applied_denoise = parameters.denoise;
            }
            Self::Invert(parameters) => {
                app.applied_invert = parameters.invert;
                app.invert_values = parameters.invert;
            }
            Self::Equalize(parameters) => {
                app.applied_equalization = parameters.equalization;
                app.equalization_strength = parameters.equalization.unwrap_or(0.0);
            }
            Self::Gamma(parameters) => {
                app.applied_gamma = parameters.gamma;
            }
            Self::CellularAutomata(parameters) => {
                app.applied_cellular_automata = parameters.cellular_automata;
            }
            Self::Configuration(parameters) => {
                parameters.apply_as_current(app);
                app.configuration_notice = Some("Applied saved configuration".to_owned());
                app.viewer_needs_fit = true;
            }
        }
    }
}

impl ExperimentConfiguration {
    fn apply_as_current(self, app: &mut VCEMApp) {
        app.histogram_bin_count = self.histogram_bin_count;
        app.applied_rotation = self.rotation;
        app.rotation = self.rotation;
        app.applied_denoise = self.denoise;
        app.denoise_method = self
            .denoise
            .map(|settings| settings.method)
            .unwrap_or(DenoiseMethod::None);
        app.denoise_kernel_size = self
            .denoise
            .map(|settings| settings.kernel_size)
            .unwrap_or(3);
        app.applied_invert = self.invert;
        app.invert_values = self.invert;
        app.applied_equalization = self.equalization;
        app.equalization_strength = self.equalization.unwrap_or(0.0);
        app.applied_gamma = self.gamma;
        app.gamma_value = self.gamma.unwrap_or(1.0);
        app.applied_cellular_automata = self.cellular_automata.clone();
        app.cellular_automata_settings = self.cellular_automata.unwrap_or_default();
        app.cellular_kernel_preset = CellularKernelPreset::Custom;
        app.cellular_kernel_angle = 0.0;
    }
}

impl CellularAutomataSeedMode {
    fn label(self) -> &'static str {
        match self {
            Self::Bright => "Bright features",
            Self::Dark => "Dark features",
        }
    }
}

impl CellularAutomataResponse {
    const ALL: [Self; 4] = [
        Self::Gaussian,
        Self::Band,
        Self::Threshold,
        Self::LowThreshold,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Gaussian => "Gaussian target",
            Self::Band => "Survival band",
            Self::Threshold => "High-pass threshold",
            Self::LowThreshold => "Low-pass threshold",
        }
    }
}

impl ImageRotation {
    const ALL: [Self; 4] = [
        Self::None,
        Self::Clockwise90,
        Self::HalfTurn,
        Self::Clockwise270,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::None => "0",
            Self::Clockwise90 => "90",
            Self::HalfTurn => "180",
            Self::Clockwise270 => "270",
        }
    }
}

impl CellularKernelPreset {
    const ALL: [Self; 6] = [
        Self::Disk,
        Self::Ring,
        Self::Line,
        Self::Cross,
        Self::Crescent,
        Self::Clear,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Custom => "Custom",
            Self::Disk => "Disk",
            Self::Ring => "Ring",
            Self::Line => "Line",
            Self::Cross => "Cross",
            Self::Crescent => "Crescent",
            Self::Clear => "Clear",
        }
    }
}

impl CellularKernelBrush {
    const ALL: [Self; 3] = [Self::Empty, Self::Positive, Self::Negative];

    fn weight(self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }

    fn hover_text(self) -> &'static str {
        match self {
            Self::Empty => "Paint empty kernel cells.",
            Self::Positive => "Paint green positive kernel cells.",
            Self::Negative => "Paint red negative kernel cells.",
        }
    }
}

fn resize_cellular_kernel(settings: &mut CellularAutomataSettings, new_size: usize) {
    let new_size = new_size.clamp(3, 15) | 1;
    let old_size = settings.kernel_size;
    let old_kernel = std::mem::take(&mut settings.kernel);
    let mut new_kernel = vec![0.0; new_size * new_size];
    let copy_size = old_size.min(new_size);
    let old_start = (old_size - copy_size) / 2;
    let new_start = (new_size - copy_size) / 2;

    for y in 0..copy_size {
        for x in 0..copy_size {
            new_kernel[(new_start + y) * new_size + new_start + x] =
                old_kernel[(old_start + y) * old_size + old_start + x];
        }
    }

    settings.kernel_size = new_size;
    settings.kernel = new_kernel;
}

fn apply_cellular_kernel_preset(
    settings: &mut CellularAutomataSettings,
    preset: CellularKernelPreset,
    angle_degrees: f32,
) {
    if preset == CellularKernelPreset::Custom {
        return;
    }

    let size = settings.kernel_size;
    let center = (size / 2) as f32;
    let radius = center.max(1.0);
    let angle = angle_degrees.to_radians();
    let direction = Vec2::new(angle.cos(), angle.sin());
    let normal = Vec2::new(-direction.y, direction.x);

    for y in 0..size {
        for x in 0..size {
            let point = Vec2::new(x as f32 - center, y as f32 - center);
            let distance = point.length();
            let along = point.dot(direction);
            let across = point.dot(normal);
            let active = match preset {
                CellularKernelPreset::Custom => false,
                CellularKernelPreset::Disk => distance <= radius,
                CellularKernelPreset::Ring => {
                    distance <= radius && distance >= (radius - 1.25).max(radius * 0.55)
                }
                CellularKernelPreset::Line => along.abs() <= radius && across.abs() <= 0.65,
                CellularKernelPreset::Cross => point.x.abs() <= 0.65 || point.y.abs() <= 0.65,
                CellularKernelPreset::Crescent => {
                    let notch_center = direction * (radius * 0.45);
                    distance <= radius && (point - notch_center).length() >= radius * 0.78
                }
                CellularKernelPreset::Clear => false,
            };
            settings.kernel[y * size + x] = if active { 1.0 } else { 0.0 };
        }
    }
}

fn kernel_brush_button(brush: CellularKernelBrush, selected: bool) -> Button<'static> {
    let mut button = kernel_weight_button(brush.weight());
    if selected {
        button = button.stroke(Stroke::new(2.0, Color32::from_rgb(230, 170, 64)));
    }
    button
}

fn kernel_weight_button(weight: f32) -> Button<'static> {
    let mut button = Button::new(kernel_weight_symbol(weight));
    if let Some(fill) = kernel_weight_fill(weight) {
        button = button.fill(fill);
    }
    button
}

fn kernel_weight_symbol(weight: f32) -> &'static str {
    if weight > 0.0 {
        "+"
    } else if weight < 0.0 {
        "-"
    } else {
        ""
    }
}

fn kernel_weight_fill(weight: f32) -> Option<Color32> {
    if weight > 0.0 {
        Some(Color32::from_rgb(27, 132, 122))
    } else if weight < 0.0 {
        Some(Color32::from_rgb(177, 71, 80))
    } else {
        None
    }
}

fn toggle_kernel_weight(weight: f32, brush: CellularKernelBrush) -> f32 {
    toggled_kernel_brush(weight, brush).weight()
}

fn toggled_kernel_brush(weight: f32, brush: CellularKernelBrush) -> CellularKernelBrush {
    if kernel_weights_match(weight, brush.weight()) {
        CellularKernelBrush::Empty
    } else {
        brush
    }
}

fn kernel_weights_match(left: f32, right: f32) -> bool {
    (left > 0.0 && right > 0.0) || (left < 0.0 && right < 0.0) || (left == 0.0 && right == 0.0)
}

fn next_kernel_weight(weight: f32, reverse: bool) -> f32 {
    if reverse {
        if weight > 0.0 {
            0.0
        } else if weight < 0.0 {
            1.0
        } else {
            -1.0
        }
    } else if weight > 0.0 {
        -1.0
    } else if weight < 0.0 {
        0.0
    } else {
        1.0
    }
}

fn cellular_response_preview(sensed: f32, settings: &CellularAutomataSettings) -> f32 {
    let center = settings.response_center.clamp(-1.0, 1.0);
    let width = settings.response_width.clamp(0.001, 2.0);

    match settings.response {
        CellularAutomataResponse::Gaussian => {
            let distance = (sensed - center) / width;
            2.0 * (-0.5 * distance * distance).exp() - 1.0
        }
        CellularAutomataResponse::Band => {
            let edge_distance = width - (sensed - center).abs();
            (edge_distance / (width * 0.15).max(0.005)).tanh()
        }
        CellularAutomataResponse::Threshold => ((sensed - center) / width.max(0.005)).tanh(),
        CellularAutomataResponse::LowThreshold => ((center - sensed) / width.max(0.005)).tanh(),
    }
}

fn zoom_hovered_plot_with_scroll(plot_ui: &mut PlotUi<'_>) {
    if !plot_ui.response().contains_pointer() {
        return;
    }

    let scroll_delta = plot_ui.ctx().input(|input| input.smooth_scroll_delta);
    if scroll_delta == Vec2::ZERO {
        return;
    }

    if scroll_delta.y != 0.0 {
        let zoom_factor = (scroll_delta.y * 0.002).exp();
        plot_ui.zoom_bounds_around_hovered(Vec2::splat(zoom_factor));
    }
    plot_ui
        .ctx()
        .input_mut(|input| input.smooth_scroll_delta = Vec2::ZERO);
}

fn validate_experiment_configuration(
    configuration: &ExperimentConfiguration,
) -> std::result::Result<(), String> {
    if configuration.schema_version != 1 {
        return Err(format!(
            "Unsupported configuration schema version {}.",
            configuration.schema_version
        ));
    }
    if !(2..=4096).contains(&configuration.histogram_bin_count) {
        return Err("Histogram bin count must be between 2 and 4096.".to_owned());
    }
    if let Some(denoise) = configuration.denoise
        && !matches!(denoise.kernel_size, 3 | 5 | 7)
    {
        return Err("Denoising kernel size must be 3, 5, or 7.".to_owned());
    }
    if configuration
        .equalization
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err("Equalization strength must be between 0 and 1.".to_owned());
    }
    if configuration
        .gamma
        .is_some_and(|value| !value.is_finite() || !(0.001..=4.0).contains(&value))
    {
        return Err("Gamma must be between 0.001 and 4.".to_owned());
    }

    if let Some(settings) = &configuration.cellular_automata {
        if !(3..=15).contains(&settings.kernel_size)
            || settings.kernel_size.is_multiple_of(2)
            || settings.kernel.len() != settings.kernel_size * settings.kernel_size
        {
            return Err("Cellular kernel must be an odd square from 3 x 3 to 15 x 15.".to_owned());
        }
        if settings.kernel.iter().any(|weight| !weight.is_finite())
            || !settings
                .kernel
                .iter()
                .any(|weight| weight.abs() > f32::EPSILON)
        {
            return Err("Cellular kernel must contain finite, non-zero weights.".to_owned());
        }
        let scalar_values = [
            settings.response_center,
            settings.response_width,
            settings.growth_rate,
            settings.decay_rate,
            settings.time_step,
        ];
        if scalar_values.iter().any(|value| !value.is_finite()) {
            return Err("Cellular response and evolution values must be finite.".to_owned());
        }
        if !(-1.0..=1.0).contains(&settings.response_center)
            || !(0.01..=1.0).contains(&settings.response_width)
            || !(0.0..=2.0).contains(&settings.growth_rate)
            || !(0.0..=2.0).contains(&settings.decay_rate)
            || !(0.01..=0.5).contains(&settings.time_step)
            || !(1..=128).contains(&settings.iterations)
        {
            return Err("Cellular settings fall outside the supported UI ranges.".to_owned());
        }
    }

    Ok(())
}

impl DisplayedImage {
    fn pixel_count(&self) -> usize {
        self.size.x as usize * self.size.y as usize
    }

    fn bin_at_pointer(&self, image_rect: Rect, pointer: Pos2) -> Option<usize> {
        if !image_rect.contains(pointer) || image_rect.width() <= 0.0 || image_rect.height() <= 0.0
        {
            return None;
        }

        let width = self.size.x as usize;
        let height = self.size.y as usize;
        if width == 0 || height == 0 || self.bin_indices.len() != width * height {
            return None;
        }

        let local = pointer - image_rect.min;
        let x = (local.x / image_rect.width() * width as f32)
            .floor()
            .clamp(0.0, (width - 1) as f32) as usize;
        let y = (local.y / image_rect.height() * height as f32)
            .floor()
            .clamp(0.0, (height - 1) as f32) as usize;

        self.bin_indices.get(y * width + x).copied()
    }

    fn bins_in_pointer_rect(&self, image_rect: Rect, selection_rect: Rect) -> BTreeSet<usize> {
        let mut bins = BTreeSet::new();
        if selection_rect.is_negative() || image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
            return bins;
        }

        let width = self.size.x as usize;
        let height = self.size.y as usize;
        if width == 0 || height == 0 || self.bin_indices.len() != width * height {
            return bins;
        }

        let min_x = ((selection_rect.min.x - image_rect.min.x) / image_rect.width() * width as f32)
            .floor()
            .clamp(0.0, (width - 1) as f32) as usize;
        let max_x = ((selection_rect.max.x - image_rect.min.x) / image_rect.width() * width as f32)
            .floor()
            .clamp(0.0, (width - 1) as f32) as usize;
        let min_y = ((selection_rect.min.y - image_rect.min.y) / image_rect.height()
            * height as f32)
            .floor()
            .clamp(0.0, (height - 1) as f32) as usize;
        let max_y = ((selection_rect.max.y - image_rect.min.y) / image_rect.height()
            * height as f32)
            .floor()
            .clamp(0.0, (height - 1) as f32) as usize;

        for y in min_y.min(max_y)..=min_y.max(max_y) {
            let row_start = y * width;
            for x in min_x.min(max_x)..=min_x.max(max_x) {
                if let Some(bin) = self.bin_indices.get(row_start + x) {
                    bins.insert(*bin);
                }
            }
        }

        bins
    }
}

fn histogram_highlight_rgba(bin_indices: &[usize], highlighted_bins: &BTreeSet<usize>) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(bin_indices.len() * 4);
    for bin in bin_indices {
        if highlighted_bins.contains(bin) {
            rgba.extend_from_slice(&[255, 184, 54, 128]);
        } else {
            rgba.extend_from_slice(&[0, 0, 0, 0]);
        }
    }
    rgba
}

impl DenoiseMethod {
    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Median => "Median",
            Self::Gaussian => "Gaussian",
        }
    }
}

impl App for VCEMApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut Frame) {
        self.poll_processing_job(ui.ctx());
        self.show_upper_panel(ui);
        self.show_left_panel(ui);
        self.show_right_panel(ui);
        self.show_bottom_panel(ui);
        self.show_central_panel(ui);
        self.show_image_picker_modal(ui.ctx());
    }
}

#[cfg(test)]
mod tests {
    use super::{CellularKernelBrush, toggle_kernel_weight, toggled_kernel_brush};

    #[test]
    fn selected_kernel_sign_toggles_off_and_back_on() {
        for brush in [CellularKernelBrush::Positive, CellularKernelBrush::Negative] {
            let selected_weight = brush.weight();
            let toggled_off = toggle_kernel_weight(selected_weight, brush);
            let toggled_back_on = toggle_kernel_weight(toggled_off, brush);

            assert_eq!(toggled_off, 0.0);
            assert_eq!(toggled_back_on, selected_weight);
        }
    }

    #[test]
    fn drag_uses_toggle_decided_by_its_starting_cell() {
        assert!(matches!(
            toggled_kernel_brush(1.0, CellularKernelBrush::Positive),
            CellularKernelBrush::Empty
        ));
        assert!(matches!(
            toggled_kernel_brush(0.0, CellularKernelBrush::Positive),
            CellularKernelBrush::Positive
        ));
        assert!(matches!(
            toggled_kernel_brush(-1.0, CellularKernelBrush::Positive),
            CellularKernelBrush::Positive
        ));
        assert!(matches!(
            toggled_kernel_brush(-1.0, CellularKernelBrush::Negative),
            CellularKernelBrush::Empty
        ));
        assert!(matches!(
            toggled_kernel_brush(0.0, CellularKernelBrush::Negative),
            CellularKernelBrush::Negative
        ));
        assert!(matches!(
            toggled_kernel_brush(1.0, CellularKernelBrush::Negative),
            CellularKernelBrush::Negative
        ));
    }
}
