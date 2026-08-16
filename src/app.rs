use eframe::{
    App, CreationContext, Frame,
    egui::{
        self, Button, CentralPanel, Color32, DragValue, Id, Modal, Panel, PointerButton, Pos2,
        Rect, ScrollArea, Sense, TextureOptions, Ui, Vec2,
    },
};
use egui_plot::{Bar, BarChart, Plot};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread,
};

use crate::analysis::{intensity_histogram_from_path, processed_image_from_path};
use crate::types::{
    DenoiseMethod, DenoiseSettings, DisplayedImage, ImageFile, IntensityHistogram, ProcessedImage,
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
    selected_histogram_bins: BTreeSet<usize>,
    image_selection_start: Option<Pos2>,
    histogram_selection_start_bin: Option<usize>,
    denoise_method: DenoiseMethod,
    denoise_kernel_size: usize,
    applied_denoise: Option<DenoiseSettings>,
    invert_values: bool,
    applied_invert: bool,
    equalization_strength: f64,
    applied_equalization: Option<f64>,
    gamma_value: f64,
    applied_gamma: Option<f64>,
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

#[derive(Clone, Copy)]
enum ProcessingKind {
    Preview {
        denoise: Option<DenoiseSettings>,
        invert: bool,
        equalization: Option<f64>,
        gamma: Option<f64>,
        bin_count: usize,
    },
    Denoise {
        denoise: Option<DenoiseSettings>,
        invert: bool,
        equalization: Option<f64>,
        gamma: Option<f64>,
        bin_count: usize,
    },
    Invert {
        denoise: Option<DenoiseSettings>,
        invert: bool,
        equalization: Option<f64>,
        gamma: Option<f64>,
        bin_count: usize,
    },
    Equalize {
        denoise: Option<DenoiseSettings>,
        invert: bool,
        equalization: Option<f64>,
        gamma: Option<f64>,
        bin_count: usize,
    },
    Gamma {
        denoise: Option<DenoiseSettings>,
        invert: bool,
        equalization: Option<f64>,
        gamma: f64,
        bin_count: usize,
    },
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
        let image_picker_dir = std::env::current_dir()
            .map(|path| path.display().to_string())
            .unwrap_or_default();

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
            selected_histogram_bins: BTreeSet::new(),
            image_selection_start: None,
            histogram_selection_start_bin: None,
            denoise_method: DenoiseMethod::None,
            denoise_kernel_size: 3,
            applied_denoise: None,
            invert_values: false,
            applied_invert: false,
            equalization_strength: 0.0,
            applied_equalization: None,
            gamma_value: 1.0,
            applied_gamma: None,
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
                ui.heading("Analysis");
                ui.separator();

                if let Some(image) = self.selected_image.and_then(|index| self.images.get(index)) {
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
                        .clicked()
                    {
                        self.reset_image_augmentations(ui.ctx());
                    }

                    ui.add_space(5.0);
                    ui.heading("Intensity Histogram");
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Bin count");
                        let response = ui.add(
                            DragValue::new(&mut self.histogram_bin_count)
                                .range(2..=4096)
                                .speed(1),
                        );
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
                        });

                    match self.denoise_method {
                        DenoiseMethod::None => {
                            ui.label("No denoising will be applied.");
                        }
                        DenoiseMethod::Median | DenoiseMethod::Gaussian => {
                            ui.horizontal(|ui| {
                                ui.label("Kernel");
                                ui.selectable_value(&mut self.denoise_kernel_size, 3, "3");
                                ui.selectable_value(&mut self.denoise_kernel_size, 5, "5");
                                ui.selectable_value(&mut self.denoise_kernel_size, 7, "7");
                            });
                        }
                    }

                    if ui
                        .add_enabled(!is_processing, Button::new("Apply Denoising"))
                        .clicked()
                    {
                        self.apply_denoising(ui.ctx());
                    }

                    ui.add_space(8.0);
                    ui.heading("Inversion");
                    ui.separator();
                    ui.checkbox(&mut self.invert_values, "Invert values");
                    if ui
                        .add_enabled(
                            !is_processing && self.invert_values != self.applied_invert,
                            Button::new("Apply Inversion"),
                        )
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
                        );
                    });
                    if ui
                        .add_enabled(
                            !is_processing
                                && Some(self.equalization_strength) != self.applied_equalization
                                && (self.equalization_strength > 0.0
                                    || self.applied_equalization.is_some()),
                            Button::new("Apply Equalization"),
                        )
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
                        );
                    });

                    if ui
                        .add_enabled(!is_processing, Button::new("Apply Gamma Adjustment"))
                        .clicked()
                    {
                        self.apply_gamma_adjustment(ui.ctx());
                    }
                } else {
                    ui.label("Analytical results live here.");
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

            let Some((texture_id, image_size, highlight_texture_id)) =
                self.displayed_image.as_ref().map(|displayed| {
                    (
                        displayed.texture.id(),
                        displayed.size,
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
                if response.dragged_by(PointerButton::Primary)
                    || response.drag_stopped_by(PointerButton::Primary)
                {
                    if let Some(displayed) = self.displayed_image.as_ref() {
                        let bins = displayed.bins_in_pointer_rect(image_rect, selection_rect);
                        if !bins.is_empty() {
                            self.select_histogram_bins(bins);
                            ui.ctx().request_repaint();
                        }
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
            self.applied_gamma = None;
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

        if self.applied_denoise.is_some()
            || self.applied_invert
            || self.applied_equalization.is_some()
            || self.applied_gamma.is_some()
        {
            match processed_image_from_path(
                &path,
                self.applied_denoise,
                self.applied_invert,
                self.applied_equalization,
                self.applied_gamma,
                self.histogram_bin_count,
            ) {
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

    fn has_image_augmentation_state(&self) -> bool {
        self.applied_denoise.is_some()
            || self.applied_invert
            || self.invert_values
            || self.applied_equalization.is_some()
            || self.equalization_strength != 0.0
            || self.applied_gamma.is_some()
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

        let denoise = self.current_denoise_settings();
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Denoise {
                denoise,
                invert: self.applied_invert,
                equalization: self.applied_equalization,
                gamma: self.applied_gamma,
                bin_count: self.histogram_bin_count,
            },
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

        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Invert {
                denoise: self.applied_denoise,
                invert: self.invert_values,
                equalization: self.applied_equalization,
                gamma: self.applied_gamma,
                bin_count: self.histogram_bin_count,
            },
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
        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Equalize {
                denoise: self.applied_denoise,
                invert: self.applied_invert,
                equalization,
                gamma: self.applied_gamma,
                bin_count: self.histogram_bin_count,
            },
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

        self.start_processing_job(
            ctx,
            image_index,
            path,
            name,
            ProcessingKind::Gamma {
                denoise: self.applied_denoise,
                invert: self.applied_invert,
                equalization: self.applied_equalization,
                gamma: self.gamma_value,
                bin_count: self.histogram_bin_count,
            },
        );
    }

    fn reset_image_augmentations(&mut self, ctx: &egui::Context) {
        self.viewer_error = None;
        self.histogram = None;
        self.histogram_error = None;
        self.reset_histogram_view = true;
        self.clear_histogram_highlight();
        self.displayed_image = None;
        self.viewer_needs_fit = true;
        self.applied_denoise = None;
        self.invert_values = false;
        self.applied_invert = false;
        self.equalization_strength = 0.0;
        self.applied_equalization = None;
        self.applied_gamma = None;
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

        thread::spawn(move || {
            let result = match kind {
                ProcessingKind::Preview {
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                } => processed_image_from_path(
                    &path,
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                ),
                ProcessingKind::Denoise {
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                } => processed_image_from_path(
                    &path,
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                ),
                ProcessingKind::Invert {
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                } => processed_image_from_path(
                    &path,
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                ),
                ProcessingKind::Equalize {
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                } => processed_image_from_path(
                    &path,
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                ),
                ProcessingKind::Gamma {
                    denoise,
                    invert,
                    equalization,
                    gamma,
                    bin_count,
                } => processed_image_from_path(
                    &path,
                    denoise,
                    invert,
                    equalization,
                    Some(gamma),
                    bin_count,
                ),
            };
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
            kind,
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
            if ui.button("Reset").clicked() {
                self.reset_histogram_view = true;
            }
            if !self.selected_histogram_bins.is_empty() && ui.button("Clear Selection").clicked() {
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

            if plot_ui.response().contains_pointer() {
                let scroll_delta = plot_ui.ctx().input(|input| input.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = (scroll_delta * 0.002).exp();
                    plot_ui.zoom_bounds_around_hovered(Vec2::splat(zoom_factor));
                }
            }

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
            ProcessingKind::Preview {
                denoise: self.applied_denoise,
                invert: self.applied_invert,
                equalization: self.applied_equalization,
                gamma: self.applied_gamma,
                bin_count: self.histogram_bin_count,
            },
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
    fn label(self) -> &'static str {
        match self {
            Self::Preview { .. } => "preview",
            Self::Denoise { .. } => "denoising",
            Self::Invert { .. } => "inversion",
            Self::Equalize { .. } => "equalization",
            Self::Gamma { .. } => "gamma adjustment",
        }
    }

    fn texture_name(self, image_name: &str) -> String {
        match self {
            Self::Preview { .. } => format!("{image_name} preview"),
            Self::Denoise { denoise, .. } => {
                let stage_name = denoise
                    .map(|settings| settings.method.label())
                    .unwrap_or("raw");
                format!("{image_name} {stage_name} denoise")
            }
            Self::Invert { invert, .. } => {
                let stage_name = if invert { "inverted" } else { "uninverted" };
                format!("{image_name} {stage_name} values")
            }
            Self::Equalize { equalization, .. } => {
                let stage_name = equalization
                    .map(|strength| format!("equalized {:.0}%", strength * 100.0))
                    .unwrap_or_else(|| "unequalized".to_owned());
                format!("{image_name} {stage_name}")
            }
            Self::Gamma { gamma, .. } => format!("{image_name} gamma {gamma:.3}"),
        }
    }

    fn apply_to(self, app: &mut VCEMApp) {
        match self {
            Self::Preview { .. } => {}
            Self::Denoise { denoise, .. } => {
                app.applied_denoise = denoise;
            }
            Self::Invert { invert, .. } => {
                app.applied_invert = invert;
                app.invert_values = invert;
            }
            Self::Equalize { equalization, .. } => {
                app.applied_equalization = equalization;
                app.equalization_strength = equalization.unwrap_or(0.0);
            }
            Self::Gamma { gamma, .. } => {
                app.applied_gamma = Some(gamma);
            }
        }
    }
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
