use eframe::{
    App, CreationContext, Frame,
    egui::{
        self, Button, CentralPanel, Color32, DragValue, Id, Modal, Panel, PointerButton, Pos2,
        Rect, ScrollArea, Sense, TextureOptions, Ui, Vec2,
    },
};
use egui_plot::{Bar, BarChart, Plot};
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::analysis::{gamma_adjusted_image_from_path, intensity_histogram_from_path};
use crate::types::{DisplayedImage, ImageFile, IntensityHistogram};

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
    gamma_value: f64,
    applied_gamma: Option<f64>,
}

struct ImageCandidate {
    path: PathBuf,
    name: String,
    selected: bool,
    kind: ImagePickerEntryKind,
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
            gamma_value: 1.0,
            applied_gamma: None,
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
                        self.applied_gamma = None;
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

                    ui.add_space(5.0);
                    ui.heading("Intensity Histogram");
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Bin count");
                        ui.add(
                            DragValue::new(&mut self.histogram_bin_count)
                                .range(2..=4096)
                                .speed(1),
                        );
                    });

                    if ui.button("Create Histogram").clicked() {
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
                    ui.heading("Gamma Adjustment");
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Gamma");
                        ui.add(
                            DragValue::new(&mut self.gamma_value)
                                .range(0.001..=2.0)
                                .speed(0.01),
                        );
                    });

                    if ui.button("Apply Gamma Adjustment").clicked() {
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
                }
            });
            ui.separator();

            let available_size = ui.available_size().max(Vec2::splat(1.0));
            let (canvas_rect, response) =
                ui.allocate_exact_size(available_size, Sense::click_and_drag());
            let painter = ui.painter_at(canvas_rect);
            painter.rect_filled(canvas_rect, 0.0, ui.visuals().extreme_bg_color);

            self.ensure_selected_image_loaded(ui.ctx());

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

            let Some(displayed) = self.displayed_image.as_ref() else {
                painter.text(
                    canvas_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Add an image to begin",
                    egui::TextStyle::Heading.resolve(ui.style()),
                    ui.visuals().weak_text_color(),
                );
                return;
            };

            let texture_id = displayed.texture.id();
            let image_size = displayed.size;

            if self.viewer_needs_fit {
                let fit_zoom = (canvas_rect.width() / image_size.x)
                    .min(canvas_rect.height() / image_size.y)
                    .clamp(0.05, 16.0);
                self.viewer_zoom = fit_zoom;
                self.viewer_pan = Vec2::ZERO;
                self.viewer_needs_fit = false;
            }

            if response.dragged() {
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

            painter.image(
                texture_id,
                image_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );

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
            self.applied_gamma = None;
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

        if let Some(gamma) = self.applied_gamma {
            match gamma_adjusted_image_from_path(&path, gamma, self.histogram_bin_count) {
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

        match gamma_adjusted_image_from_path(&path, self.gamma_value, self.histogram_bin_count) {
            Ok(processed) => {
                let texture_name = format!("{} gamma {:.3}", name, self.gamma_value);
                let texture =
                    ctx.load_texture(texture_name, processed.color_image, TextureOptions::LINEAR);
                let should_fit = self.displayed_image.is_none();
                self.displayed_image = Some(DisplayedImage {
                    image_index,
                    size: processed.size,
                    texture,
                });
                self.histogram = Some(processed.histogram);
                self.applied_gamma = Some(self.gamma_value);
                self.reset_histogram_view = true;
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
        });
    }

    fn show_histogram_plot(&mut self, ui: &mut Ui, histogram: &IntensityHistogram) {
        let bars = histogram
            .bins
            .iter()
            .enumerate()
            .map(|(index, count)| {
                Bar::new(index as f64, *count as f64)
                    .width(1.0)
                    .fill(Color32::from_rgb(120, 170, 220))
            })
            .collect();

        let mut plot = Plot::new("intensity_histogram")
            .height(180.0)
            .allow_drag(true)
            .pan_pointer_button(PointerButton::Primary)
            .allow_scroll(false)
            .allow_zoom(true)
            .allow_boxed_zoom(false)
            .allow_axis_zoom_drag(false);

        if self.reset_histogram_view {
            plot = plot.reset();
            self.reset_histogram_view = false;
        }

        plot.show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("Intensity", bars).width(1.0));

            if plot_ui.response().contains_pointer() {
                let scroll_delta = plot_ui.ctx().input(|input| input.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = (scroll_delta * 0.002).exp();
                    plot_ui.zoom_bounds_around_hovered(Vec2::splat(zoom_factor));
                }
            }
        });
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

        if let Some(gamma) = self.applied_gamma {
            match gamma_adjusted_image_from_path(&path, gamma, self.histogram_bin_count) {
                Ok(processed) => {
                    let texture = ctx.load_texture(
                        format!("gamma-adjusted image {image_index}"),
                        processed.color_image,
                        TextureOptions::LINEAR,
                    );
                    self.histogram = Some(processed.histogram);
                    self.displayed_image = Some(DisplayedImage {
                        image_index,
                        size: processed.size,
                        texture,
                    });
                    self.viewer_needs_fit = true;
                }
                Err(error) => {
                    self.viewer_error = Some(error);
                }
            }
        } else {
            match Self::load_displayed_image(ctx, image_index, path) {
                Ok(image) => {
                    self.displayed_image = Some(image);
                    self.viewer_needs_fit = true;
                }
                Err(error) => {
                    self.viewer_error = Some(error);
                }
            }
        }
    }

    fn load_displayed_image(
        ctx: &egui::Context,
        image_index: usize,
        path: PathBuf,
    ) -> std::result::Result<DisplayedImage, String> {
        let reader = image::ImageReader::open(&path)
            .map_err(|error| format!("Could not open {}: {}", path.display(), error))?
            .with_guessed_format()
            .map_err(|error| format!("Could not inspect {}: {}", path.display(), error))?;

        let image = reader
            .decode()
            .map_err(|error| format!("Could not decode {}: {}", path.display(), error))?;
        let size = [image.width() as usize, image.height() as usize];
        let rgba = image.to_rgba8();
        let pixels = rgba.as_flat_samples();
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Untitled image")
            .to_owned();
        let texture = ctx.load_texture(&name, color_image, TextureOptions::LINEAR);

        Ok(DisplayedImage {
            image_index,
            size: Vec2::new(size[0] as f32, size[1] as f32),
            texture,
        })
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

impl App for VCEMApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut Frame) {
        self.show_upper_panel(ui);
        self.show_left_panel(ui);
        self.show_right_panel(ui);
        self.show_bottom_panel(ui);
        self.show_central_panel(ui);
        self.show_image_picker_modal(ui.ctx());
    }
}
