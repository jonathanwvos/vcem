use super::*;
use crate::automata::CellularAutomataResult;
use crate::masks::{
    self, LayerSource, MAX_LAYERS, MAX_RECIPE_BYTES, MAX_REGIONS, MaskDocument, MaskFrame,
    MaskOperation, MaskSession, ReviewClass, ReviewRegion,
};
use std::collections::BTreeMap;

const MAX_CAPTURED_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Preview {
    #[default]
    CurrentCa,
    ImageOnly,
    Layer(u64),
    Compare,
}
struct RebuildJob {
    receiver: Receiver<Result<MaskSession, String>>,
    cancelled: Arc<AtomicBool>,
}

pub(super) struct MaskUi {
    frame: Option<MaskFrame>,
    sessions: BTreeMap<PathBuf, MaskSession>,
    name: String,
    left: Option<u64>,
    right: Option<u64>,
    operation: MaskOperation,
    preview: Preview,
    masks: BTreeMap<u64, Vec<u8>>,
    counts: BTreeMap<u64, usize>,
    texture: Option<egui::TextureHandle>,
    dirty: bool,
    recipe_path: String,
    notice: Option<String>,
    job: Option<RebuildJob>,
    draw_regions: bool,
    region_class: ReviewClass,
    region_start: Option<[f32; 2]>,
    region_end: Option<[f32; 2]>,
    region_coverage: Vec<[f32; 3]>,
}
impl Default for MaskUi {
    fn default() -> Self {
        Self {
            frame: None,
            sessions: BTreeMap::new(),
            name: "Fibril candidates".to_owned(),
            left: None,
            right: None,
            operation: MaskOperation::Subtract,
            preview: Preview::CurrentCa,
            masks: BTreeMap::new(),
            counts: BTreeMap::new(),
            texture: None,
            dirty: true,
            recipe_path: String::new(),
            notice: None,
            job: None,
            draw_regions: false,
            region_class: ReviewClass::Fibril,
            region_start: None,
            region_end: None,
            region_coverage: Vec::new(),
        }
    }
}
impl MaskUi {
    pub(super) fn suggest_name(&mut self, name: String) {
        self.name = name;
    }
    pub(super) fn is_busy(&self) -> bool {
        self.job.is_some()
    }
    pub(super) fn activate(&mut self, frame: MaskFrame) {
        let changed_image = self
            .frame
            .as_ref()
            .is_none_or(|f| f.source.path != frame.source.path);
        if changed_image {
            // Stable path-based suffix keeps identically named images in separate recipes.
            let key = frame
                .source
                .path
                .as_os_str()
                .as_encoded_bytes()
                .iter()
                .fold(0xcbf29ce484222325_u64, |hash, byte| {
                    (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
                });
            self.recipe_path = std::env::current_dir()
                .unwrap_or_default()
                .join(format!(
                    "{}-{key:016x}.masks.json",
                    frame
                        .source
                        .path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ))
                .display()
                .to_string();
            self.preview = Preview::CurrentCa;
            self.left = None;
            self.right = None;
            self.draw_regions = false;
            self.notice = None;
        }
        if self.frame.as_ref() != Some(&frame) {
            self.texture = None;
            self.masks.clear();
            self.region_coverage.clear();
            self.region_start = None;
            self.region_end = None;
        }
        self.sessions
            .entry(frame.source.path.clone())
            .or_insert_with(|| MaskSession::new(frame.clone()));
        if let Some(session) = self.sessions.get_mut(&frame.source.path) {
            if session.document.layers.is_empty() && session.document.regions.is_empty() {
                session.document.frame = frame.clone();
            }
        }
        self.frame = Some(frame);
        self.dirty = true;
    }
    fn session(&self) -> Option<&MaskSession> {
        self.sessions.get(&self.frame.as_ref()?.source.path)
    }
    fn compatible(&self) -> bool {
        self.session()
            .is_some_and(|s| Some(&s.document.frame) == self.frame.as_ref())
    }
    pub(super) fn suppress_other_overlays(&self) -> bool {
        self.preview != Preview::CurrentCa
    }
    pub(super) fn show_current_ca(&mut self) {
        self.preview = Preview::CurrentCa;
        self.dirty = true;
    }
    pub(super) fn texture_id(&self) -> Option<egui::TextureId> {
        self.compatible()
            .then(|| self.texture.as_ref().map(|t| t.id()))
            .flatten()
    }
    fn capture(
        &mut self,
        configuration: ExperimentConfiguration,
        result: CellularAutomataResult,
    ) -> Result<(), String> {
        let frame = self.frame.as_ref().ok_or("Load an image first.")?;
        if masks::SourceStamp::read(&frame.source.path)? != frame.source {
            return Err("The source file changed. Reload the image first.".to_owned());
        }
        let retained_bytes: usize = self
            .sessions
            .values()
            .map(MaskSession::activation_bytes)
            .sum();
        if retained_bytes + result.state.len() * std::mem::size_of::<f32>() > MAX_CAPTURED_BYTES {
            return Err("Captured-mask memory limit reached (256 MiB). Save recipes and remove unused layers before capturing more.".to_owned());
        }
        let session = self.sessions.get_mut(&frame.source.path).unwrap();
        let id = session.capture(frame, self.name.clone(), configuration, result)?;
        self.left.get_or_insert(id);
        self.right = Some(id);
        self.preview = Preview::Layer(id);
        self.name = format!("Layer {}", session.document.layers.len() + 1);
        self.notice =
            Some("Captured final activations. Threshold changes do not rerun the CA.".to_owned());
        self.dirty = true;
        Ok(())
    }
    fn capture_lab(
        &mut self,
        configuration: ExperimentConfiguration,
        model: &LabModelSettings,
        lab: &AutomataLabSession,
    ) -> Result<(), String> {
        let frame = self.frame.as_ref().ok_or("Load an image first.")?;
        if masks::SourceStamp::read(&frame.source.path)? != frame.source {
            return Err("The source image changed. Reload it first.".into());
        }
        if model.program_document(lab.calibration())? != *lab.program_document() {
            return Err("Lab settings do not match the applied run.".into());
        }
        let retained: usize = self
            .sessions
            .values()
            .map(MaskSession::activation_bytes)
            .sum();
        if retained + frame.pixel_count()? * 4 > MAX_CAPTURED_BYTES {
            return Err("Captured-mask memory limit reached (256 MiB).".into());
        }
        let experiment = LabExperimentConfiguration {
            schema_version: 1,
            preprocessing: configuration,
            automaton: model.clone(),
            generations: lab.generation(),
        };
        let result = crate::lab_configuration::snapshot_result(lab)?;
        let session = self.sessions.get_mut(&frame.source.path).unwrap();
        let id = session.capture_lab(frame, self.name.clone(), experiment, result)?;
        self.left.get_or_insert(id);
        self.right = Some(id);
        self.preview = Preview::Layer(id);
        self.name = format!("Layer {}", session.document.layers.len() + 1);
        self.dirty = true;
        self.notice = Some(format!(
            "Captured Lab generation {} with its image preparation and automaton.",
            lab.generation()
        ));
        Ok(())
    }
    fn combine(&mut self, invert: bool) -> Result<(), String> {
        if !self.compatible() {
            return Err("Rebuild layers for this view first.".to_owned());
        }
        let left = self.left.ok_or("Select layer A.")?;
        let source = if invert {
            LayerSource::Invert { input: left }
        } else {
            LayerSource::Combine {
                left,
                operation: self.operation,
                right: self.right.ok_or("Select layer B.")?,
            }
        };
        let frame = self.frame.as_ref().unwrap();
        let session = self.sessions.get_mut(&frame.source.path).unwrap();
        let id = session.add_layer(self.name.clone(), source)?;
        self.preview = Preview::Layer(id);
        self.name = format!("Layer {}", session.document.layers.len() + 1);
        self.dirty = true;
        Ok(())
    }
    fn begin_rebuild(
        &mut self,
        mut document: MaskDocument,
        ctx: &egui::Context,
    ) -> Result<(), String> {
        if self.job.is_some() {
            return Err("A mask rebuild is already running.".to_owned());
        }
        document.validate()?;
        document.retarget(self.frame.as_ref().ok_or("Load an image first.")?)?;
        let retained_bytes: usize = self
            .sessions
            .iter()
            .filter(|(path, _)| **path != document.frame.source.path)
            .map(|(_, session)| session.activation_bytes())
            .sum();
        let new_bytes = document
            .layers
            .iter()
            .filter(|layer| {
                matches!(
                    layer.source,
                    LayerSource::Ca { .. } | LayerSource::Lab { .. }
                )
            })
            .count()
            * document.frame.pixel_count()?
            * std::mem::size_of::<f32>();
        if retained_bytes + new_bytes > MAX_CAPTURED_BYTES {
            return Err("Rebuilding would exceed the captured-mask memory limit (256 MiB). Remove unused layers first.".to_owned());
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_cancelled = cancelled.clone();
        let (sender, receiver) = mpsc::channel();
        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = masks::rebuild(document, &worker_cancelled);
            let _ = sender.send(result);
            ctx.request_repaint();
        });
        self.job = Some(RebuildJob {
            receiver,
            cancelled,
        });
        self.notice =
            Some("Rebuilding captured CA layers; the current masks remain available.".to_owned());
        Ok(())
    }
    pub(super) fn poll(&mut self, ctx: &egui::Context) {
        let Some(job) = self.job.as_ref() else {
            return;
        };
        let result = match job.receiver.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(100));
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Mask worker stopped unexpectedly; previous masks were retained.".to_owned())
            }
        };
        let cancelled = job.cancelled.load(Ordering::Relaxed);
        self.job = None;
        if cancelled {
            self.notice = Some("Rebuild cancelled; previous masks retained.".to_owned());
            return;
        }
        match result {
            Ok(session) => {
                let path = session.document.frame.source.path.clone();
                self.sessions.insert(path, session);
                self.dirty = true;
                self.notice = Some(
                    "Mask layers rebuilt. Review regions were aligned to the current rotation."
                        .to_owned(),
                );
            }
            Err(error) => self.notice = Some(error),
        }
    }
    fn save(&self) -> Result<(), String> {
        let document = &self.session().ok_or("Load an image first.")?.document;
        document.validate()?;
        let path = PathBuf::from(self.recipe_path.trim());
        if path.as_os_str().is_empty() {
            return Err("Enter a mask recipe path.".to_owned());
        }
        // A recipe must never overwrite its source image.
        if path.canonicalize().ok().as_ref() == Some(&document.frame.source.path) {
            return Err("Choose a recipe path different from the source image.".to_owned());
        }
        if path.exists() {
            let existing = read_recipe(&path)?;
            if existing.frame.source.path != document.frame.source.path {
                return Err(
                    "This recipe path belongs to another image. Choose a different save path."
                        .to_owned(),
                );
            }
        }
        let json = serde_json::to_string_pretty(document).map_err(|e| e.to_string())?;
        if json.len() as u64 > MAX_RECIPE_BYTES {
            return Err(
                "Mask recipe is too large. Remove unused layers or reduce rule tables.".into(),
            );
        }
        fs::write(path, json).map_err(|e| format!("Could not save mask recipe: {e}"))
    }
    fn load(&mut self, ctx: &egui::Context) -> Result<(), String> {
        let document = read_recipe(Path::new(self.recipe_path.trim()))?;
        self.begin_rebuild(document, ctx)
    }
    pub(super) fn refresh(&mut self, ctx: &egui::Context) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        self.texture = None;
        self.counts.clear();
        self.region_coverage.clear();
        if !self.compatible() {
            self.masks.clear();
            return;
        }
        let Some(session) = self.session() else {
            return;
        };
        match session.evaluate() {
            Ok(masks) => {
                self.counts = masks
                    .iter()
                    .map(|(id, mask)| (*id, mask.iter().filter(|v| **v != 0).count()))
                    .collect();
                self.masks = masks;
            }
            Err(error) => {
                self.masks.clear();
                self.notice = Some(error);
                return;
            }
        }
        if self.left.is_none_or(|id| !self.masks.contains_key(&id)) {
            self.left = self.masks.keys().next().copied();
        }
        if self.right.is_none_or(|id| !self.masks.contains_key(&id)) {
            self.right = self.masks.keys().next_back().copied();
        }
        if let Preview::Layer(id) = self.preview {
            if !self.masks.contains_key(&id) {
                self.preview = Preview::ImageOnly;
            }
        }
        let frame = self.frame.as_ref().unwrap();
        let rgba = match self.preview {
            Preview::Layer(id) => self.masks[&id]
                .iter()
                .flat_map(|v| {
                    if *v != 0 {
                        [30, 210, 190, 180]
                    } else {
                        [0, 0, 0, 0]
                    }
                })
                .collect(),
            Preview::Compare => {
                if let (Some(a), Some(b)) = (self.left, self.right) {
                    masks::overlap_rgba(&self.masks[&a], &self.masks[&b])
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        };
        if !rgba.is_empty() {
            self.texture = Some(ctx.load_texture(
                "mask workspace",
                egui::ColorImage::from_rgba_unmultiplied(frame.size, &rgba),
                TextureOptions::NEAREST,
            ));
        }
        if let (Some(a), Some(b)) = (self.left, self.right) {
            let overlap: Vec<u8> = self.masks[&a]
                .iter()
                .zip(&self.masks[&b])
                .map(|(a, b)| a & b)
                .collect();
            self.region_coverage = self
                .session()
                .unwrap()
                .document
                .regions
                .iter()
                .map(|region| {
                    [
                        region.coverage(&self.masks[&a], frame.size),
                        region.coverage(&self.masks[&b], frame.size),
                        region.coverage(&overlap, frame.size),
                    ]
                })
                .collect();
        }
        ctx.request_repaint();
    }

    fn show(
        &mut self,
        ui: &mut Ui,
        busy: bool,
        configuration: ExperimentConfiguration,
        result: Option<&CellularAutomataResult>,
        lab: Option<(&LabModelSettings, &AutomataLabSession)>,
        lab_mode: bool,
    ) {
        if self.frame.is_none() {
            ui.label("Load an image to create masks.");
            return;
        }
        let compatible = self.compatible();
        if !compatible {
            ui.colored_label(ui.visuals().warn_fg_color, "Saved masks use a different image version, rotation, or downsampling. Rebuild for this view before combining them.");
        }
        ui.small("Masks are provisional selections. Inspect overlap before excluding suspected contamination.");
        if let Some(message) = &self.notice {
            ui.label(message);
        }
        if self.is_busy() {
            ui.spinner();
            if ui
                .button("Cancel rebuild")
                .on_hover_text("Stops after the current CA layer finishes.")
                .clicked()
            {
                self.job
                    .as_ref()
                    .unwrap()
                    .cancelled
                    .store(true, Ordering::Relaxed);
            }
        }
        let disabled = busy || self.is_busy();
        ui.add_enabled_ui(!disabled, |ui| {
            if !compatible && ui.button("Start new workspace (clear old masks)").on_hover_text("Save the old recipe first if you want to keep its layers and regions.").clicked() {
                let frame = self.frame.as_ref().unwrap().clone();
                self.sessions.insert(frame.source.path.clone(), MaskSession::new(frame));
                self.preview = Preview::ImageOnly;
                self.left = None;
                self.right = None;
                self.dirty = true;
                self.notice = Some("Started a new mask workspace for the current image.".to_owned());
            }
            ui.text_edit_singleline(&mut self.name).on_hover_text("Name for the next captured or combined layer.");
            if lab_mode {
                if ui.add_enabled(compatible && lab.is_some(), Button::new("Capture Lab state as layer")).clicked() {
                    if let Some((model, session)) = lab {
                        if let Err(e) = self.capture_lab(configuration.clone(), model, session) { self.notice = Some(e); }
                    }
                }
                ui.small("Pause playback to capture. Discrete mask activation = state / (state count − 1).");
            }
            if !lab_mode && ui.add_enabled(compatible && result.is_some(), Button::new("Capture current CA as layer")).clicked() {
                if let Some(result) = result {
                    if let Err(e) = self.capture(configuration.clone(), result.clone()) { self.notice = Some(e); }
                }
            }
            ui.small(format!("Up to {MAX_LAYERS} layers per image. A threshold selects final activation, not the growth response."));
            let key = self.frame.as_ref().unwrap().source.path.clone();
            let mut remove = None;
            if let Some(session) = self.sessions.get_mut(&key) {
                for layer in &mut session.document.layers {
                    ui.push_id(layer.id, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("#{}", layer.id));
                            if ui.text_edit_singleline(&mut layer.name).changed() { self.dirty = true; }
                            if ui.small_button("×").on_hover_text("Remove this layer; referenced layers must be removed last.").clicked() { remove = Some(layer.id); }
                        });
                        match &mut layer.source {
                            LayerSource::Ca { threshold, .. } | LayerSource::Lab { threshold, .. } => {
                                if ui.add(egui::Slider::new(threshold, 0.0..=1.0).text("Mask threshold").max_decimals(3)).changed() { self.dirty = true; }
                            }
                            LayerSource::Combine { left, operation, right } => { ui.small(format!("#{left} {} #{right}", operation.label())); }
                            LayerSource::Invert { input } => { ui.small(format!("NOT #{input}")); }
                        }
                        ui.horizontal(|ui| {
                            if ui.selectable_label(self.preview == Preview::Layer(layer.id), "Show mask").clicked() { self.preview = Preview::Layer(layer.id); self.dirty = true; }
                            if let Some(mask) = self.masks.get(&layer.id) {
                                let count = self.counts.get(&layer.id).copied().unwrap_or(0);
                                ui.label(format!("{:.1}% selected", count as f32 * 100.0 / mask.len() as f32));
                            }
                        });
                        ui.separator();
                    });
                }
            }
            if let Some(id) = remove {
                match self.sessions.get_mut(&key).unwrap().remove(id) { Ok(()) => self.dirty = true, Err(e) => self.notice = Some(e) }
            }
            let labels = self.session().unwrap().document.layers.iter().map(|l| (l.id, format!("#{} {}", l.id, l.name))).collect::<Vec<_>>();
            for (title, selection) in [("A", &mut self.left), ("B", &mut self.right)] {
                egui::ComboBox::from_id_salt(("mask operand",title)).selected_text(labels.iter().find(|(id,_)| Some(*id) == *selection).map(|(_,name)| format!("{title}: {name}")).unwrap_or(format!("Select {title}"))).show_ui(ui, |ui| {
                    for (id,name) in &labels { if ui.selectable_value(selection, Some(*id), name).changed() { self.dirty = true; } }
                });
            }
            egui::ComboBox::from_id_salt("mask operation").selected_text(self.operation.label()).show_ui(ui, |ui| {
                for op in [MaskOperation::And, MaskOperation::Or, MaskOperation::Xor, MaskOperation::Subtract] { ui.selectable_value(&mut self.operation, op, op.label()); }
            });
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(compatible && self.left.is_some() && self.right.is_some(), Button::new("Create combination")).clicked() {
                    if let Err(e) = self.combine(false) { self.notice = Some(e); }
                }
                if ui.add_enabled(compatible && self.left.is_some(), Button::new("Create NOT A")).clicked() {
                    if let Err(e) = self.combine(true) { self.notice = Some(e); }
                }
            });
            ui.horizontal_wrapped(|ui| {
                for (mode,label) in [(Preview::CurrentCa,"Current CA"),(Preview::ImageOnly,"Image only"),(Preview::Compare,"Compare A / B")] {
                    if ui.selectable_value(&mut self.preview, mode, label).changed() { self.dirty = true; }
                }
            });
            if self.preview == Preview::Compare {
                ui.colored_label(Color32::from_rgb(30,210,190), "A only");
                ui.colored_label(Color32::from_rgb(220,90,190), "B only");
                ui.colored_label(Color32::from_rgb(100,140,255), "A AND B — inspect before subtraction");
            }
            ui.collapsing("Review regions", |ui| {
                ui.add_enabled(compatible, egui::Checkbox::new(&mut self.draw_regions,"Draw review regions"));
                for class in [ReviewClass::Fibril,ReviewClass::Contamination,ReviewClass::Uncertain] { ui.selectable_value(&mut self.region_class, class, class.label()); }
                ui.small("Drag a rectangle on the image. Percentages show selected pixels within each region, not classification accuracy.");
                let mut remove = None;
                for (index,region) in self.session().unwrap().document.regions.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}: {}",index+1,region.class.label()));
                        if ui.small_button("Remove").clicked() { remove = Some(index); }
                    });
                    if let Some([a,b,overlap]) = self.region_coverage.get(index) { ui.small(format!("A {:.1}% · B {:.1}% · overlap {:.1}%", a*100.0,b*100.0,overlap*100.0)); }
                }
                if let Some(index) = remove { self.sessions.get_mut(&key).unwrap().document.regions.remove(index); self.dirty = true; }
            });
            ui.collapsing("Save / rebuild mask recipe", |ui| {
                ui.text_edit_singleline(&mut self.recipe_path);
                ui.small("Saves configurations, thresholds, combinations and regions. Loading reruns the CAs on this image using the current rotation and downsampling.");
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Save recipe").clicked() { self.notice = Some(match self.save() { Ok(()) => "Saved mask recipe.".to_owned(), Err(e) => e }); }
                    if ui.button("Load & rebuild").clicked() { if let Err(e) = self.load(ui.ctx()) { self.notice = Some(e); } }
                    if ui.button("Rebuild for current view").clicked() {
                        let document = self.session().unwrap().document.clone();
                        if let Err(e) = self.begin_rebuild(document, ui.ctx()) { self.notice = Some(e); }
                    }
                });
            });
        });
    }
    pub(super) fn marking(&self) -> bool {
        self.draw_regions && self.compatible() && !self.is_busy()
    }
    pub(super) fn region_pointer(&mut self, response: &egui::Response, image_rect: Rect) {
        let normalized = |p: Pos2| {
            [
                ((p.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0),
                ((p.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0),
            ]
        };
        if response.drag_started_by(PointerButton::Primary) {
            self.region_start = response
                .ctx
                .input(|i| i.pointer.press_origin())
                .filter(|p| image_rect.contains(*p))
                .map(normalized);
        }
        if self.region_start.is_some() {
            self.region_end = response.interact_pointer_pos().map(normalized);
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            if let (Some(a), Some(b)) = (self.region_start.take(), self.region_end.take()) {
                let region = ReviewRegion {
                    class: self.region_class,
                    bounds: [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[0].max(b[0]),
                        a[1].max(b[1]),
                    ],
                };
                let frame = self.frame.as_ref().unwrap();
                let session = self.sessions.get_mut(&frame.source.path).unwrap();
                if region.bounds[0] < region.bounds[2] && region.bounds[1] < region.bounds[3] {
                    if session.document.regions.len() < MAX_REGIONS {
                        session.document.regions.push(region);
                        self.dirty = true;
                    } else {
                        self.notice = Some(format!("Review region limit reached ({MAX_REGIONS})."));
                    }
                }
            }
        }
    }
    pub(super) fn paint_regions(&self, painter: &egui::Painter, image_rect: Rect) {
        if !self.compatible() {
            return;
        }
        let rect = |b: [f32; 4]| {
            Rect::from_min_max(
                image_rect.min + image_rect.size() * Vec2::new(b[0], b[1]),
                image_rect.min + image_rect.size() * Vec2::new(b[2], b[3]),
            )
        };
        for (index, region) in self.session().unwrap().document.regions.iter().enumerate() {
            let color = match region.class {
                ReviewClass::Fibril => Color32::from_rgb(30, 210, 190),
                ReviewClass::Contamination => Color32::from_rgb(220, 90, 190),
                ReviewClass::Uncertain => Color32::from_rgb(255, 185, 50),
            };
            let rect = rect(region.bounds);
            painter.rect_stroke(rect, 0.0, Stroke::new(1.5, color), egui::StrokeKind::Inside);
            painter.text(
                rect.min,
                egui::Align2::LEFT_TOP,
                format!("{}", index + 1),
                egui::FontId::proportional(13.0),
                color,
            );
        }
        if let (Some(a), Some(b)) = (self.region_start, self.region_end) {
            painter.rect_stroke(
                rect([
                    a[0].min(b[0]),
                    a[1].min(b[1]),
                    a[0].max(b[0]),
                    a[1].max(b[1]),
                ]),
                0.0,
                Stroke::new(1.5, Color32::YELLOW),
                egui::StrokeKind::Inside,
            );
        }
    }
}
fn read_recipe(path: &Path) -> Result<MaskDocument, String> {
    use std::io::Read;
    let file = fs::File::open(path).map_err(|e| format!("Could not open recipe: {e}"))?;
    let mut bytes = Vec::new();
    file.take(MAX_RECIPE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_RECIPE_BYTES {
        return Err("Mask recipe is too large.".to_owned());
    }
    serde_json::from_slice(&bytes).map_err(|e| format!("Invalid mask recipe: {e}"))
}

impl VCEMApp {
    pub(super) fn show_mask_controls(&mut self, ui: &mut Ui, busy: bool) {
        ui.collapsing("Mask layers & review", |ui| {
            if self
                .displayed_image
                .as_ref()
                .is_none_or(|d| Some(d.image_index) != self.selected_image)
            {
                ui.label("Wait for the selected image to load.");
                return;
            }
            let mut configuration = self.applied_experiment_configuration();
            let lab_mode = self.view_mode == ViewMode::AutomataLab;
            if lab_mode {
                configuration.cellular_automata = None;
                if let Some(displayed) = &self.displayed_image {
                    configuration.histogram_bin_count = displayed.bin_count;
                }
            }
            let lab = self
                .lab_applied_model
                .as_ref()
                .zip(self.automata_lab_session.as_ref());
            self.masks.show(
                ui,
                busy,
                configuration,
                self.displayed_image
                    .as_ref()
                    .and_then(|d| d.cellular_automata_result.as_ref()),
                lab,
                lab_mode,
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(name: &str) -> (PathBuf, ExperimentConfiguration) {
        let dir = std::env::temp_dir().join(format!("vcem-mask-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("source.png");
        image::GrayImage::from_fn(8, 4, |x, y| image::Luma([(x * 20 + y * 12) as u8]))
            .save(&path)
            .unwrap();
        let config = ExperimentConfiguration {
            cellular_automata: Some(CellularAutomataSettings {
                iterations: 1,
                ..Default::default()
            }),
            ..Default::default()
        };
        (path, config)
    }
    fn finish(ui: &mut MaskUi, ctx: &egui::Context) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while ui.is_busy() {
            assert!(Instant::now() < deadline, "mask job timed out");
            ui.poll(ctx);
            thread::sleep(Duration::from_millis(1));
        }
        ui.refresh(ctx);
    }
    #[test]
    fn inspect_discrete_and_lenia_layers_combine_save_and_rebuild_together() {
        let ctx = egui::Context::default();
        let (path, configuration) = fixture("mixed-lab");
        let (processed, frame) = masks::process_source(&path, &configuration).unwrap();
        let mut ui = MaskUi::default();
        ui.activate(frame.clone());
        ui.capture(
            configuration.clone(),
            processed.cellular_automata_result.unwrap(),
        )
        .unwrap();
        let mut preprocessing = configuration;
        preprocessing.cellular_automata = None;
        for model in [
            LabModelSettings::Discrete(DiscreteLabSettings {
                direct_table: vec![1, 0],
                ..Default::default()
            }),
            LabModelSettings::Continuous(LeniaLabSettings::default()),
        ] {
            let experiment = LabExperimentConfiguration {
                schema_version: 1,
                preprocessing: preprocessing.clone(),
                automaton: model.clone(),
                generations: 2,
            };
            let mut lab = experiment
                .initialize(
                    &processed.prepared_values,
                    processed.calibration,
                    frame.size,
                )
                .unwrap();
            lab.step(2).unwrap();
            ui.capture_lab(preprocessing.clone(), &model, &lab).unwrap();
        }
        let session = ui.sessions.get_mut(&frame.source.path).unwrap();
        let combination = session
            .add_layer(
                "Mixed intersection".into(),
                LayerSource::Combine {
                    left: 1,
                    operation: MaskOperation::And,
                    right: 2,
                },
            )
            .unwrap();
        session
            .add_layer(
                "Subtract Lenia".into(),
                LayerSource::Combine {
                    left: combination,
                    operation: MaskOperation::Subtract,
                    right: 3,
                },
            )
            .unwrap();
        let before = session.evaluate().unwrap();
        assert_eq!(
            before[&combination],
            before[&1]
                .iter()
                .zip(&before[&2])
                .map(|(a, b)| a & b)
                .collect::<Vec<_>>()
        );
        ui.recipe_path = path
            .parent()
            .unwrap()
            .join("mixed.json")
            .display()
            .to_string();
        ui.save().unwrap();
        ui.load(&ctx).unwrap();
        finish(&mut ui, &ctx);
        assert_eq!(before, ui.session().unwrap().evaluate().unwrap());
        assert_eq!(ui.session().unwrap().document.schema_version, 2);
        assert!(ui.session().unwrap().activation_bytes() > 0);
        // Source-image version 1 recipes remain supported and upgrade on Lab capture.
        let mut old = masks::MaskDocument::new(frame);
        old.schema_version = 1;
        old.validate().unwrap();
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn capture_save_rebuild_and_image_switch_preserve_independent_layers() {
        let ctx = egui::Context::default();
        let (path, config) = fixture("roundtrip");
        let (processed, frame) = masks::process_source(&path, &config).unwrap();
        let mut ui = MaskUi::default();
        ui.activate(frame.clone());
        ui.name = "Fibrils".to_owned();
        ui.capture(config.clone(), processed.cellular_automata_result.unwrap())
            .unwrap();
        let mut second = config.clone();
        second.gamma = Some(2.0);
        let (processed, _) = masks::process_source(&path, &second).unwrap();
        ui.name = "Contamination".to_owned();
        ui.capture(second, processed.cellular_automata_result.unwrap())
            .unwrap();
        ui.name = "Fibrils minus contamination".to_owned();
        ui.combine(false).unwrap();
        ui.sessions
            .get_mut(&frame.source.path)
            .unwrap()
            .document
            .regions
            .push(ReviewRegion {
                class: ReviewClass::Fibril,
                bounds: [0.1, 0.2, 0.5, 0.8],
            });
        ui.refresh(&ctx);
        let before = ui.masks.clone();
        ui.recipe_path = path.with_extension("masks.json").display().to_string();
        ui.save().unwrap();
        let recipe_path = ui.recipe_path.clone();
        let old = ui.sessions.remove(&frame.source.path).unwrap();
        ui.sessions
            .insert(frame.source.path.clone(), MaskSession::new(frame.clone()));
        ui.load(&ctx).unwrap();
        // Loading leaves the current session in place until the entire job succeeds.
        assert!(ui.session().unwrap().document.layers.is_empty());
        finish(&mut ui, &ctx);
        assert_eq!(ui.masks, before);
        assert_eq!(ui.session().unwrap().document.layers.len(), 3);
        assert_eq!(ui.session().unwrap().document.regions.len(), 1);
        assert_eq!(ui.session().unwrap().results.len(), old.results.len());

        let (other_path, _) = fixture("other");
        let (_, other) = masks::process_source(&other_path, &config).unwrap();
        ui.activate(other);
        ui.refresh(&ctx);
        assert!(ui.masks.is_empty());
        ui.recipe_path = recipe_path;
        assert!(
            ui.load(&ctx).is_err(),
            "must reject recipe for another image"
        );
        ui.activate(frame.clone());
        ui.refresh(&ctx);
        assert_eq!(ui.masks, before);

        let mut rotated = config;
        rotated.rotation = ImageRotation::Clockwise90;
        rotated.downsampling = DownsamplingMethod::Nearest;
        let (_, rotated_frame) = masks::process_source(&path, &rotated).unwrap();
        ui.activate(rotated_frame.clone());
        ui.refresh(&ctx);
        assert!(!ui.compatible());
        assert!(ui.texture_id().is_none());
        assert!(!ui.marking());
        ui.begin_rebuild(ui.session().unwrap().document.clone(), &ctx)
            .unwrap();
        finish(&mut ui, &ctx);
        assert!(ui.compatible());
        assert_eq!(ui.session().unwrap().document.frame, rotated_frame);
        let bounds = ui.session().unwrap().document.regions[0].bounds;
        for (actual, expected) in bounds.iter().zip([0.2, 0.1, 0.8, 0.5]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
        fs::remove_dir_all(other_path.parent().unwrap()).unwrap();
    }
    #[test]
    fn failed_or_cancelled_rebuild_retains_previous_masks_and_source_is_protected() {
        let ctx = egui::Context::default();
        let (path, config) = fixture("failure");
        let (processed, frame) = masks::process_source(&path, &config).unwrap();
        let mut ui = MaskUi::default();
        ui.activate(frame.clone());
        ui.capture(config, processed.cellular_automata_result.unwrap())
            .unwrap();
        ui.refresh(&ctx);
        let before = ui.masks.clone();
        ui.recipe_path = path.display().to_string();
        assert!(ui.save().is_err());
        ui.begin_rebuild(ui.session().unwrap().document.clone(), &ctx)
            .unwrap();
        ui.job
            .as_ref()
            .unwrap()
            .cancelled
            .store(true, Ordering::Relaxed);
        finish(&mut ui, &ctx);
        assert_eq!(ui.masks, before);
        assert!(ui.notice.as_ref().unwrap().contains("cancelled"));
        ui.recipe_path = path.with_extension("masks.json").display().to_string();
        fs::write(&ui.recipe_path, "not JSON").unwrap();
        assert!(ui.load(&ctx).is_err());
        fs::write(&ui.recipe_path, vec![b' '; MAX_RECIPE_BYTES as usize + 1]).unwrap();
        assert!(ui.load(&ctx).is_err());
        // Replacing the source must not silently reuse coordinates or overwrite old masks.
        fs::write(&path, b"changed source").unwrap();
        ui.begin_rebuild(ui.session().unwrap().document.clone(), &ctx)
            .unwrap();
        finish(&mut ui, &ctx);
        assert_eq!(ui.masks, before);
        assert!(ui.notice.as_ref().unwrap().contains("changed"));
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
    #[test]
    fn applying_new_ca_shows_its_result_without_losing_captured_layers() {
        let ctx = egui::Context::default();
        let (path, config) = fixture("new-ca");
        let mut app = VCEMApp::initial_state();
        app.add_image_paths(vec![path.clone()]);
        let finish = |app: &mut VCEMApp| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while app.processing_job.is_some() {
                assert!(Instant::now() < deadline);
                app.poll_processing_job(&ctx);
                thread::sleep(Duration::from_millis(1));
            }
            assert!(app.viewer_error.is_none(), "{:?}", app.viewer_error);
            app.masks.refresh(&ctx);
        };
        app.cellular_automata_settings = config.cellular_automata.unwrap();
        app.apply_cellular_automata(&ctx, true);
        finish(&mut app);
        let captured = app
            .displayed_image
            .as_ref()
            .unwrap()
            .cellular_automata_result
            .clone()
            .unwrap();
        app.masks
            .capture(app.applied_experiment_configuration(), captured)
            .unwrap();
        app.masks.refresh(&ctx);
        assert!(app.masks.suppress_other_overlays());
        app.cellular_automata_settings.response_center = 0.4;
        app.apply_cellular_automata(&ctx, true);
        finish(&mut app);
        assert!(!app.masks.suppress_other_overlays());
        assert!(app.masks.texture_id().is_none());
        assert_eq!(app.masks.session().unwrap().document.layers.len(), 1);
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn recipes_with_identical_image_names_do_not_overwrite_each_other() {
        let (first_path, config) = fixture("same-name-first");
        let (second_path, _) = fixture("same-name-second");
        let (_, first) = masks::process_source(&first_path, &config).unwrap();
        let (_, second) = masks::process_source(&second_path, &config).unwrap();
        let mut ui = MaskUi::default();
        ui.activate(first);
        let first_default = ui.recipe_path.clone();
        ui.recipe_path = first_path
            .with_extension("masks.json")
            .display()
            .to_string();
        ui.save().unwrap();
        let saved_path = ui.recipe_path.clone();
        let original = fs::read(&saved_path).unwrap();
        ui.activate(second);
        assert_ne!(ui.recipe_path, first_default);
        ui.recipe_path = saved_path.clone();
        assert!(ui.save().unwrap_err().contains("another image"));
        assert_eq!(fs::read(saved_path).unwrap(), original);
        fs::remove_dir_all(first_path.parent().unwrap()).unwrap();
        fs::remove_dir_all(second_path.parent().unwrap()).unwrap();
    }

    #[test]
    fn drawing_review_region_uses_image_coordinates() {
        let ctx = egui::Context::default();
        let (path, config) = fixture("region-drag");
        let (_, frame) = masks::process_source(&path, &config).unwrap();
        let mut masks = MaskUi::default();
        masks.activate(frame);
        masks.draw_regions = true;
        let image_rect = Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(200.0, 100.0));
        let start = Pos2::new(50.0, 50.0);
        let end = Pos2::new(150.0, 100.0);
        let pointer = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        for events in [
            vec![],
            vec![egui::Event::PointerMoved(start), pointer(start, true)],
            vec![egui::Event::PointerMoved(end)],
            vec![pointer(end, false)],
        ] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    events,
                    ..Default::default()
                },
                |ui| {
                    let (_, response) =
                        ui.allocate_exact_size(Vec2::new(400.0, 200.0), Sense::click_and_drag());
                    masks.region_pointer(&response, image_rect);
                },
            );
        }
        let regions = &masks.session().unwrap().document.regions;
        assert_eq!(regions.len(), 1);
        for (actual, expected) in regions[0].bounds.iter().zip([0.2, 0.3, 0.7, 0.8]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(masks.region_start.is_none());
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn mask_panel_renders_headlessly_with_layers_and_review_regions() {
        let ctx = egui::Context::default();
        let (path, config) = fixture("panel");
        let (processed, frame) = masks::process_source(&path, &config).unwrap();
        let mut masks = MaskUi::default();
        masks.activate(frame.clone());
        let result = processed.cellular_automata_result.unwrap();
        masks.capture(config.clone(), result.clone()).unwrap();
        masks
            .sessions
            .get_mut(&frame.source.path)
            .unwrap()
            .document
            .regions
            .push(ReviewRegion {
                class: ReviewClass::Uncertain,
                bounds: [0.0, 0.0, 0.5, 0.5],
            });
        masks.refresh(&ctx);
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            masks.show(ui, false, config.clone(), Some(&result), None, false);
        });
        assert!(!output.shapes.is_empty());
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
