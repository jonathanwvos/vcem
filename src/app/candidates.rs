use super::*;
use crate::discrete_candidates::{CandidateRequest, DiscreteCandidate};

struct CandidateJob {
    receiver: Receiver<Result<Vec<DiscreteCandidate>, String>>,
    cancel: Arc<AtomicBool>,
    discard: bool,
}
#[derive(Default)]
pub(super) struct CandidateUi {
    selecting: bool,
    start: Option<[f32; 2]>,
    region: Option<[usize; 4]>,
    candidates: Vec<DiscreteCandidate>,
    textures: Vec<egui::TextureHandle>,
    notice: Option<String>,
    job: Option<CandidateJob>,
}
impl CandidateUi {
    pub(super) fn invalidate(&mut self) {
        if let Some(job) = &mut self.job {
            job.cancel.store(true, Ordering::Relaxed);
            job.discard = true;
        }
        self.selecting = false;
        self.start = None;
        self.region = None;
        self.candidates.clear();
        self.textures.clear();
        self.notice = None;
    }
    pub(super) fn selecting(&self) -> bool {
        self.selecting
    }
    fn busy(&self) -> bool {
        self.job.is_some()
    }
    pub(super) fn poll(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else {
            return;
        };
        let result = match job.receiver.try_recv() {
            Ok(value) => value,
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(50));
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("Candidate worker stopped. Try generating again.".into())
            }
        };
        let discard = job.discard || job.cancel.load(Ordering::Relaxed);
        self.job = None;
        if discard {
            return;
        }
        match result {
            Ok(candidates) => {
                self.textures = candidates
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        ctx.load_texture(
                            format!("CA candidate {i}"),
                            egui::ColorImage::from_rgba_unmultiplied(
                                c.preview_size,
                                &c.preview_rgba,
                            ),
                            TextureOptions::NEAREST,
                        )
                    })
                    .collect();
                self.candidates = candidates;
                self.notice=Some("Preview a candidate on the full image, then capture its Lab state as a mask layer.".into());
            }
            Err(error) => self.notice = Some(error),
        }
    }
    fn start_job(&mut self, request: CandidateRequest, ctx: &egui::Context) {
        if self.job.is_some() {
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (sender, receiver) = mpsc::channel();
        let ctx = ctx.clone();
        thread::spawn(move || {
            let result = crate::discrete_candidates::generate(request, &worker_cancel);
            let _ = sender.send(result);
            ctx.request_repaint();
        });
        self.candidates.clear();
        self.textures.clear();
        self.selecting = false;
        self.start = None;
        self.notice = None;
        self.job = Some(CandidateJob {
            receiver,
            cancel,
            discard: false,
        });
    }
    pub(super) fn region_pointer(
        &mut self,
        response: &egui::Response,
        image_rect: Rect,
        size: [usize; 2],
    ) {
        let normalized = |p: Pos2| {
            [
                ((p.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0),
                ((p.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0),
            ]
        };
        if response.drag_started_by(PointerButton::Primary)
            && let Some(p) = response
                .ctx
                .input(|i| i.pointer.press_origin())
                .filter(|p| image_rect.contains(*p))
        {
            self.start = Some(normalized(p));
        }
        if let (Some(a), Some(p)) = (self.start, response.interact_pointer_pos()) {
            let b = normalized(p);
            self.region = Some([
                (a[0].min(b[0]) * size[0] as f32).floor() as usize,
                (a[1].min(b[1]) * size[1] as f32).floor() as usize,
                ((a[0].max(b[0]) * size[0] as f32).ceil() as usize).min(size[0]),
                ((a[1].max(b[1]) * size[1] as f32).ceil() as usize).min(size[1]),
            ]);
        }
        if response.drag_stopped_by(PointerButton::Primary) {
            if self.start.take().is_some() {
                self.selecting = false;
                self.candidates.clear();
                self.textures.clear();
                self.notice = None;
            }
        }
    }
    pub(super) fn paint(&self, painter: &egui::Painter, image_rect: Rect, size: [usize; 2]) {
        if let Some([l, t, r, b]) = self.region {
            let bounds = Rect::from_min_max(
                image_rect.min
                    + Vec2::new(
                        l as f32 / size[0] as f32 * image_rect.width(),
                        t as f32 / size[1] as f32 * image_rect.height(),
                    ),
                image_rect.min
                    + Vec2::new(
                        r as f32 / size[0] as f32 * image_rect.width(),
                        b as f32 / size[1] as f32 * image_rect.height(),
                    ),
            );
            painter.rect_stroke(
                bounds,
                0.0,
                Stroke::new(2.0, Color32::from_rgb(255, 184, 54)),
                egui::StrokeKind::Inside,
            );
            painter.text(
                bounds.left_top(),
                egui::Align2::LEFT_BOTTOM,
                "Candidate example",
                egui::FontId::proportional(12.0),
                Color32::from_rgb(255, 184, 54),
            );
        }
    }
}
impl VCEMApp {
    pub(super) fn show_candidate_controls(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new("Candidates from region").default_open(true).show(ui,|ui|{
            ui.small("Draw an example region. Candidates learn local state patterns that are more common inside it than in the surrounding image.");
            let busy=self.processing_job.is_some()||self.automata_lab_job.is_some()||self.automata_lab_playing||self.masks.is_busy();
            ui.horizontal_wrapped(|ui|{
                let label=if self.candidates.selecting {"Cancel region selection"}else{"Select example region"};
                if ui.add_enabled(!busy&&!self.candidates.busy()&&self.displayed_image.is_some(),Button::new(label)).clicked(){
                    self.candidates.selecting=!self.candidates.selecting;self.candidates.start=None;
                    self.clear_histogram_highlight();
                }
                if ui.add_enabled(!self.candidates.busy(),Button::new("Clear region")).clicked(){self.candidates.invalidate();}
            });
            if self.candidates.selecting {ui.label("Drag a rectangle on the image.");}
            if let Some([l,t,r,b])=self.candidates.region {ui.small(format!("Example: {} × {} pixels",r-l,b-t));}
            if ui.add_enabled(!busy&&!self.candidates.busy()&&!self.candidates.selecting&&self.candidates.region.is_some(),Button::new("Generate candidates")).clicked() {
                if let (Some(displayed),Some(region))=(&self.displayed_image,self.candidates.region) {
                    self.candidates.start_job(CandidateRequest{
                        values:displayed.prepared_values.clone(),calibration:displayed.calibration,size:[displayed.size.x as usize,displayed.size.y as usize],region,
                        kernel_size:self.automata_lab_kernel_grid_size,state_count:self.automata_lab_discrete_settings.state_count,seed:self.lab_random_seed,
                        current_kernel:self.automata_lab_discrete_settings.kernel.clone(),
                    },ui.ctx());
                }
            }
            if self.candidates.busy() {
                ui.horizontal(|ui|{ui.spinner();ui.label("Fitting candidate rules…");});
                if ui.button("Cancel generation").clicked(){
                    if let Some(job)=&self.candidates.job{job.cancel.store(true,Ordering::Relaxed);}
                    self.candidates.notice=Some("Candidate generation cancelled.".into());
                }
            }
            if let Some(notice)=&self.candidates.notice{ui.label(notice);}
            let mut selected=None;
            for (i,candidate) in self.candidates.candidates.iter().enumerate() {
                ui.push_id(("candidate",i),|ui|{
                    ui.separator();ui.strong(&candidate.name);
                    if let Some(texture)=self.candidates.textures.get(i) {ui.image((texture.id(),Vec2::new(candidate.preview_size[0] as f32,candidate.preview_size[1] as f32)));}
                    ui.small(format!("Example {:.0}% · surroundings {:.0}%",candidate.region_coverage*100.0,candidate.background_coverage*100.0));
                    ui.small("One generation · teal mask · amber example border");
                    if ui.add_enabled(!busy,Button::new("Preview on image")).clicked(){selected=Some(i);}
                });
            }
            if !self.candidates.candidates.is_empty(){ui.small("Coverage is sampled on the fitting region, not measured accuracy. Preview on image replaces the current Lab run; captured layers remain available.");}
            if let Some(index)=selected {
                let candidate=self.candidates.candidates[index].clone();
                self.set_lab_draft_model(LabModelSettings::Discrete(candidate.settings));
                self.initialize_automata_lab(ui.ctx());
                self.step_automata_lab(ui.ctx(),1);
                self.masks.suggest_name(format!("Candidate: {}",candidate.name));
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> CandidateRequest {
        CandidateRequest {
            values: (0..32 * 24)
                .map(|i| {
                    if (8..24).contains(&(i % 32)) && (6..18).contains(&(i / 32)) {
                        255
                    } else {
                        0
                    }
                })
                .collect(),
            calibration: crate::quantization::IntensityCalibration::full_scale(256).unwrap(),
            size: [32, 24],
            region: [8, 6, 24, 18],
            kernel_size: 3,
            state_count: 4,
            seed: 1,
            current_kernel: Vec::new(),
        }
    }
    fn finish(workspace: &mut CandidateUi, ctx: &egui::Context) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while workspace.busy() {
            assert!(Instant::now() < deadline);
            workspace.poll(ctx);
            thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn candidate_jobs_publish_previews_but_discard_results_after_image_changes() {
        let ctx = egui::Context::default();
        let mut workspace = CandidateUi::default();
        workspace.region = Some([8, 6, 24, 18]);
        workspace.start_job(request(), &ctx);
        finish(&mut workspace, &ctx);
        assert!(!workspace.candidates.is_empty());
        assert_eq!(workspace.candidates.len(), workspace.textures.len());
        workspace.start_job(request(), &ctx);
        workspace.invalidate();
        finish(&mut workspace, &ctx);
        assert!(workspace.candidates.is_empty());
        assert!(workspace.textures.is_empty());
        assert!(workspace.region.is_none());
    }
    #[test]
    fn selecting_an_example_uses_press_origin_and_image_coordinates() {
        let ctx = egui::Context::default();
        let mut workspace = CandidateUi::default();
        workspace.selecting = true;
        let image_rect = Rect::from_min_size(Pos2::new(50.0, 50.0), Vec2::new(200.0, 100.0));
        let start = Pos2::new(70.0, 70.0);
        let end = Pos2::new(190.0, 130.0);
        for events in [
            vec![],
            vec![
                egui::Event::PointerMoved(start),
                egui::Event::PointerButton {
                    pos: start,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                },
            ],
            vec![egui::Event::PointerMoved(end)],
            vec![egui::Event::PointerButton {
                pos: end,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Default::default(),
            }],
        ] {
            let _ = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(500.0, 300.0))),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let (_, response) =
                        ui.allocate_exact_size(Vec2::new(400.0, 200.0), Sense::click_and_drag());
                    workspace.region_pointer(&response, image_rect, [100, 50]);
                },
            );
        }
        assert_eq!(workspace.region, Some([10, 10, 70, 40]));
        assert!(!workspace.selecting);
    }
}
