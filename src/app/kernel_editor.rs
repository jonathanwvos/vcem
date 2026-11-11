use super::*;

/// Shared composition controls for Inspect and discrete Lab kernels.
pub(super) fn show(
    ui: &mut Ui,
    settings: &mut CellularAutomataSettings,
    preset: &mut CellularKernelPreset,
    angle: &mut f32,
    brush: &mut CellularKernelBrush,
    drag_brush: &mut Option<CellularKernelBrush>,
) {
    let old_size = settings.kernel_size;
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
        resize_cellular_kernel(settings, new_size);
        *preset = CellularKernelPreset::Custom;
    }

    let previous_preset = *preset;
    egui::ComboBox::from_label("Preset")
        .selected_text(preset.label())
        .show_ui(ui, |ui| {
            for choice in CellularKernelPreset::ALL {
                ui.selectable_value(preset, choice, choice.label());
            }
        })
        .response
        .on_hover_text("Seeds the kernel editor with common sensing shapes.");
    if *preset != previous_preset {
        apply_cellular_kernel_preset(settings, *preset, *angle);
    }
    if matches!(
        *preset,
        CellularKernelPreset::Line | CellularKernelPreset::Crescent
    ) {
        let angle_changed = ui
            .add(
                egui::Slider::new(angle, 0.0..=180.0)
                    .text("Angle")
                    .suffix(" degrees"),
            )
            .on_hover_text("Rotates directional presets before applying the filter.")
            .changed();
        if angle_changed {
            apply_cellular_kernel_preset(settings, *preset, *angle);
        }
    }
    ui.checkbox(
                &mut settings.omni_directional,
                "Omni-directional pass",
            )
            .on_hover_text(
                "Tests cardinal rotations of asymmetric kernels in each iteration and keeps the strongest response.",
            );

    ui.add_space(4.0);
    let size = settings.kernel_size;
    let palette_width = ui.spacing().interact_size.x + 8.0;
    let cell_size =
        ((ui.available_width() - palette_width).max(0.0) / size as f32 - 2.0).clamp(8.0, 24.0);
    let primary_down = ui.input(|input| input.pointer.button_down(PointerButton::Primary));
    if !primary_down {
        *drag_brush = None;
    }
    let pointer_pos = ui.input(|input| input.pointer.latest_pos());

    ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    for y in 0..size {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::splat(2.0);
                            ui.spacing_mut().button_padding = Vec2::ZERO;
                            ui.spacing_mut().interact_size = Vec2::splat(cell_size);
                            ui.style_mut().override_font_id = Some(egui::FontId::proportional(cell_size * 0.65));
                            for x in 0..size {
                                let index = y * size + x;
                                let weight = settings.kernel[index];
                                let response = ui
                                    .add_sized(
                                        [cell_size, cell_size],
                                        kernel_weight_button(weight).sense(Sense::click_and_drag()),
                                    )
                                    .on_hover_text(format!("Offset ({}, {}). Click toggles the selected sign; drag paints or erases. Right click cycles in reverse.", x as i32 - (size/2) as i32, y as i32 - (size/2) as i32));
                                if x == size/2 && y == size/2 {
                                    ui.painter().rect_stroke(response.rect,0.0,Stroke::new(1.0,Color32::WHITE),egui::StrokeKind::Inside);
                                }

                                if response.clicked_by(PointerButton::Primary) {
                                    settings.kernel[index] =
                                        toggle_kernel_weight(weight, *brush);
                                    *preset = CellularKernelPreset::Custom;
                                } else if response.clicked_by(PointerButton::Secondary) {
                                    settings.kernel[index] =
                                        next_kernel_weight(weight, true);
                                    *preset = CellularKernelPreset::Custom;
                                }

                                if response.drag_started_by(PointerButton::Primary) {
                                    let paint = toggled_kernel_brush(
                                        weight,
                                        *brush,
                                    );
                                    *drag_brush = Some(paint);
                                    settings.kernel[index] =
                                        paint.weight();
                                    *preset = CellularKernelPreset::Custom;
                                }

                                if primary_down
                                    && pointer_pos.is_some_and(|pos| response.rect.contains(pos))
                                    && let Some(brush) = *drag_brush
                                {
                                    settings.kernel[index] = brush.weight();
                                    *preset = CellularKernelPreset::Custom;
                                }
                            }
                        });
                    }
                });

                ui.add_space(6.0);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 2.0;
                    ui.spacing_mut().button_padding = Vec2::ZERO;
                    ui.spacing_mut().interact_size = Vec2::splat(cell_size);
                    ui.style_mut().override_font_id = Some(egui::FontId::proportional(cell_size * 0.65));
                    for paint in CellularKernelBrush::ALL {
                        let response = ui
                            .add_sized(
                                [cell_size, cell_size],
                                kernel_brush_button(paint, *brush == paint),
                            )
                            .on_hover_text(paint.hover_text());
                        if response.clicked() {
                            *brush = paint;
                        }
                    }
                });
            });
}
