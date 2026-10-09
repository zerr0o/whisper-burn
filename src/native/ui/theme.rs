use eframe::egui;

pub const BG: egui::Color32 = egui::Color32::from_rgb(18, 20, 24);
pub const SURFACE: egui::Color32 = egui::Color32::from_rgb(27, 30, 35);
pub const SURFACE_ALT: egui::Color32 = egui::Color32::from_rgb(34, 38, 45);
pub const BORDER: egui::Color32 = egui::Color32::from_rgb(52, 57, 66);
pub const TEXT: egui::Color32 = egui::Color32::from_rgb(241, 240, 237);
pub const MUTED: egui::Color32 = egui::Color32::from_rgb(174, 180, 191);
pub const DIM: egui::Color32 = egui::Color32::from_rgb(152, 160, 172);
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(233, 173, 135);
pub const ACCENT_BG: egui::Color32 = egui::Color32::from_rgb(34, 31, 29);
pub const GREEN: egui::Color32 = egui::Color32::from_rgb(141, 188, 162);
pub const RED: egui::Color32 = egui::Color32::from_rgb(229, 139, 139);

pub fn apply_dark_theme(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut headings = fonts.families[&egui::FontFamily::Proportional].clone();
    // Use the system font on Windows; keep egui's bundled fonts as fallbacks.
    if let Some(windows) = std::env::var_os("WINDIR") {
        let dir = std::path::PathBuf::from(windows).join("Fonts");
        for (name, file) in [
            ("interface", "segoeui.ttf"),
            ("interface-bold", "seguisb.ttf"),
        ] {
            if let Ok(bytes) = std::fs::read(dir.join(file)) {
                fonts
                    .font_data
                    .insert(name.into(), egui::FontData::from_owned(bytes).into());
                if name == "interface" {
                    fonts
                        .families
                        .get_mut(&egui::FontFamily::Proportional)
                        .unwrap()
                        .insert(0, name.into());
                } else {
                    headings.insert(0, name.into());
                }
            }
        }
    }
    fonts
        .families
        .insert(egui::FontFamily::Name("heading".into()), headings);
    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(13.0));
    style
        .text_styles
        .insert(egui::TextStyle::Small, egui::FontId::proportional(12.0));
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::new(22.0, egui::FontFamily::Name("heading".into())),
    );
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.button_padding = egui::vec2(14.0, 7.0);
    style.spacing.interact_size.y = 32.0;
    style.spacing.icon_width = 18.0;
    style.spacing.icon_spacing = 8.0;
    style.spacing.scroll = egui::style::ScrollStyle {
        foreground_color: true,
        bar_width: 4.0,
        bar_inner_margin: 8.0,
        ..egui::style::ScrollStyle::solid()
    };
    style.animation_time = 0.0;

    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = BG;
    visuals.window_fill = SURFACE;
    visuals.faint_bg_color = SURFACE_ALT;
    visuals.extreme_bg_color = BG;
    visuals.code_bg_color = SURFACE_ALT;
    visuals.override_text_color = Some(TEXT);
    visuals.window_corner_radius = 12.into();
    visuals.window_stroke = egui::Stroke::new(1.0_f32, BORDER);
    visuals.menu_corner_radius = 8.into();
    visuals.hyperlink_color = ACCENT;
    visuals.warn_fg_color = ACCENT;
    visuals.error_fg_color = RED;
    for widget in [
        &mut visuals.widgets.noninteractive,
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        widget.bg_fill = SURFACE_ALT;
        widget.weak_bg_fill = SURFACE_ALT;
        widget.bg_stroke = egui::Stroke::new(1.0_f32, BORDER);
        widget.fg_stroke = egui::Stroke::new(1.5_f32, MUTED);
        widget.corner_radius = 7.into();
        widget.expansion = 0.0;
    }
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(45, 49, 57);
    visuals.widgets.hovered.weak_bg_fill = visuals.widgets.hovered.bg_fill;
    visuals.widgets.hovered.bg_stroke.color = ACCENT;
    visuals.widgets.hovered.fg_stroke.color = ACCENT;
    visuals.widgets.active.bg_stroke.color = ACCENT;
    visuals.selection.bg_fill = ACCENT_BG;
    visuals.selection.stroke = egui::Stroke::new(1.5_f32, ACCENT);
    style.visuals = visuals;
    ctx.set_style(style);

    if let Ok(icon) =
        eframe::icon_data::from_png_bytes(include_bytes!("../../../assets/app-icon.png"))
    {
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [icon.width as usize, icon.height as usize],
            &icon.rgba,
        );
        let texture = ctx.load_texture("brand-icon", image, egui::TextureOptions::LINEAR);
        ctx.data_mut(|data| data.insert_temp(egui::Id::new("brand-icon"), texture));
    }
}

/// A fixed-length activity arc stays visible at every animation phase.
pub fn spinner(ui: &mut egui::Ui) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::hover());
    response.widget_info(|| egui::WidgetInfo::new(egui::WidgetType::ProgressIndicator));
    if ui.is_rect_visible(rect) {
        let angle = ui.input(|i| i.time) * std::f64::consts::TAU;
        let points = (0..=24)
            .map(|step| {
                let (sin, cos) = (angle + step as f64 / 24.0 * 240_f64.to_radians()).sin_cos();
                rect.center() + 12.0 * egui::vec2(cos as f32, sin as f32)
            })
            .collect();
        ui.painter().add(egui::Shape::line(
            points,
            egui::Stroke::new(2.5_f32, ACCENT),
        ));
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(33));
    }
}

pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .corner_radius(12)
        .inner_margin(20)
}

pub fn heading(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text)
        .font(egui::FontId::new(
            22.0,
            egui::FontFamily::Name("heading".into()),
        ))
        .color(TEXT)
}

pub fn eyebrow(text: &str) -> egui::RichText {
    egui::RichText::new(text).size(12.0).color(MUTED)
}

pub fn primary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(text).color(BG))
            .fill(ACCENT)
            .stroke(egui::Stroke::NONE)
            .min_size(egui::vec2(0.0, 36.0)),
    )
}

pub fn secondary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(egui::Button::new(text).min_size(egui::vec2(0.0, 34.0)))
}

pub fn brand(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        if let Some(texture) = ui
            .ctx()
            .data(|data| data.get_temp::<egui::TextureHandle>(egui::Id::new("brand-icon")))
        {
            ui.image((texture.id(), egui::vec2(36.0, 36.0)));
        }
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 1.0;
            ui.label(heading("Whisper Burn").size(21.0));
            ui.label(
                egui::RichText::new("Local speech to text")
                    .size(12.0)
                    .color(MUTED),
            );
        });
    });
}

pub fn footer(ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("app_footer")
        .default_height(38.0)
        .frame(
            egui::Frame::new()
                .fill(BG)
                .inner_margin(egui::Margin::symmetric(24, 10)),
        )
        .show(ctx, |ui| {
            ui.spacing_mut().interact_size.y = 16.0;
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Local inference. Your audio stays on this device.")
                        .size(12.0)
                        .color(DIM),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                            .size(12.0)
                            .color(DIM),
                    );
                });
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_arc_stays_visible_at_all_phases() {
        let ctx = egui::Context::default();
        for time in [0.0, 0.2, 1.57, 3.14, 6.28] {
            let output = ctx.run(
                egui::RawInput {
                    time: Some(time),
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| spinner(ui));
                },
            );
            let arc = output
                .shapes
                .iter()
                .find(|s| {
                    matches!(&s.shape,
                egui::Shape::Path(path) if path.points.len() == 25)
                })
                .expect("activity arc");
            let bounds = arc.shape.visual_bounding_rect();
            assert!(
                bounds.width() >= 19.0 && bounds.height() >= 19.0,
                "Activity indicator collapsed at phase {time}: {bounds:?}"
            );
        }
    }
}
