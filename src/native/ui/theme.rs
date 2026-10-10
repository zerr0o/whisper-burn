use eframe::egui;

pub const BG: egui::Color32 = egui::Color32::from_rgb(16, 17, 20);
pub const SURFACE: egui::Color32 = egui::Color32::from_rgb(23, 25, 29);
pub const SURFACE_ALT: egui::Color32 = egui::Color32::from_rgb(31, 34, 39);
pub const BORDER: egui::Color32 = egui::Color32::from_rgb(44, 48, 55);
pub const BORDER_STRONG: egui::Color32 = egui::Color32::from_rgb(66, 71, 80);
pub const TEXT: egui::Color32 = egui::Color32::from_rgb(230, 231, 234);
pub const MUTED: egui::Color32 = egui::Color32::from_rgb(160, 166, 176);
pub const DIM: egui::Color32 = egui::Color32::from_rgb(124, 131, 142);
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(224, 146, 98);
pub const ACCENT_BG: egui::Color32 = egui::Color32::from_rgb(58, 42, 33);
pub const GREEN: egui::Color32 = egui::Color32::from_rgb(92, 184, 128);
pub const RED: egui::Color32 = egui::Color32::from_rgb(235, 92, 92);

/// Cards and popups.
pub const RADIUS: u8 = 4;
/// Buttons, inputs, key caps and badges.
pub const CONTROL_RADIUS: u8 = 3;

fn semibold(size: f32) -> egui::FontId {
    egui::FontId::new(size, egui::FontFamily::Name("heading".into()))
}

pub fn apply_dark_theme(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let mut headings = fonts.families[&egui::FontFamily::Proportional].clone();
    // Use the system fonts on Windows; keep egui's bundled fonts as fallbacks.
    if let Some(windows) = std::env::var_os("WINDIR") {
        let dir = std::path::PathBuf::from(windows).join("Fonts");
        for (name, files) in [
            ("interface", &["segoeui.ttf"][..]),
            ("interface-semibold", &["seguisb.ttf"][..]),
            ("mono", &["CascadiaMono.ttf", "consola.ttf"][..]),
        ] {
            let Some(bytes) = files
                .iter()
                .find_map(|file| std::fs::read(dir.join(file)).ok())
            else {
                continue;
            };
            fonts
                .font_data
                .insert(name.into(), egui::FontData::from_owned(bytes).into());
            match name {
                "interface" => fonts
                    .families
                    .get_mut(&egui::FontFamily::Proportional)
                    .unwrap()
                    .insert(0, name.into()),
                "interface-semibold" => headings.insert(0, name.into()),
                _ => fonts
                    .families
                    .get_mut(&egui::FontFamily::Monospace)
                    .unwrap()
                    .insert(0, name.into()),
            }
        }
    }
    fonts
        .families
        .insert(egui::FontFamily::Name("heading".into()), headings);
    ctx.set_fonts(fonts);

    let mut style = (*ctx.style()).clone();
    for (text_style, font) in [
        (egui::TextStyle::Body, egui::FontId::proportional(13.0)),
        (egui::TextStyle::Button, egui::FontId::proportional(13.0)),
        (egui::TextStyle::Small, egui::FontId::proportional(11.5)),
        (egui::TextStyle::Monospace, egui::FontId::monospace(12.5)),
        (egui::TextStyle::Heading, semibold(18.0)),
    ] {
        style.text_styles.insert(text_style, font);
    }
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 4.0);
    style.spacing.interact_size.y = 28.0;
    style.spacing.icon_width = 15.0;
    style.spacing.icon_width_inner = 9.0;
    style.spacing.icon_spacing = 7.0;
    style.spacing.menu_margin = egui::Margin::same(4);
    style.spacing.scroll = egui::style::ScrollStyle {
        bar_width: 4.0,
        bar_inner_margin: 6.0,
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
    visuals.window_corner_radius = RADIUS.into();
    visuals.menu_corner_radius = RADIUS.into();
    visuals.window_stroke = egui::Stroke::new(1.0_f32, BORDER_STRONG);
    let shadow = egui::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: egui::Color32::from_black_alpha(110),
    };
    visuals.window_shadow = shadow;
    visuals.popup_shadow = shadow;
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
        widget.corner_radius = CONTROL_RADIUS.into();
        widget.expansion = 0.0;
    }
    visuals.widgets.inactive.fg_stroke.color = TEXT;
    for widget in [
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
    ] {
        widget.bg_fill = egui::Color32::from_rgb(38, 41, 47);
        widget.weak_bg_fill = widget.bg_fill;
        widget.bg_stroke.color = BORDER_STRONG;
        widget.fg_stroke.color = TEXT;
    }
    visuals.selection.bg_fill = ACCENT_BG;
    visuals.selection.stroke = egui::Stroke::new(1.0_f32, ACCENT);
    style.visuals = visuals;
    ctx.set_style(style);

    if let Ok(icon) =
        eframe::icon_data::from_png_bytes(include_bytes!("../../../assets/app-icon.png"))
    {
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [icon.width as usize, icon.height as usize],
            &icon.rgba,
        );
        // Mipmaps keep the 512 px icon clean at app-bar size.
        let options = egui::TextureOptions {
            mipmap_mode: Some(egui::TextureFilter::Linear),
            ..egui::TextureOptions::LINEAR
        };
        let texture = ctx.load_texture("brand-icon", image, options);
        ctx.data_mut(|data| data.insert_temp(egui::Id::new("brand-icon"), texture));
    }
}

/// A fixed-length activity arc stays visible at every animation phase.
pub fn spinner(ui: &mut egui::Ui, size: f32) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    response.widget_info(|| egui::WidgetInfo::new(egui::WidgetType::ProgressIndicator));
    if ui.is_rect_visible(rect) {
        let angle = ui.input(|i| i.time) * std::f64::consts::TAU;
        let radius = size / 2.0 - 2.0;
        let points = (0..=24)
            .map(|step| {
                let (sin, cos) = (angle + step as f64 / 24.0 * 240_f64.to_radians()).sin_cos();
                rect.center() + radius * egui::vec2(cos as f32, sin as f32)
            })
            .collect();
        ui.painter().add(egui::Shape::line(
            points,
            egui::Stroke::new(2.0_f32, ACCENT),
        ));
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(33));
    }
}

pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(egui::Stroke::new(1.0_f32, BORDER))
        .corner_radius(RADIUS)
        .inner_margin(14)
}

/// Page title, 18 pt semibold.
pub fn heading(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).font(semibold(18.0)).color(TEXT)
}

/// Emphasised label, e.g. a model name or a panel title.
pub fn title(text: impl Into<String>, size: f32) -> egui::RichText {
    egui::RichText::new(text).font(semibold(size)).color(TEXT)
}

/// Section label above a group of controls.
pub fn section(text: &str) -> egui::RichText {
    egui::RichText::new(text).font(semibold(12.0)).color(MUTED)
}

/// Secondary metadata: model names, durations, sizes.
pub fn meta(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).size(12.0).color(DIM)
}

pub fn primary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(text).font(semibold(13.0)).color(BG))
            .fill(ACCENT)
            .stroke(egui::Stroke::NONE)
            .min_size(egui::vec2(0.0, 28.0)),
    )
}

pub fn secondary_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(egui::Button::new(text).min_size(egui::vec2(0.0, 28.0)))
}

/// Text-only button for low-emphasis or destructive actions.
pub fn quiet_button(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add(
        egui::Button::new(egui::RichText::new(text).color(MUTED))
            .fill(egui::Color32::TRANSPARENT)
            .stroke(egui::Stroke::NONE)
            .min_size(egui::vec2(0.0, 28.0)),
    )
}

/// Small outlined status label.
pub fn badge(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    let galley =
        ui.painter()
            .layout_no_wrap(text.to_owned(), egui::FontId::proportional(11.0), color);
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(galley.size().x + 12.0, 18.0),
        egui::Sense::hover(),
    );
    ui.painter().rect_stroke(
        rect,
        CONTROL_RADIUS,
        egui::Stroke::new(1.0_f32, color.gamma_multiply(0.5)),
        egui::StrokeKind::Inside,
    );
    ui.painter()
        .galley(rect.center() - galley.size() / 2.0, galley, color);
}

/// App bar shared by every screen: icon and name on the left, `right` aligned right.
pub fn top_bar(ui: &mut egui::Ui, right: impl FnOnce(&mut egui::Ui)) {
    ui.allocate_ui_with_layout(
        egui::vec2(ui.available_width(), 28.0),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.set_min_size(egui::vec2(ui.available_width(), 28.0));
            if let Some(texture) = ui
                .ctx()
                .data(|data| data.get_temp::<egui::TextureHandle>(egui::Id::new("brand-icon")))
            {
                ui.image((texture.id(), egui::vec2(20.0, 20.0)));
            }
            ui.label(title("Whisper Burn", 14.0));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), right);
        },
    );
}

/// Title and one-line description for setup and library screens.
pub fn page_title(ui: &mut egui::Ui, text: &str, subtitle: &str) {
    ui.label(heading(text));
    ui.add_space(2.0);
    ui.label(egui::RichText::new(subtitle).color(MUTED));
}

pub fn error_banner(ui: &mut egui::Ui, message: &str) {
    egui::Frame::new()
        .fill(RED.gamma_multiply(0.08))
        .stroke(egui::Stroke::new(1.0_f32, RED.gamma_multiply(0.5)))
        .corner_radius(CONTROL_RADIUS)
        .inner_margin(egui::Margin::symmetric(10, 6))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(egui::RichText::new(message).color(RED));
        });
}

pub fn footer(ctx: &egui::Context) {
    egui::TopBottomPanel::bottom("app_footer")
        .default_height(30.0)
        .frame(
            egui::Frame::new()
                .fill(BG)
                .inner_margin(egui::Margin::symmetric(20, 7)),
        )
        .show(ctx, |ui| {
            ui.spacing_mut().interact_size.y = 16.0;
            ui.horizontal(|ui| {
                ui.label(meta("Local inference. Audio stays on this device.").size(11.5));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(meta(concat!("v", env!("CARGO_PKG_VERSION"))).size(11.5));
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
                    egui::CentralPanel::default().show(ctx, |ui| spinner(ui, 20.0));
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
                bounds.width() >= 12.0 && bounds.height() >= 12.0,
                "Activity indicator collapsed at phase {time}: {bounds:?}"
            );
        }
    }
}
