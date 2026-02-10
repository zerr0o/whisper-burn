use eframe::egui;

pub fn apply_dark_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();

    let bg = egui::Color32::from_rgb(26, 26, 46); // #1a1a2e
    let widget_bg = egui::Color32::from_rgb(22, 33, 62); // #16213e
    let accent = egui::Color32::from_rgb(233, 69, 96); // #e94560
    let text = egui::Color32::from_rgb(234, 234, 234); // #eaeaea

    visuals.panel_fill = bg;
    visuals.window_fill = bg;
    visuals.faint_bg_color = widget_bg;
    visuals.extreme_bg_color = egui::Color32::from_rgb(16, 16, 32);

    visuals.override_text_color = Some(text);

    // Widget styling
    visuals.widgets.noninteractive.bg_fill = widget_bg;
    visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 40, 70));

    visuals.widgets.inactive.bg_fill = widget_bg;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(50, 50, 80));

    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(30, 44, 80);
    visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, accent);

    visuals.widgets.active.bg_fill = accent;
    visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, accent);

    visuals.selection.bg_fill = accent.linear_multiply(0.4);
    visuals.selection.stroke = egui::Stroke::new(1.0, accent);

    visuals.hyperlink_color = accent;

    ctx.set_visuals(visuals);
}
