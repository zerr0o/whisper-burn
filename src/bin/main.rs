#![windows_subsystem = "windows"]

fn main() -> eframe::Result {
    tracing_subscriber::fmt().with_target(false).init();

    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../../assets/app-icon.png"))
        .expect("embedded app icon should be a valid PNG");

    let viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([700.0, 550.0])
        .with_title("Whisper Burn")
        .with_icon(icon);

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "Whisper Burn",
        options,
        Box::new(|cc| {
            whisper_burn::native::ui::theme::apply_dark_theme(&cc.egui_ctx);
            Ok(Box::new(whisper_burn::native::app::NativeApp::new()))
        }),
    )
}
