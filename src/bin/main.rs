#![windows_subsystem = "windows"]

fn main() -> eframe::Result {
    tracing_subscriber::fmt().with_target(false).init();

    let viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([700.0, 550.0])
        .with_title("Whisper Burn");

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
