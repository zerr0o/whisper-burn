fn main() -> eframe::Result {
    tracing_subscriber::fmt().with_target(false).init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([700.0, 500.0])
            .with_title("Whisper Burn"),
        ..Default::default()
    };

    eframe::run_native(
        "Whisper Burn",
        options,
        Box::new(|_cc| Ok(Box::new(whisper_burn::native::app::NativeApp::new()))),
    )
}
