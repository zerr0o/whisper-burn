fn main() -> eframe::Result {
    tracing_subscriber::fmt().with_target(false).init();

    let config = whisper_burn::native::config::load_config();
    let start_minimized = config.start_minimized;

    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([700.0, 550.0])
        .with_title("Whisper Burn");

    if start_minimized {
        viewport = viewport.with_visible(false);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "Whisper Burn",
        options,
        Box::new(|cc| {
            // Apply dark theme
            whisper_burn::native::ui::theme::apply_dark_theme(&cc.egui_ctx);

            // Extract HWND on Windows
            let hwnd = extract_hwnd(cc);

            Ok(Box::new(whisper_burn::native::app::NativeApp::new(hwnd)))
        }),
    )
}

fn extract_hwnd(_cc: &eframe::CreationContext<'_>) -> Option<isize> {
    #[cfg(windows)]
    {
        unsafe {
            let title = windows::core::w!("Whisper Burn");
            if let Ok(hwnd) = windows::Win32::UI::WindowsAndMessaging::FindWindowW(None, title) {
                if !hwnd.is_invalid() {
                    return Some(hwnd.0 as isize);
                }
            }
        }
        None
    }
    #[cfg(not(windows))]
    {
        None
    }
}
