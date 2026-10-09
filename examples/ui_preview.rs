//! Native presentation fixtures. No model, microphone, downloads, or saved settings.
//! cargo run --example ui-preview -- [ready|empty|models|recording|processing|choose|confirm|download|loading]
use eframe::egui;
use std::{sync::atomic::Ordering, time::Duration};
use whisper_burn::native::{
    config::AppConfig,
    download::{DownloadProgress, ModelVariant},
    hotkey::HotkeyCapture,
    ui::{
        download_screen, loading_screen, main_screen, model_manager_screen,
        status_indicator::AppStatus, theme,
    },
};

struct Preview {
    screen: String,
    config: AppConfig,
    capture: HotkeyCapture,
    progress: DownloadProgress,
    samples: Vec<f32>,
}

impl eframe::App for Preview {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if !self.capture.listening {
            for (key, screen) in [
                (egui::Key::F1, "ready"),
                (egui::Key::F2, "empty"),
                (egui::Key::F3, "models"),
                (egui::Key::F4, "recording"),
                (egui::Key::F5, "processing"),
                (egui::Key::F6, "choose"),
                (egui::Key::F7, "confirm"),
                (egui::Key::F8, "download"),
                (egui::Key::F9, "loading"),
            ] {
                if ctx.input(|input| input.key_pressed(key)) {
                    self.screen = screen.into();
                }
            }
            for (key, size) in [
                (egui::Key::F10, [640.0, 580.0]),
                (egui::Key::F11, [620.0, 540.0]),
                (egui::Key::F12, [780.0, 640.0]),
            ] {
                if ctx.input(|input| input.key_pressed(key)) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size.into()));
                }
            }
        }
        theme::footer(ctx);
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(24)).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                match self.screen.as_str() {
                    "models" => {
                        if matches!(model_manager_screen::draw(ui, ModelVariant::LargeV3), model_manager_screen::ModelManagerAction::Back) {
                            self.screen = "ready".into();
                        }
                    }
                    "recording" => main_screen::draw_recording(ui, &self.samples, 16000, Duration::from_millis(8400), "Ctrl + Win"),
                    "choose" => {
                        if matches!(download_screen::draw_choose_model(ui), download_screen::ChooseAction::Select(_)) { self.screen = "confirm".into(); }
                    }
                    "confirm" => match download_screen::draw_confirm(ui, ModelVariant::LargeV3) {
                        download_screen::ConfirmAction::Download => self.screen = "download".into(),
                        download_screen::ConfirmAction::Back => self.screen = "choose".into(),
                        _ => {}
                    },
                    "download" => download_screen::draw_progress(ui, &self.progress, ModelVariant::LargeV3),
                    "loading" => loading_screen::draw(ui, "Loading Whisper Large V3. This may take a minute..."),
                    _ => {
                        let text = if self.screen == "empty" { "" } else {
                            "Les idées viennent plus facilement quand on les dit à voix haute.\n\nPréparer le compte rendu de la réunion, puis partager les prochaines étapes avec l’équipe."
                        };
                        let status = if self.screen == "processing" { AppStatus::Processing } else if self.screen == "empty" { AppStatus::Ready } else { AppStatus::Done };
                        if matches!(main_screen::draw_ready(ui, text, 1270, ModelVariant::LargeV3, &mut self.config, status, &mut self.capture), main_screen::MainAction::OpenModelManager) {
                            self.screen = "models".into();
                        }
                    }
                }
            });
        });
    }
}

fn main() -> eframe::Result {
    let screen = std::env::args().nth(1).unwrap_or_else(|| "ready".into());
    let mut config = AppConfig::default();
    config.language = "fr".into();
    config.hotkey.modifiers = vec!["CONTROL".into(), "SUPER".into()];
    config.hotkey.key = String::new();
    config.auto_paste = true;
    config.auto_mute = true;
    let progress = DownloadProgress::default();
    progress.tokenizer_bytes.store(2_480_617, Ordering::Relaxed);
    progress.tokenizer_total.store(2_480_617, Ordering::Relaxed);
    progress.gguf_bytes.store(283_115_520, Ordering::Relaxed);
    progress.gguf_total.store(1_127_776_064, Ordering::Relaxed);
    let samples = (0..32000)
        .map(|i| {
            let t = i as f32 / 16000.0;
            (t * 800.0).sin() * (0.07 + (t * 8.0).sin().abs() * 0.45)
        })
        .collect();
    eframe::run_native(
        "Whisper Burn — UI preview",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([780.0, 640.0])
                .with_min_inner_size([620.0, 540.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            theme::apply_dark_theme(&cc.egui_ctx);
            Ok(Box::new(Preview {
                screen,
                config,
                capture: HotkeyCapture::new(),
                progress,
                samples,
            }))
        }),
    )
}
