use eframe::egui;
use std::time::Duration;

use crate::native::config::AppConfig;
use crate::native::download::ModelVariant;
use crate::native::hotkey::{self, HotkeyCapture};
use crate::native::ui::status_indicator::{self, AppStatus};
use crate::ALL_LANGUAGES;
use super::waveform;

pub enum MainAction {
    None,
    HotkeyChanged,
    ConfigChanged,
    OpenModelManager,
}

pub fn draw_ready(
    ui: &mut egui::Ui,
    last_result: &str,
    last_inference_ms: u128,
    variant: ModelVariant,
    config: &mut AppConfig,
    status: AppStatus,
    hotkey_capture: &mut HotkeyCapture,
) -> MainAction {
    let mut action = MainAction::None;

    ui.vertical_centered(|ui| {
        // Title bar: "Whisper Burn" + status on same line
        ui.horizontal(|ui| {
            ui.heading("Whisper Burn");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                status_indicator::draw_status(ui, status);
            });
        });

        // Subtitle: model name
        ui.label(
            egui::RichText::new(format!("Model: {}", variant.display_name()))
                .size(13.0)
                .color(egui::Color32::from_gray(140)),
        );

        ui.add_space(16.0);

        // Hotkey instruction (large)
        let hotkey_display = hotkey::HotkeyState::display_string(config);
        ui.label(
            egui::RichText::new(format!("Hold {} to record", hotkey_display))
                .size(24.0)
                .color(egui::Color32::from_rgb(180, 180, 200)),
        );

        ui.add_space(20.0);

        // Transcription zone
        if !last_result.is_empty() {
            ui.group(|ui| {
                ui.set_min_width(500.0);

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Last transcription")
                            .size(14.0)
                            .color(egui::Color32::from_gray(140)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Copy").clicked() {
                            ui.ctx().copy_text(last_result.to_string());
                        }
                        ui.label(
                            egui::RichText::new(format!("{last_inference_ms} ms"))
                                .size(12.0)
                                .color(egui::Color32::from_gray(120)),
                        );
                    });
                });

                ui.add_space(4.0);
                let mut text = last_result.to_string();
                ui.add(
                    egui::TextEdit::multiline(&mut text)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY)
                        .interactive(false),
                );
            });
        } else {
            ui.group(|ui| {
                ui.set_min_width(500.0);
                ui.set_min_height(60.0);
                ui.vertical_centered(|ui| {
                    ui.add_space(18.0);
                    ui.label(
                        egui::RichText::new("Transcription will appear here")
                            .size(14.0)
                            .color(egui::Color32::from_gray(90)),
                    );
                });
            });
        }

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(8.0);

        // Hotkey config: display + Change button / listener
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Hotkey:")
                    .size(14.0)
                    .color(egui::Color32::from_gray(160)),
            );

            if hotkey_capture.listening {
                // Poll keys and check for capture
                if let Some((mods, key)) = hotkey_capture.poll() {
                    config.hotkey.modifiers = mods;
                    config.hotkey.key = key;
                    action = MainAction::HotkeyChanged;
                }

                // Show what's currently being pressed
                let display = hotkey_capture.current_display();
                ui.label(
                    egui::RichText::new(&display)
                        .size(14.0)
                        .color(egui::Color32::from_rgb(233, 69, 96)),
                );

                if ui.button("Cancel").clicked() {
                    hotkey_capture.listening = false;
                }

                // Request frequent repaints during capture
                ui.ctx().request_repaint();
            } else {
                ui.label(
                    egui::RichText::new(&hotkey_display)
                        .size(14.0),
                );

                if ui.button("Change").clicked() {
                    hotkey_capture.start();
                }
            }
        });

        ui.add_space(4.0);

        // Language selector + Model Manager button
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Language:")
                    .size(14.0)
                    .color(egui::Color32::from_gray(160)),
            );

            let current_name = ALL_LANGUAGES
                .iter()
                .find(|l| l.code.unwrap_or("auto") == config.language)
                .map(|l| l.name)
                .unwrap_or("Auto");

            egui::ComboBox::from_id_salt("lang_selector")
                .selected_text(current_name)
                .height(300.0)
                .show_ui(ui, |ui| {
                    for lang in &ALL_LANGUAGES {
                        let code = lang.code.unwrap_or("auto");
                        let label = if lang.code.is_some() {
                            format!("{} ({})", lang.name, code)
                        } else {
                            lang.name.to_string()
                        };
                        if ui.selectable_label(config.language == code, &label).clicked() {
                            config.language = code.to_string();
                            action = MainAction::ConfigChanged;
                        }
                    }
                });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("\u{1F4E6} Models").on_hover_text("Model Manager").clicked() {
                    action = MainAction::OpenModelManager;
                }
            });
        });

        ui.add_space(4.0);

        // Toggles inline
        ui.horizontal(|ui| {
            if ui.checkbox(&mut config.auto_paste, "Auto-paste").changed() {
                action = MainAction::ConfigChanged;
            }
            if ui.checkbox(&mut config.auto_mute, "Auto-mute").changed() {
                action = MainAction::ConfigChanged;
            }
        });
    });

    action
}

pub fn draw_recording(
    ui: &mut egui::Ui,
    samples: &[f32],
    sample_rate: u32,
    elapsed: Duration,
    hotkey_display: &str,
) {
    ui.vertical_centered(|ui| {
        ui.add_space(20.0);

        // Status indicator
        status_indicator::draw_status(ui, AppStatus::Recording);

        ui.add_space(8.0);

        let secs = elapsed.as_secs_f32();
        ui.label(
            egui::RichText::new(format!("Recording... {secs:.1}s"))
                .size(20.0)
                .color(egui::Color32::from_rgb(233, 69, 96)),
        );

        ui.add_space(16.0);

        waveform::draw_waveform(ui, samples, sample_rate);

        ui.add_space(16.0);

        ui.label(
            egui::RichText::new(format!("Release {} to stop", hotkey_display))
                .size(14.0)
                .color(egui::Color32::from_gray(140)),
        );
    });
}
