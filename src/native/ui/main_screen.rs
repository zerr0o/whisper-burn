use eframe::egui;
use std::time::Duration;

use crate::native::download::ModelVariant;
use crate::native::ui::status_indicator::{self, AppStatus};
use crate::Language;
use crate::ALL_LANGUAGES;
use super::waveform;

pub enum MainAction {
    None,
    LanguageChanged(Language),
    OpenSettings,
    OpenModelManager,
}

pub fn draw_ready(
    ui: &mut egui::Ui,
    last_result: &str,
    last_inference_ms: u128,
    variant: ModelVariant,
    current_lang: Language,
    hotkey_display: &str,
    status: AppStatus,
) -> MainAction {
    let mut action = MainAction::None;

    ui.vertical_centered(|ui| {
        // Top bar with settings + model manager
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("\u{2699}").on_hover_text("Settings").clicked() {
                    action = MainAction::OpenSettings;
                }
                if ui.button("\u{1F4E6}").on_hover_text("Model Manager").clicked() {
                    action = MainAction::OpenModelManager;
                }
            });
        });

        ui.add_space(16.0);
        ui.heading("Whisper Burn");
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(format!("Model: {}", variant.display_name()))
                .size(13.0)
                .color(egui::Color32::from_gray(140)),
        );
        ui.add_space(8.0);

        // Status indicator
        status_indicator::draw_status(ui, status);

        ui.add_space(12.0);

        // Language selector
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Language:")
                    .size(14.0)
                    .color(egui::Color32::from_gray(160)),
            );
            let mut selected_code = current_lang.code.unwrap_or("auto").to_string();
            let display = current_lang.display_name();
            egui::ComboBox::from_id_salt("lang_selector")
                .selected_text(display)
                .height(300.0)
                .show_ui(ui, |ui| {
                    for lang in &ALL_LANGUAGES {
                        let code = lang.code.unwrap_or("auto");
                        let label = if lang.code.is_some() {
                            format!("{} ({})", lang.name, code)
                        } else {
                            lang.name.to_string()
                        };
                        if ui.selectable_label(selected_code == code, &label).clicked() {
                            selected_code = code.to_string();
                        }
                    }
                });
            let new_lang = Language::from_code(&selected_code);
            if *new_lang != current_lang {
                action = MainAction::LanguageChanged(*new_lang);
            }
        });

        ui.add_space(20.0);

        ui.label(
            egui::RichText::new(format!("Hold {} to record", hotkey_display))
                .size(22.0)
                .color(egui::Color32::from_rgb(180, 180, 200)),
        );

        ui.add_space(30.0);

        // Transcription zone
        if !last_result.is_empty() {
            ui.group(|ui| {
                ui.set_min_width(500.0);

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Last transcription:")
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
                ui.centered_and_justified(|ui| {
                    ui.label(
                        egui::RichText::new("Transcription will appear here")
                            .size(14.0)
                            .color(egui::Color32::from_gray(90)),
                    );
                });
            });
        }
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
