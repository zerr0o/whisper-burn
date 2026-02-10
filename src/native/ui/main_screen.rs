use eframe::egui;
use std::time::Duration;

use crate::native::download::ModelVariant;
use crate::Language;
use super::waveform;

/// Draw the ready screen. Returns `Some(lang)` if the user changed the language.
pub fn draw_ready(
    ui: &mut egui::Ui,
    last_result: &str,
    last_inference_ms: u128,
    variant: ModelVariant,
    current_lang: Language,
) -> Option<Language> {
    let mut new_lang = None;

    ui.vertical_centered(|ui| {
        ui.add_space(40.0);
        ui.heading("Whisper Burn");
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(format!("Model: {}", variant.display_name()))
                .size(13.0)
                .color(egui::Color32::from_gray(140)),
        );
        ui.add_space(12.0);

        // Language selector
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Language:")
                    .size(14.0)
                    .color(egui::Color32::from_gray(160)),
            );
            let mut selected = current_lang;
            egui::ComboBox::from_id_salt("lang_selector")
                .selected_text(selected.display_name())
                .show_ui(ui, |ui| {
                    for lang in Language::ALL {
                        ui.selectable_value(&mut selected, lang, lang.display_name());
                    }
                });
            if selected != current_lang {
                new_lang = Some(selected);
            }
        });

        ui.add_space(20.0);

        ui.label(
            egui::RichText::new("Hold SPACE to record")
                .size(22.0)
                .color(egui::Color32::from_rgb(180, 180, 200)),
        );

        ui.add_space(40.0);

        if !last_result.is_empty() {
            ui.group(|ui| {
                ui.set_min_width(500.0);
                ui.label(
                    egui::RichText::new("Last transcription:")
                        .size(14.0)
                        .color(egui::Color32::from_gray(140)),
                );
                ui.add_space(4.0);
                ui.label(egui::RichText::new(last_result).size(16.0));
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(format!("Inference: {last_inference_ms} ms"))
                        .size(12.0)
                        .color(egui::Color32::from_gray(120)),
                );
            });
        }
    });

    new_lang
}

pub fn draw_recording(
    ui: &mut egui::Ui,
    samples: &[f32],
    sample_rate: u32,
    elapsed: Duration,
) {
    ui.vertical_centered(|ui| {
        ui.add_space(20.0);

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
            egui::RichText::new("Release SPACE to stop")
                .size(14.0)
                .color(egui::Color32::from_gray(140)),
        );
    });
}
