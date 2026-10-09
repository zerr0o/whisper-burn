use eframe::egui;
use std::time::Duration;

use super::{
    status_indicator::{self, AppStatus},
    theme, waveform,
};
use crate::native::config::AppConfig;
use crate::native::download::ModelVariant;
use crate::native::hotkey::{self, HotkeyCapture};
use crate::ALL_LANGUAGES;

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
    let hotkey_display = hotkey::HotkeyState::display_string(config);

    ui.horizontal(|ui| {
        theme::brand(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if theme::secondary_button(ui, "Models")
                .on_hover_text("Manage local models")
                .clicked()
            {
                action = MainAction::OpenModelManager;
            }
            status_indicator::draw_status(ui, status);
        });
    });
    ui.add_space(4.0);

    // The header stays visible; only the content below it can scroll.
    egui::ScrollArea::vertical()
        .id_salt("dictation_body")
        .min_scrolled_height(0.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            theme::card()
                .fill(theme::ACCENT_BG)
                .inner_margin(12)
                .stroke(egui::Stroke::new(
                    1.0_f32,
                    egui::Color32::from_rgb(78, 62, 51),
                ))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    let left_width = (ui.available_width() - 170.0).max(220.0);
                    ui.horizontal(|ui| {
                        ui.allocate_ui_with_layout(
                            egui::vec2(left_width, 58.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                ui.set_min_size(egui::vec2(left_width, 58.0));
                                microphone(ui, 44.0, theme::ACCENT);
                                ui.add_space(6.0);
                                ui.vertical(|ui| {
                                    ui.label(
                                        theme::heading(if status == AppStatus::Processing {
                                            "Finding your words."
                                        } else {
                                            "Ready when you are."
                                        })
                                        .size(25.0),
                                    );
                                    ui.label(
                                        egui::RichText::new(if status == AppStatus::Processing {
                                            "Transcribing on your GPU."
                                        } else {
                                            "Hold your shortcut. Release to transcribe."
                                        })
                                        .size(13.0)
                                        .color(theme::MUTED),
                                    );
                                });
                            },
                        );
                        ui.allocate_ui_with_layout(
                            egui::vec2(160.0, 58.0),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.set_min_size(egui::vec2(160.0, 58.0));
                                ui.spacing_mut().item_spacing.y = 4.0;
                                if status == AppStatus::Processing {
                                    ui.add_space(8.0);
                                    theme::spinner(ui);
                                    ui.label(theme::eyebrow("PROCESSING"));
                                } else {
                                    keycap(ui, &hotkey_display);
                                    ui.label(theme::eyebrow("HOLD TO RECORD"));
                                }
                            },
                        );
                    });
                });
            ui.add_space(4.0);

            let transcript_height = (ui.available_height() - 244.0).clamp(60.0, 240.0);
            theme::card().inner_margin(12).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(theme::eyebrow("LATEST TRANSCRIPT"));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let copy_id = ui.id().with("copied_at");
                        let now = ui.input(|i| i.time);
                        let copied = ui
                            .data(|data| data.get_temp::<f64>(copy_id))
                            .is_some_and(|at| now - at < 2.0);
                        if ui
                            .add_enabled(
                                !last_result.is_empty(),
                                egui::Button::new(if copied { "Copied" } else { "Copy text" }),
                            )
                            .clicked()
                        {
                            ui.ctx().copy_text(last_result.to_owned());
                            ui.data_mut(|data| data.insert_temp(copy_id, now));
                            ui.ctx().request_repaint_after(Duration::from_secs(2));
                        }
                        if !last_result.is_empty() {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{:.2} s",
                                    last_inference_ms as f64 / 1000.0
                                ))
                                .size(12.0)
                                .color(theme::DIM),
                            );
                        }
                    });
                });
                ui.add_space(6.0);
                egui::ScrollArea::vertical()
                    .id_salt("transcript_scroll")
                    .min_scrolled_height(transcript_height)
                    .max_height(transcript_height)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if last_result.is_empty() {
                            ui.add_space(10.0);
                            ui.label(
                                theme::heading("A little less typing.")
                                    .size(18.0)
                                    .color(theme::MUTED),
                            );
                            ui.label(
                                egui::RichText::new("Your next transcript will appear here.")
                                    .size(13.0)
                                    .color(theme::DIM),
                            );
                        } else {
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(last_result)
                                        .size(16.0)
                                        .color(theme::TEXT),
                                )
                                .wrap()
                                .selectable(true),
                            );
                        }
                    });
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new(variant.display_name())
                        .size(12.0)
                        .color(theme::DIM),
                );
            });
            // Keep compact controls close; balance the footer gap on taller windows.
            ui.add_space((ui.ctx().screen_rect().height() - 580.0).clamp(4.0, 18.0));

            ui.separator();
            ui.add_space(2.0);
            ui.columns(2, |columns| {
                let ui = &mut columns[0];
                ui.label(theme::eyebrow("SHORTCUT"));
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), 34.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.set_min_height(34.0);
                        if hotkey_capture.listening {
                            if let Some((mods, key)) = hotkey_capture.poll() {
                                config.hotkey.modifiers = mods;
                                config.hotkey.key = key;
                                action = MainAction::HotkeyChanged;
                            }
                            ui.label(
                                egui::RichText::new(hotkey_capture.current_display())
                                    .size(13.0)
                                    .color(theme::ACCENT),
                            );
                            if theme::secondary_button(ui, "Cancel").clicked() {
                                hotkey_capture.listening = false;
                            }
                            ui.ctx().request_repaint();
                        } else {
                            keycap(ui, &hotkey_display);
                            if theme::secondary_button(ui, "Change").clicked() {
                                hotkey_capture.start();
                            }
                        }
                    },
                );

                let ui = &mut columns[1];
                ui.label(theme::eyebrow("SPOKEN LANGUAGE"));
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), 34.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.set_min_height(34.0);
                        let current_name = ALL_LANGUAGES
                            .iter()
                            .find(|l| l.code.unwrap_or("auto") == config.language)
                            .map(|l| match l.code {
                                Some(code) => format!("{} ({code})", l.name),
                                None => l.name.to_owned(),
                            })
                            .unwrap_or_else(|| "Auto".into());
                        let opening = !egui::ComboBox::is_open(
                            ui.ctx(),
                            ui.make_persistent_id("lang_selector"),
                        );
                        egui::ComboBox::from_id_salt("lang_selector")
                            .selected_text(current_name)
                            .width(ui.available_width())
                            .height(200.0)
                            .icon(|ui, rect, visuals, is_open, _| {
                                let c = rect.center();
                                let y = if is_open { -3.0 } else { 3.0 };
                                ui.painter().add(egui::Shape::line(
                                    vec![
                                        c + egui::vec2(-4.0, -y / 2.0),
                                        c + egui::vec2(0.0, y / 2.0),
                                        c + egui::vec2(4.0, -y / 2.0),
                                    ],
                                    visuals.fg_stroke,
                                ));
                            })
                            .show_ui(ui, |ui| {
                                for lang in &ALL_LANGUAGES {
                                    let code = lang.code.unwrap_or("auto");
                                    let label = if lang.code.is_some() {
                                        format!("{} ({})", lang.name, code)
                                    } else {
                                        lang.name.to_string()
                                    };
                                    let selected = config.language == code;
                                    let response = ui.selectable_label(selected, label);
                                    if opening && selected {
                                        response.scroll_to_me(Some(egui::Align::Center));
                                    }
                                    if response.clicked() {
                                        config.language = code.to_string();
                                        action = MainAction::ConfigChanged;
                                    }
                                }
                            });
                    },
                );
            });
            ui.columns(2, |columns| {
                if columns[0]
                    .checkbox(&mut config.auto_paste, "Auto-paste text")
                    .on_hover_text("Paste each transcript into the app you are using.")
                    .changed()
                {
                    action = MainAction::ConfigChanged;
                }
                if columns[1]
                    .checkbox(&mut config.auto_mute, "Mute while recording")
                    .on_hover_text(
                        "Temporarily mute system audio to keep it out of your recording.",
                    )
                    .changed()
                {
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
    ui.horizontal(|ui| {
        theme::brand(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            status_indicator::draw_status(ui, AppStatus::Recording);
        });
    });
    ui.add_space(12.0);
    egui::ScrollArea::vertical()
        .id_salt("recording_body")
        .min_scrolled_height(0.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            theme::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.vertical_centered(|ui| {
                    ui.add_space(if ui.ctx().screen_rect().height() < 600.0 {
                        4.0
                    } else {
                        16.0
                    });
                    microphone(ui, 48.0, theme::ACCENT);
                    ui.add_space(12.0);
                    ui.label(theme::eyebrow("LISTENING TO YOU"));
                    ui.label(
                        theme::heading(format!(
                            "{:02}:{:02}",
                            elapsed.as_secs() / 60,
                            elapsed.as_secs() % 60
                        ))
                        .size(44.0),
                    );
                    ui.add_space(12.0);
                    waveform::draw_waveform(ui, samples, sample_rate);
                    ui.add_space(16.0);
                    ui.label(theme::heading("Let your thoughts flow.").size(22.0));
                    ui.label(
                        egui::RichText::new(format!("Release {hotkey_display} to transcribe."))
                            .size(14.0)
                            .color(theme::MUTED),
                    );
                    ui.add_space(12.0);
                });
            });
        });
}

fn keycap(ui: &mut egui::Ui, shortcut: &str) {
    egui::Frame::new()
        .fill(theme::BG)
        .stroke(egui::Stroke::new(1.0_f32, theme::BORDER))
        .corner_radius(6)
        .inner_margin(egui::Margin::symmetric(10, 5))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(shortcut).size(13.0).color(theme::TEXT));
        });
}

fn microphone(ui: &mut egui::Ui, size: f32, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let c = rect.center();
    let painter = ui.painter();
    let stroke = egui::Stroke::new(1.8_f32, color);
    painter.rect_stroke(
        egui::Rect::from_center_size(
            c - egui::vec2(0.0, size * 0.1),
            egui::vec2(size * 0.24, size * 0.42),
        ),
        size * 0.12,
        stroke,
        egui::StrokeKind::Middle,
    );
    let y = c.y + size * 0.07;
    painter.add(egui::Shape::line(
        vec![
            egui::pos2(c.x - size * 0.22, y - size * 0.08),
            egui::pos2(c.x - size * 0.22, y),
            egui::pos2(c.x - size * 0.16, y + size * 0.16),
            egui::pos2(c.x, y + size * 0.21),
            egui::pos2(c.x + size * 0.16, y + size * 0.16),
            egui::pos2(c.x + size * 0.22, y),
            egui::pos2(c.x + size * 0.22, y - size * 0.08),
        ],
        stroke,
    ));
    painter.line_segment(
        [
            egui::pos2(c.x, y + size * 0.21),
            egui::pos2(c.x, y + size * 0.35),
        ],
        stroke,
    );
    painter.line_segment(
        [
            egui::pos2(c.x - size * 0.12, y + size * 0.35),
            egui::pos2(c.x + size * 0.12, y + size * 0.35),
        ],
        stroke,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_layout_keeps_controls_visible() {
        for size in [[780.0, 640.0], [640.0, 580.0], [620.0, 540.0]] {
            for scale in [1.0, 1.25] {
                let mut transcript_top: Option<f32> = None;
                for (status, text) in [
                    (AppStatus::Ready, ""),
                    (AppStatus::Done, "A transcript with enough words to wrap onto more than one line. \n\nThe transcript can scroll independently without hiding the settings."),
                    (AppStatus::Processing, "Previous transcript."),
                    (AppStatus::Recording, ""),
                ] {
                    let ctx = egui::Context::default();
                    ctx.set_pixels_per_point(scale);
                    theme::apply_dark_theme(&ctx);
                    let mut config = AppConfig::default();
                    config.language = "fr".into();
                    config.hotkey.modifiers = vec!["CONTROL".into(), "SUPER".into()];
                    config.hotkey.key.clear();
                    let mut capture = HotkeyCapture::new();
                    // Let egui settle the scroll areas and font layout at native DPI.
                    for _ in 0..3 {
                        let output = ctx.run(
                            egui::RawInput {
                                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into())),
                                ..Default::default()
                            },
                            |ctx| {
                                theme::footer(ctx);
                                egui::CentralPanel::default()
                                    .frame(egui::Frame::new().fill(theme::BG).inner_margin(24))
                                    .show(ctx, |ui| {
                                        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                                            if status == AppStatus::Recording {
                                                draw_recording(ui, &[], 16000, Duration::from_secs(8), "Ctrl + Win");
                                            } else {
                                                draw_ready(ui, text, 1270, ModelVariant::LargeV3,
                                                    &mut config, status, &mut capture);
                                            }
                                        });
                                    });
                            },
                        );
                        let label_bounds = |label: &str| {
                            output.shapes.iter().find_map(|clipped| {
                                if let egui::Shape::Text(text) = &clipped.shape {
                                    (text.galley.text() == label).then_some((clipped.clip_rect,
                                        text.visual_bounding_rect()))
                                } else { None }
                            }).unwrap_or_else(|| panic!("Missing {label} at {size:?}, {scale}, {status:?}"))
                        };
                        let labels: &[&str] = if status == AppStatus::Recording {
                            &["Whisper Burn", "Recording", "Let your thoughts flow.",
                                "Release Ctrl + Win to transcribe."]
                        } else {
                            &["Whisper Burn", "Models", "SHORTCUT", "SPOKEN LANGUAGE", "Change",
                                "Auto-paste text", "Mute while recording", "Whisper Large V3"]
                        };
                        for label in labels {
                            let (clip, bounds) = label_bounds(label);
                            assert!(clip.expand(0.5).contains_rect(bounds),
                                "Clipped {label} at {size:?}, {scale}, {status:?}: {bounds:?} outside {clip:?}");
                        }
                        if matches!(status, AppStatus::Ready | AppStatus::Done) {
                            let chord = label_bounds("Ctrl + Win").1;
                            let caption = label_bounds("HOLD TO RECORD").1;
                            assert!((chord.center().x - caption.center().x).abs() <= 2.0,
                                "Shortcut not centered: {:?} vs {:?}", chord.center(), caption.center());
                        }
                        for clipped in &output.shapes {
                            if let egui::Shape::Rect(rect) = &clipped.shape {
                                if rect.fill == theme::BG && rect.stroke.color == theme::BORDER {
                                    assert!(rect.rect.height() <= 40.0, "Stretched keycap: {:?}", rect.rect);
                                }
                                if rect.fill == theme::SURFACE && rect.rect.width() > 400.0 {
                                    assert!(clipped.clip_rect.expand(0.5).contains_rect(rect.rect),
                                        "Card clipped at {size:?}, {scale}, {status:?}: {:?}", rect.rect);
                                    if status != AppStatus::Recording {
                                        if let Some(y) = transcript_top {
                                            assert!((rect.rect.top() - y).abs() < 1.0, "Hero height shifts with state");
                                        }
                                        transcript_top = Some(rect.rect.top());
                                    }
                                }
                                // A scrollbar can have an inverted, unpainted rect on its first frame.
                                let body_scrollbar = rect.rect.is_positive()
                                    && rect.fill == theme::MUTED
                                    && rect.rect.width() <= 6.1
                                    && rect.rect.left() > size[0] - 34.0
                                    && rect.rect.height() > 20.0;
                                assert!(!body_scrollbar, "Unneeded body scrollbar at {size:?}, {scale}, {status:?}: {:?}", rect.rect);
                            }
                        }
                    }
                }
            }
        }
    }
}
