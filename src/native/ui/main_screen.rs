use eframe::egui;
use std::time::Duration;

use super::{
    status_indicator::{self, AppStatus},
    theme, waveform,
};
use crate::native::config::{AppConfig, HotkeyConfig};
use crate::native::download::ModelVariant;
use crate::native::hotkey::{self, HotkeyCapture};
use crate::ALL_LANGUAGES;

pub enum MainAction {
    None,
    HotkeyChanged,
    ConfigChanged,
    OpenModelManager,
}

/// Inner height of the status panel, identical for Ready and Processing.
const PANEL_HEIGHT: f32 = 40.0;

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
    let processing = status == AppStatus::Processing;

    theme::top_bar(ui, |ui| {
        if theme::secondary_button(ui, "Models")
            .on_hover_text("Manage local models")
            .clicked()
        {
            action = MainAction::OpenModelManager;
        }
        status_indicator::draw_status(ui, status);
    });
    ui.add_space(10.0);

    // The app bar stays visible; only the content below it can scroll.
    egui::ScrollArea::vertical()
        .id_salt("dictation_body")
        .min_scrolled_height(0.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // The transcript takes the height left by the other blocks, measured on the previous pass.
            let fixed_id = ui.id().with("fixed_height");
            let fixed_height: f32 = ui.data(|data| data.get_temp(fixed_id)).unwrap_or(330.0);
            let transcript_height = (ui.available_height() - fixed_height).clamp(60.0, 300.0);
            theme::card()
                .inner_margin(egui::Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.allocate_ui_with_layout(
                        egui::vec2(ui.available_width(), PANEL_HEIGHT),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.set_min_size(egui::vec2(ui.available_width(), PANEL_HEIGHT));
                            if processing {
                                theme::spinner(ui, 22.0);
                            } else {
                                microphone(ui, 22.0, theme::ACCENT);
                            }
                            ui.add_space(4.0);
                            ui.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 1.0;
                                ui.label(theme::title(
                                    if processing { "Transcribing…" } else { "Push to talk" },
                                    15.0,
                                ));
                                ui.label(
                                    egui::RichText::new(if processing {
                                        format!("{} on the local GPU", variant.display_name())
                                    } else {
                                        "Hold the shortcut to record. Release to transcribe.".into()
                                    })
                                    .size(12.5)
                                    .color(theme::MUTED),
                                );
                            });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| keys(ui, &hotkey_display, !processing),
                            );
                        },
                    );
                });
            ui.add_space(8.0);

            theme::card()
                .inner_margin(egui::Margin::symmetric(14, 10))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.allocate_ui_with_layout(
                        egui::vec2(ui.available_width(), 24.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.label(theme::section("Last transcript"));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let copy_id = ui.id().with("copied_at");
                                    let now = ui.input(|i| i.time);
                                    let copied = ui
                                        .data(|data| data.get_temp::<f64>(copy_id))
                                        .is_some_and(|at| now - at < 2.0);
                                    if ui
                                        .add_enabled(
                                            !last_result.is_empty(),
                                            egui::Button::new(if copied { "Copied" } else { "Copy" })
                                                .min_size(egui::vec2(0.0, 24.0)),
                                        )
                                        .clicked()
                                    {
                                        ui.ctx().copy_text(last_result.to_owned());
                                        ui.data_mut(|data| data.insert_temp(copy_id, now));
                                        ui.ctx().request_repaint_after(Duration::from_secs(2));
                                    }
                                    if !last_result.is_empty() {
                                        ui.label(
                                            theme::meta(format!(
                                                "{:.2} s",
                                                last_inference_ms as f64 / 1000.0
                                            )),
                                        );
                                    }
                                },
                            );
                        },
                    );
                    ui.add_space(4.0);
                    ui.scope(|ui| {
                        // A quiet scrollbar on the card surface.
                        ui.visuals_mut().extreme_bg_color = theme::SURFACE;
                        ui.visuals_mut().widgets.inactive.fg_stroke.color = theme::BORDER_STRONG;
                        egui::ScrollArea::vertical()
                            .id_salt("transcript_scroll")
                            .min_scrolled_height(transcript_height)
                            .max_height(transcript_height)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                if last_result.is_empty() {
                                    ui.add_space(4.0);
                                    ui.label(egui::RichText::new("No transcript yet").color(theme::MUTED));
                                    ui.label(
                                        theme::meta(format!(
                                            "Hold {hotkey_display} and speak. The text appears here."
                                        ))
                                        .size(12.5),
                                    );
                                } else {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(last_result)
                                                .size(14.0)
                                                .line_height(Some(20.0))
                                                .color(theme::TEXT),
                                        )
                                        .wrap()
                                        .selectable(true),
                                    );
                                }
                            });
                    });
                    ui.add_space(6.0);
                    ui.label(theme::meta(variant.display_name()));
                });
            // Keep compact controls close; balance the footer gap on taller windows.
            ui.add_space((ui.ctx().screen_rect().height() - 580.0).clamp(4.0, 16.0));

            ui.separator();
            ui.add_space(4.0);
            ui.columns(2, |columns| {
                let ui = &mut columns[0];
                ui.label(theme::section("Shortcut"));
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), 28.0),
                    // Long shortcuts wrap the buttons onto a second line.
                    egui::Layout::left_to_right(egui::Align::Center).with_main_wrap(true),
                    |ui| {
                        ui.set_min_height(28.0);
                        if hotkey_capture.listening {
                            if let Some((mods, key)) = hotkey_capture.poll() {
                                config.hotkey.modifiers = mods;
                                config.hotkey.key = key;
                                action = MainAction::HotkeyChanged;
                            }
                            ui.label(
                                egui::RichText::new(hotkey_capture.current_display())
                                    .color(theme::ACCENT),
                            );
                            if theme::secondary_button(ui, "Cancel").clicked() {
                                hotkey_capture.listening = false;
                            }
                            ui.ctx().request_repaint();
                        } else {
                            keys(ui, &hotkey_display, true);
                            ui.add_space(4.0);
                            if theme::secondary_button(ui, "Change").clicked() {
                                hotkey_capture.start();
                            }
                            if config.hotkey != HotkeyConfig::default()
                                && theme::quiet_button(ui, "Reset")
                                    .on_hover_text(format!(
                                        "Restore the default shortcut ({})",
                                        hotkey::HotkeyState::display_string(&AppConfig::default())
                                    ))
                                    .clicked()
                            {
                                config.hotkey = HotkeyConfig::default();
                                action = MainAction::HotkeyChanged;
                            }
                        }
                    },
                );

                let ui = &mut columns[1];
                ui.label(theme::section("Spoken language"));
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), 28.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.set_min_height(28.0);
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
                            .height(220.0)
                            .icon(|ui, rect, visuals, is_open, _| {
                                let c = rect.center();
                                let y = if is_open { -2.5 } else { 2.5 };
                                ui.painter().add(egui::Shape::line(
                                    vec![
                                        c + egui::vec2(-3.5, -y / 2.0),
                                        c + egui::vec2(0.0, y / 2.0),
                                        c + egui::vec2(3.5, -y / 2.0),
                                    ],
                                    egui::Stroke::new(1.5_f32, visuals.fg_stroke.color),
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
            ui.add_space(2.0);
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
            let measured = ui.min_rect().height() - transcript_height;
            if (measured - fixed_height).abs() > 0.5 {
                ui.data_mut(|data| data.insert_temp(fixed_id, measured));
                ui.ctx().request_discard("fit transcript height");
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
    theme::top_bar(ui, |ui| {
        status_indicator::draw_status(ui, AppStatus::Recording);
    });
    ui.add_space(10.0);
    egui::ScrollArea::vertical()
        .id_salt("recording_body")
        .min_scrolled_height(0.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let card_height = ui.available_height();
            theme::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.set_min_height(card_height - 30.0);
                // Centre the content using the height measured on the previous pass.
                let height_id = ui.id().with("recording_content_height");
                let content_height: f32 = ui.data(|data| data.get_temp(height_id)).unwrap_or(0.0);
                ui.add_space(((card_height - 30.0 - content_height) / 2.0).max(0.0));
                let content = ui.vertical_centered(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{:02}:{:02}",
                            elapsed.as_secs() / 60,
                            elapsed.as_secs() % 60
                        ))
                        .font(egui::FontId::new(
                            28.0,
                            egui::FontFamily::Name("heading".into()),
                        ))
                        .color(theme::TEXT),
                    );
                    ui.add_space(10.0);
                    let wave_height = (card_height - 230.0).clamp(44.0, 72.0);
                    waveform::draw_waveform(ui, samples, sample_rate, wave_height);
                    ui.add_space(10.0);
                    ui.label(
                        egui::RichText::new(format!("Release {hotkey_display} to transcribe."))
                            .color(theme::MUTED),
                    );
                });
                let measured = content.response.rect.height();
                if (measured - content_height).abs() > 0.5 {
                    ui.data_mut(|data| data.insert_temp(height_id, measured));
                    ui.ctx().request_discard("center recording content");
                }
            });
        });
}

/// Shortcut as individual key caps, in reading order for either layout direction.
fn keys(ui: &mut egui::Ui, shortcut: &str, enabled: bool) {
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let mut parts: Vec<&str> = shortcut.split(" + ").collect();
        if ui.layout().prefer_right_to_left() {
            parts.reverse();
        }
        let color = if enabled { theme::TEXT } else { theme::DIM };
        for part in parts {
            let galley = ui.painter().layout_no_wrap(
                part.to_owned(),
                egui::FontId::proportional(12.0),
                color,
            );
            let size = egui::vec2((galley.size().x + 14.0).max(24.0), 22.0);
            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
            ui.painter().rect(
                rect,
                theme::CONTROL_RADIUS,
                theme::BG,
                egui::Stroke::new(1.0_f32, theme::BORDER_STRONG),
                egui::StrokeKind::Inside,
            );
            ui.painter()
                .galley(rect.center() - galley.size() / 2.0, galley, color);
        }
    });
}

fn microphone(ui: &mut egui::Ui, size: f32, color: egui::Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    let c = rect.center();
    let painter = ui.painter();
    let stroke = egui::Stroke::new(1.5_f32, color);
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
                    // Let egui settle the scroll areas, measured heights and font layout.
                    for _ in 0..3 {
                        let output = ctx.run(
                            egui::RawInput {
                                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into())),
                                ..Default::default()
                            },
                            |ctx| {
                                theme::footer(ctx);
                                egui::CentralPanel::default()
                                    .frame(egui::Frame::new().fill(theme::BG).inner_margin(20))
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
                            &["Whisper Burn", "Recording", "00:08", "Release Ctrl + Win to transcribe."]
                        } else {
                            &["Whisper Burn", "Models", "Ctrl", "Win", "Shortcut", "Spoken language",
                                "Change", "Auto-paste text", "Mute while recording", "Whisper Large V3"]
                        };
                        for label in labels {
                            let (clip, bounds) = label_bounds(label);
                            assert!(clip.expand(0.5).contains_rect(bounds),
                                "Clipped {label} at {size:?}, {scale}, {status:?}: {bounds:?} outside {clip:?}");
                        }
                        let mut cards = Vec::new();
                        for clipped in &output.shapes {
                            if let egui::Shape::Rect(rect) = &clipped.shape {
                                if rect.fill == theme::BG && rect.stroke.color == theme::BORDER_STRONG {
                                    assert!(rect.rect.height() <= 24.0, "Stretched key cap: {:?}", rect.rect);
                                }
                                if rect.fill == theme::SURFACE && rect.rect.width() > 400.0 {
                                    assert!(clipped.clip_rect.expand(0.5).contains_rect(rect.rect),
                                        "Card clipped at {size:?}, {scale}, {status:?}: {:?}", rect.rect);
                                    cards.push(rect.rect);
                                }
                                // A scrollbar can have an inverted, unpainted rect on its first frame.
                                let body_scrollbar = rect.rect.is_positive()
                                    && rect.fill == theme::TEXT
                                    && rect.rect.width() <= 4.1
                                    && rect.rect.left() > size[0] - 30.0
                                    && rect.rect.height() > 20.0;
                                assert!(!body_scrollbar, "Unneeded body scrollbar at {size:?}, {scale}, {status:?}: {:?}", rect.rect);
                            }
                        }
                        if status != AppStatus::Recording {
                            // The status panel keeps one height, so the transcript never jumps.
                            let top = cards.get(1).expect("transcript card").top();
                            if let Some(y) = transcript_top {
                                assert!((top - y).abs() < 1.0, "Status panel height shifts with state");
                            }
                            transcript_top = Some(top);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reset_appears_only_for_a_custom_shortcut_and_stays_in_its_column() {
        let render = |size: [f32; 2], scale: f32, config: &mut AppConfig| {
            let ctx = egui::Context::default();
            ctx.set_pixels_per_point(scale);
            theme::apply_dark_theme(&ctx);
            let mut capture = HotkeyCapture::new();
            let mut shapes = Vec::new();
            for _ in 0..3 {
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into())),
                    ..Default::default()
                };
                shapes = ctx
                    .run(input, |ctx| {
                        theme::footer(ctx);
                        egui::CentralPanel::default()
                            .frame(egui::Frame::new().inner_margin(20))
                            .show(ctx, |ui| {
                                draw_ready(
                                    ui,
                                    "",
                                    0,
                                    ModelVariant::LargeV3,
                                    config,
                                    AppStatus::Ready,
                                    &mut capture,
                                );
                            });
                    })
                    .shapes;
            }
            shapes
        };
        let text_bounds = |shapes: &[egui::epaint::ClippedShape], label: &str| {
            shapes.iter().find_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) if text.galley.text() == label => {
                    Some(text.visual_bounding_rect())
                }
                _ => None,
            })
        };
        for size in [[780.0, 640.0], [620.0, 540.0]] {
            for scale in [1.0, 1.25] {
                let shapes = render(size, scale, &mut AppConfig::default());
                assert!(
                    text_bounds(&shapes, "Reset").is_none(),
                    "Reset shown for the default shortcut"
                );

                let mut config = AppConfig::default();
                config.hotkey.modifiers = ["CONTROL", "ALT", "SHIFT", "SUPER"]
                    .map(String::from)
                    .into();
                config.hotkey.key = "PAGEDOWN".into();
                let shapes = render(size, scale, &mut config);
                let column = text_bounds(&shapes, "Spoken language")
                    .expect("language label")
                    .left();
                for label in ["Change", "Reset"] {
                    let bounds = text_bounds(&shapes, label)
                        .unwrap_or_else(|| panic!("Missing {label} at {size:?}, {scale}"));
                    assert!(
                        bounds.right() < column,
                        "{label} overlaps the language column at {size:?}, {scale}: {bounds:?}"
                    );
                }
            }
        }
    }
}
