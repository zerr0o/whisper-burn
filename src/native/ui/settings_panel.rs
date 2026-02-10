use eframe::egui;

use crate::native::config::AppConfig;
use crate::native::hotkey;
use crate::ALL_LANGUAGES;

pub enum SettingsAction {
    None,
    HotkeyChanged,
    Close,
}

pub struct SettingsState {
    pub open: bool,
    pub listening_for_hotkey: bool,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self {
            open: false,
            listening_for_hotkey: false,
        }
    }
}

pub fn draw(
    ctx: &egui::Context,
    config: &mut AppConfig,
    state: &mut SettingsState,
) -> SettingsAction {
    if !state.open {
        return SettingsAction::None;
    }

    let mut action = SettingsAction::None;

    egui::Window::new("Settings")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .fixed_size([400.0, 350.0])
        .show(ctx, |ui| {
            ui.add_space(8.0);

            // Hotkey
            ui.horizontal(|ui| {
                ui.label("Push-to-Talk Hotkey:");
                let display = hotkey::HotkeyState::display_string(config);
                if state.listening_for_hotkey {
                    ui.colored_label(
                        egui::Color32::from_rgb(233, 69, 96),
                        "Press a key...",
                    );

                    // Listen for F1-F12 keys
                    let keys = [
                        (egui::Key::F1, "F1"), (egui::Key::F2, "F2"),
                        (egui::Key::F3, "F3"), (egui::Key::F4, "F4"),
                        (egui::Key::F5, "F5"), (egui::Key::F6, "F6"),
                        (egui::Key::F7, "F7"), (egui::Key::F8, "F8"),
                        (egui::Key::F9, "F9"), (egui::Key::F10, "F10"),
                        (egui::Key::F11, "F11"), (egui::Key::F12, "F12"),
                        (egui::Key::Space, "SPACE"),
                        (egui::Key::Insert, "INSERT"),
                        (egui::Key::Delete, "DELETE"),
                        (egui::Key::Home, "HOME"),
                        (egui::Key::End, "END"),
                        (egui::Key::PageUp, "PAGEUP"),
                        (egui::Key::PageDown, "PAGEDOWN"),
                    ];

                    for (key, name) in keys {
                        if ctx.input(|i| i.key_pressed(key)) {
                            let mut modifiers = Vec::new();
                            ctx.input(|i| {
                                if i.modifiers.ctrl {
                                    modifiers.push("CONTROL".into());
                                }
                                if i.modifiers.alt {
                                    modifiers.push("ALT".into());
                                }
                                if i.modifiers.shift {
                                    modifiers.push("SHIFT".into());
                                }
                            });
                            config.hotkey.key = name.into();
                            config.hotkey.modifiers = modifiers;
                            state.listening_for_hotkey = false;
                            action = SettingsAction::HotkeyChanged;
                            break;
                        }
                    }

                    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                        state.listening_for_hotkey = false;
                    }
                } else {
                    ui.label(&display);
                    if ui.button("Change").clicked() {
                        state.listening_for_hotkey = true;
                    }
                }
            });

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // Language selector
            ui.horizontal(|ui| {
                ui.label("Language:");
                let current_name = ALL_LANGUAGES
                    .iter()
                    .find(|l| {
                        l.code.unwrap_or("auto") == config.language
                    })
                    .map(|l| l.name)
                    .unwrap_or("Auto");

                egui::ComboBox::from_id_salt("settings_lang")
                    .selected_text(current_name)
                    .show_ui(ui, |ui| {
                        for lang in &ALL_LANGUAGES {
                            let code = lang.code.unwrap_or("auto");
                            let label = format!("{} ({})", lang.name, code);
                            if ui.selectable_label(config.language == code, &label).clicked() {
                                config.language = code.to_string();
                            }
                        }
                    });
            });

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // Toggles
            ui.checkbox(&mut config.auto_paste, "Auto-paste transcription (Ctrl+V)");
            ui.add_space(4.0);
            ui.checkbox(&mut config.auto_mute, "Mute system audio while recording");
            ui.add_space(4.0);
            ui.checkbox(&mut config.start_minimized, "Start minimized to tray");

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                if ui.button("Close").clicked() {
                    state.open = false;
                    action = SettingsAction::Close;
                }
            });
        });

    action
}
