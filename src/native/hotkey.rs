use global_hotkey::{
    GlobalHotKeyEvent, GlobalHotKeyManager,
    hotkey::{Code, HotKey, Modifiers},
};

use super::config::AppConfig;

pub enum HotkeyEvent {
    Pressed,
    Released,
    None,
}

pub struct HotkeyState {
    manager: GlobalHotKeyManager,
    current: Option<HotKey>,
    was_pressed: bool,
}

impl HotkeyState {
    pub fn new(config: &AppConfig) -> Self {
        let manager = GlobalHotKeyManager::new().expect("Failed to create hotkey manager");
        let mut state = Self {
            manager,
            current: None,
            was_pressed: false,
        };
        state.register_from_config(config);
        state
    }

    pub fn poll(&mut self) -> HotkeyEvent {
        let mut event = HotkeyEvent::None;
        while let Ok(ev) = GlobalHotKeyEvent::receiver().try_recv() {
            if let Some(ref hk) = self.current {
                if ev.id() == hk.id() {
                    match ev.state() {
                        global_hotkey::HotKeyState::Pressed => {
                            if !self.was_pressed {
                                self.was_pressed = true;
                                event = HotkeyEvent::Pressed;
                            }
                        }
                        global_hotkey::HotKeyState::Released => {
                            if self.was_pressed {
                                self.was_pressed = false;
                                event = HotkeyEvent::Released;
                            }
                        }
                    }
                }
            }
        }
        event
    }

    pub fn update_hotkey(&mut self, config: &AppConfig) {
        self.unregister_current();
        self.register_from_config(config);
    }

    fn register_from_config(&mut self, config: &AppConfig) {
        let code = code_from_string(&config.hotkey.key);
        let mods = modifiers_from_strings(&config.hotkey.modifiers);
        let hk = HotKey::new(Some(mods), code);
        if self.manager.register(hk).is_ok() {
            self.current = Some(hk);
        }
    }

    fn unregister_current(&mut self) {
        if let Some(hk) = self.current.take() {
            let _ = self.manager.unregister(hk);
        }
        self.was_pressed = false;
    }

    pub fn display_string(config: &AppConfig) -> String {
        let mut parts = Vec::new();
        for m in &config.hotkey.modifiers {
            parts.push(m.clone());
        }
        parts.push(config.hotkey.key.clone());
        parts.join(" + ")
    }
}

pub fn code_from_string(s: &str) -> Code {
    match s.to_uppercase().as_str() {
        "F1" => Code::F1,
        "F2" => Code::F2,
        "F3" => Code::F3,
        "F4" => Code::F4,
        "F5" => Code::F5,
        "F6" => Code::F6,
        "F7" => Code::F7,
        "F8" => Code::F8,
        "F9" => Code::F9,
        "F10" => Code::F10,
        "F11" => Code::F11,
        "F12" => Code::F12,
        "SPACE" => Code::Space,
        "INSERT" => Code::Insert,
        "DELETE" => Code::Delete,
        "HOME" => Code::Home,
        "END" => Code::End,
        "PAGEUP" => Code::PageUp,
        "PAGEDOWN" => Code::PageDown,
        "PAUSE" => Code::Pause,
        "SCROLLLOCK" => Code::ScrollLock,
        "NUMLOCK" => Code::NumLock,
        "CAPSLOCK" => Code::CapsLock,
        _ => Code::F2,
    }
}

pub fn modifiers_from_strings(mods: &[String]) -> Modifiers {
    let mut result = Modifiers::empty();
    for m in mods {
        match m.to_uppercase().as_str() {
            "CONTROL" | "CTRL" => result |= Modifiers::CONTROL,
            "ALT" => result |= Modifiers::ALT,
            "SHIFT" => result |= Modifiers::SHIFT,
            "SUPER" | "WIN" => result |= Modifiers::SUPER,
            _ => {}
        }
    }
    result
}

pub fn key_name_from_code(code: Code) -> &'static str {
    match code {
        Code::F1 => "F1",
        Code::F2 => "F2",
        Code::F3 => "F3",
        Code::F4 => "F4",
        Code::F5 => "F5",
        Code::F6 => "F6",
        Code::F7 => "F7",
        Code::F8 => "F8",
        Code::F9 => "F9",
        Code::F10 => "F10",
        Code::F11 => "F11",
        Code::F12 => "F12",
        Code::Space => "SPACE",
        Code::Insert => "INSERT",
        Code::Delete => "DELETE",
        Code::Home => "HOME",
        Code::End => "END",
        Code::PageUp => "PAGEUP",
        Code::PageDown => "PAGEDOWN",
        Code::Pause => "PAUSE",
        Code::ScrollLock => "SCROLLLOCK",
        Code::NumLock => "NUMLOCK",
        Code::CapsLock => "CAPSLOCK",
        _ => "?",
    }
}
