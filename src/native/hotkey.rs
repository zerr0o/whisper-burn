use super::config::AppConfig;

pub enum HotkeyEvent {
    Pressed,
    Released,
    None,
}

// --- Hotkey detection via GetAsyncKeyState polling ---

pub struct HotkeyState {
    was_active: bool,
}

impl HotkeyState {
    pub fn new() -> Self {
        Self { was_active: false }
    }

    /// Poll whether the configured hotkey combo is currently held.
    /// Detects press/release transitions.
    pub fn poll(&mut self, config: &AppConfig) -> HotkeyEvent {
        let active = is_combo_pressed(config);
        if active && !self.was_active {
            self.was_active = true;
            return HotkeyEvent::Pressed;
        }
        if !active && self.was_active {
            self.was_active = false;
            return HotkeyEvent::Released;
        }
        HotkeyEvent::None
    }

    pub fn display_string(config: &AppConfig) -> String {
        let mut parts = Vec::new();
        for m in &config.hotkey.modifiers {
            parts.push(modifier_display(m));
        }
        if !config.hotkey.key.is_empty() {
            parts.push(key_display(&config.hotkey.key));
        }
        if parts.is_empty() {
            return "(no hotkey set)".to_string();
        }
        parts.join(" + ")
    }
}

#[cfg(windows)]
fn is_combo_pressed(config: &AppConfig) -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

    let has_mods = !config.hotkey.modifiers.is_empty();
    let has_key = !config.hotkey.key.is_empty();

    if !has_mods && !has_key {
        return false;
    }

    unsafe {
        // All configured modifiers must be held
        for m in &config.hotkey.modifiers {
            if !is_modifier_down(m) {
                return false;
            }
        }

        // Trigger key must be held (if configured)
        if has_key {
            let vk = key_to_vk(&config.hotkey.key);
            if vk == 0 || GetAsyncKeyState(vk) >= 0 {
                return false;
            }
        }

        true
    }
}

#[cfg(not(windows))]
fn is_combo_pressed(_config: &AppConfig) -> bool {
    false
}

// --- Hotkey capture ---

/// Virtual key codes for trigger keys (non-modifier)
#[cfg(windows)]
const TRIGGER_KEYS: &[(i32, &str)] = &[
    (0x70, "F1"),  (0x71, "F2"),  (0x72, "F3"),  (0x73, "F4"),
    (0x74, "F5"),  (0x75, "F6"),  (0x76, "F7"),  (0x77, "F8"),
    (0x78, "F9"),  (0x79, "F10"), (0x7A, "F11"), (0x7B, "F12"),
    (0x20, "SPACE"), (0x2D, "INSERT"), (0x2E, "DELETE"),
    (0x24, "HOME"),  (0x23, "END"),
    (0x21, "PAGEUP"), (0x22, "PAGEDOWN"),
];

pub struct HotkeyCapture {
    pub listening: bool,
    accumulated_modifiers: Vec<String>,
    accumulated_key: Option<String>,
    any_pressed_this_session: bool,
    waiting_for_clean_start: bool,
}

impl HotkeyCapture {
    pub fn new() -> Self {
        Self {
            listening: false,
            accumulated_modifiers: Vec::new(),
            accumulated_key: None,
            any_pressed_this_session: false,
            waiting_for_clean_start: false,
        }
    }

    /// Start listening. Waits for all keys to be released first (clean start).
    pub fn start(&mut self) {
        self.listening = true;
        self.accumulated_modifiers.clear();
        self.accumulated_key = None;
        self.any_pressed_this_session = false;
        self.waiting_for_clean_start = true;
    }

    /// Poll key states during capture. Returns Some((modifiers, key)) when
    /// a combo was pressed and then all keys released.
    /// `key` may be empty for modifier-only combos.
    pub fn poll(&mut self) -> Option<(Vec<String>, String)> {
        if !self.listening {
            return None;
        }

        #[cfg(windows)]
        {
            use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

            let mut any_currently_pressed = false;

            unsafe {
                // Check and accumulate modifiers
                let mod_checks: &[(&str, &[i32])] = &[
                    ("CONTROL", &[0xA2, 0xA3]),
                    ("ALT", &[0xA4, 0xA5]),
                    ("SHIFT", &[0xA0, 0xA1]),
                    ("SUPER", &[0x5B, 0x5C]),
                ];

                for &(name, vks) in mod_checks {
                    if vks.iter().any(|&vk| GetAsyncKeyState(vk) < 0) {
                        any_currently_pressed = true;
                        if !self.accumulated_modifiers.iter().any(|m| m == name) {
                            self.accumulated_modifiers.push(name.to_string());
                        }
                    }
                }

                // Check and accumulate trigger key (keep last one seen)
                for &(vk, name) in TRIGGER_KEYS {
                    if GetAsyncKeyState(vk) < 0 {
                        any_currently_pressed = true;
                        self.accumulated_key = Some(name.to_string());
                        break;
                    }
                }
            }

            // Wait for clean start (all keys released after clicking "Change")
            if self.waiting_for_clean_start {
                if !any_currently_pressed {
                    self.waiting_for_clean_start = false;
                }
                return None;
            }

            if any_currently_pressed {
                self.any_pressed_this_session = true;
            } else if self.any_pressed_this_session {
                // All keys released — capture complete
                self.any_pressed_this_session = false;
                self.listening = false;

                let mods = std::mem::take(&mut self.accumulated_modifiers);
                let key = self.accumulated_key.take().unwrap_or_default();

                if !mods.is_empty() || !key.is_empty() {
                    return Some((mods, key));
                }
            }

            None
        }

        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Display what's being pressed during capture
    pub fn current_display(&self) -> String {
        let mut parts: Vec<String> = self
            .accumulated_modifiers
            .iter()
            .map(|m| modifier_display(m))
            .collect();
        if let Some(ref k) = self.accumulated_key {
            parts.push(key_display(k));
        }
        if parts.is_empty() {
            "Press your hotkey...".to_string()
        } else {
            parts.join(" + ")
        }
    }
}

// --- Display helpers ---

fn modifier_display(m: &str) -> String {
    match m.to_uppercase().as_str() {
        "CONTROL" | "CTRL" => "Ctrl".into(),
        "ALT" => "Alt".into(),
        "SHIFT" => "Shift".into(),
        "SUPER" | "WIN" => "Win".into(),
        other => other.into(),
    }
}

fn key_display(k: &str) -> String {
    match k.to_uppercase().as_str() {
        "SPACE" => "Space".into(),
        "INSERT" => "Insert".into(),
        "DELETE" => "Delete".into(),
        "HOME" => "Home".into(),
        "END" => "End".into(),
        "PAGEUP" => "Page Up".into(),
        "PAGEDOWN" => "Page Down".into(),
        other => other.into(),
    }
}

// --- Win32 key helpers ---

#[cfg(windows)]
unsafe fn is_modifier_down(m: &str) -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
    match m.to_uppercase().as_str() {
        "CONTROL" | "CTRL" => GetAsyncKeyState(0xA2) < 0 || GetAsyncKeyState(0xA3) < 0,
        "ALT" => GetAsyncKeyState(0xA4) < 0 || GetAsyncKeyState(0xA5) < 0,
        "SHIFT" => GetAsyncKeyState(0xA0) < 0 || GetAsyncKeyState(0xA1) < 0,
        "SUPER" | "WIN" => GetAsyncKeyState(0x5B) < 0 || GetAsyncKeyState(0x5C) < 0,
        _ => false,
    }
}

#[cfg(windows)]
fn key_to_vk(key: &str) -> i32 {
    match key.to_uppercase().as_str() {
        "F1" => 0x70,  "F2" => 0x71,  "F3" => 0x72,  "F4" => 0x73,
        "F5" => 0x74,  "F6" => 0x75,  "F7" => 0x76,  "F8" => 0x77,
        "F9" => 0x78,  "F10" => 0x79, "F11" => 0x7A, "F12" => 0x7B,
        "SPACE" => 0x20,
        "INSERT" => 0x2D, "DELETE" => 0x2E,
        "HOME" => 0x24,   "END" => 0x23,
        "PAGEUP" => 0x21,  "PAGEDOWN" => 0x22,
        _ => 0,
    }
}
