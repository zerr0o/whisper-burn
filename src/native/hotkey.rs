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

/// Modifier names in display order, with their left and right virtual key codes.
const MODIFIERS: [(&str, [i32; 2]); 4] = [
    ("CONTROL", [0xA2, 0xA3]),
    ("ALT", [0xA4, 0xA5]),
    ("SHIFT", [0xA0, 0xA1]),
    ("SUPER", [0x5B, 0x5C]),
];

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

/// Records a new shortcut. Every key pressed counts until all keys are released.
pub struct HotkeyCapture {
    pub listening: bool,
    /// Modifiers pressed during this capture, in `MODIFIERS` order.
    modifiers: [bool; 4],
    key: Option<&'static str>,
    pressed: bool,
    /// Keys still held when the capture started are ignored until released.
    waiting_for_release: bool,
}

impl HotkeyCapture {
    pub fn new() -> Self {
        Self {
            listening: false,
            modifiers: [false; 4],
            key: None,
            pressed: false,
            waiting_for_release: false,
        }
    }

    /// Start listening. Waits for all keys to be released first (clean start).
    pub fn start(&mut self) {
        *self = Self {
            listening: true,
            waiting_for_release: true,
            ..Self::new()
        };
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

            // SAFETY: GetAsyncKeyState only reads the global keyboard state.
            let down = |vk: i32| unsafe { GetAsyncKeyState(vk) < 0 };
            let modifiers = MODIFIERS.map(|(_, vks)| vks.into_iter().any(down));
            let key = TRIGGER_KEYS
                .iter()
                .find(|&&(vk, _)| down(vk))
                .map(|&(_, name)| name);
            self.update(modifiers, key)
        }

        #[cfg(not(windows))]
        {
            None
        }
    }

    /// Advances the capture with the keys held right now.
    fn update(
        &mut self,
        modifiers: [bool; 4],
        key: Option<&'static str>,
    ) -> Option<(Vec<String>, String)> {
        let any_held = modifiers.contains(&true) || key.is_some();
        if self.waiting_for_release {
            self.waiting_for_release = any_held;
            return None;
        }
        if any_held {
            self.pressed = true;
            for (seen, held) in self.modifiers.iter_mut().zip(modifiers) {
                *seen |= held;
            }
            self.key = key.or(self.key);
            return None;
        }
        if !self.pressed {
            return None;
        }
        // Every key is released: the combination is complete.
        let combo: (Vec<String>, String) = (
            self.modifier_names().map(String::from).collect(),
            self.key.unwrap_or_default().to_string(),
        );
        *self = Self::new();
        Some(combo)
    }

    fn modifier_names(&self) -> impl Iterator<Item = &'static str> {
        MODIFIERS
            .iter()
            .zip(self.modifiers)
            .filter(|&(_, seen)| seen)
            .map(|((name, _), _)| *name)
    }

    /// Display what's being pressed during capture
    pub fn current_display(&self) -> String {
        let parts: Vec<String> = self
            .modifier_names()
            .map(modifier_display)
            .chain(self.key.map(key_display))
            .collect();
        if parts.is_empty() {
            "Press the new shortcut…".to_string()
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

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: [bool; 4] = [false; 4];
    const CTRL: [bool; 4] = [true, false, false, false];
    const WIN: [bool; 4] = [false, false, false, true];
    const CTRL_WIN: [bool; 4] = [true, false, false, true];

    fn started() -> HotkeyCapture {
        let mut capture = HotkeyCapture::new();
        capture.start();
        capture
    }

    #[test]
    fn modifier_only_shortcut_keeps_both_keys_in_any_order() {
        for (first, last) in [(CTRL, WIN), (WIN, CTRL)] {
            let mut capture = started();
            for held in [NONE, first, CTRL_WIN, last] {
                assert_eq!(capture.update(held, None), None);
            }
            assert_eq!(capture.current_display(), "Ctrl + Win");
            assert_eq!(
                capture.update(NONE, None),
                Some((vec!["CONTROL".into(), "SUPER".into()], String::new()))
            );
            assert!(!capture.listening);
        }
    }

    #[test]
    fn keys_held_when_capture_starts_are_ignored() {
        let mut capture = started();
        assert_eq!(capture.update(CTRL, None), None);
        assert_eq!(capture.update(NONE, None), None);
        assert_eq!(capture.update(NONE, Some("F9")), None);
        assert_eq!(capture.update(NONE, None), Some((Vec::new(), "F9".into())));
    }

    #[test]
    fn default_shortcut_is_ctrl_win() {
        assert_eq!(
            HotkeyState::display_string(&AppConfig::default()),
            "Ctrl + Win"
        );
    }
}
