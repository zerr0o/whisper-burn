use tray_icon::{
    TrayIcon, TrayIconBuilder,
    menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem},
    Icon,
};

pub enum TrayAction {
    ToggleWindow,
    Quit,
    None,
}

pub struct TrayState {
    _tray: TrayIcon,
    toggle_id: tray_icon::menu::MenuId,
    quit_id: tray_icon::menu::MenuId,
}

impl TrayState {
    pub fn new() -> Self {
        let toggle_item = MenuItem::new("Show/Hide", true, None);
        let quit_item = MenuItem::new("Quit", true, None);
        let toggle_id = toggle_item.id().clone();
        let quit_id = quit_item.id().clone();

        let menu = Menu::new();
        let _ = menu.append(&toggle_item);
        let _ = menu.append(&PredefinedMenuItem::separator());
        let _ = menu.append(&quit_item);

        let icon = create_default_icon();
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("Whisper Burn")
            .with_icon(icon)
            .build()
            .expect("Failed to build tray icon");

        Self {
            _tray: tray,
            toggle_id,
            quit_id,
        }
    }

    pub fn poll(&self) -> TrayAction {
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if event.id == self.toggle_id {
                return TrayAction::ToggleWindow;
            }
            if event.id == self.quit_id {
                return TrayAction::Quit;
            }
        }
        TrayAction::None
    }
}

fn create_default_icon() -> Icon {
    // 16x16 RGBA icon - simple microphone shape in red/dark
    let size = 16u32;
    let mut rgba = vec![0u8; (size * size * 4) as usize];

    for y in 0..size {
        for x in 0..size {
            let idx = ((y * size + x) * 4) as usize;
            let cx = x as f32 - 7.5;
            let cy = y as f32 - 7.5;
            let dist = (cx * cx + cy * cy).sqrt();

            if dist < 6.0 {
                // Red circle
                rgba[idx] = 233;     // R
                rgba[idx + 1] = 69;  // G
                rgba[idx + 2] = 96;  // B
                rgba[idx + 3] = 255; // A
            } else if dist < 7.0 {
                // Anti-aliased edge
                let alpha = ((7.0 - dist) * 255.0) as u8;
                rgba[idx] = 233;
                rgba[idx + 1] = 69;
                rgba[idx + 2] = 96;
                rgba[idx + 3] = alpha;
            }
        }
    }

    Icon::from_rgba(rgba, size, size).expect("Failed to create tray icon")
}

#[cfg(windows)]
#[allow(dead_code)]
pub fn set_window_visible(hwnd: isize, visible: bool) {
    use windows::Win32::UI::WindowsAndMessaging::{ShowWindow, SW_SHOW, SW_HIDE, SetForegroundWindow};
    use windows::Win32::Foundation::HWND;

    unsafe {
        let hwnd = HWND(hwnd as *mut _);
        if visible {
            let _ = ShowWindow(hwnd, SW_SHOW);
            let _ = SetForegroundWindow(hwnd);
        } else {
            let _ = ShowWindow(hwnd, SW_HIDE);
        }
    }
}
