use anyhow::Result;

pub fn auto_paste(text: &str) -> Result<()> {
    use arboard::Clipboard;
    use enigo::{Direction, Enigo, Key, Keyboard, Settings};

    let mut clipboard = Clipboard::new()
        .map_err(|e| anyhow::anyhow!("Clipboard init failed: {e}"))?;

    // Save current clipboard content
    let old_text = clipboard.get_text().ok();

    // Set our text
    clipboard
        .set_text(text)
        .map_err(|e| anyhow::anyhow!("Clipboard set failed: {e}"))?;

    // Wait for clipboard to stabilize
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Simulate Ctrl+V
    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| anyhow::anyhow!("Enigo init failed: {e}"))?;
    enigo.key(Key::Control, Direction::Press)
        .map_err(|e| anyhow::anyhow!("Key press failed: {e}"))?;
    std::thread::sleep(std::time::Duration::from_millis(10));
    enigo.key(Key::Unicode('v'), Direction::Click)
        .map_err(|e| anyhow::anyhow!("Key click failed: {e}"))?;
    std::thread::sleep(std::time::Duration::from_millis(10));
    enigo.key(Key::Control, Direction::Release)
        .map_err(|e| anyhow::anyhow!("Key release failed: {e}"))?;

    // Wait, then restore
    std::thread::sleep(std::time::Duration::from_millis(100));
    if let Some(old) = old_text {
        let _ = clipboard.set_text(old);
    }

    Ok(())
}
