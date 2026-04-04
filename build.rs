use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=assets/app-icon.png");

    #[cfg(windows)]
    embed_windows_icon().expect("failed to embed Windows app icon");
}

#[cfg(windows)]
fn embed_windows_icon() -> Result<(), Box<dyn std::error::Error>> {
    let png_path = Path::new("assets/app-icon.png");
    let ico_path = generate_ico(png_path)?;

    let mut res = winres::WindowsResource::new();
    res.set_icon(ico_path.to_string_lossy().as_ref());
    res.compile()?;

    Ok(())
}

#[cfg(windows)]
fn generate_ico(png_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use ico::{IconDir, IconDirEntry, IconImage, ResourceType};
    use image::imageops::FilterType;

    let source = image::open(png_path)?.into_rgba8();
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let ico_path = out_dir.join("app-icon.ico");

    let mut icon_dir = IconDir::new(ResourceType::Icon);
    for size in [16, 24, 32, 48, 64, 128, 256] {
        let resized = image::imageops::resize(&source, size, size, FilterType::Lanczos3);
        let icon = IconImage::from_rgba_data(size, size, resized.into_raw());
        icon_dir.add_entry(IconDirEntry::encode(&icon)?);
    }

    let mut file = File::create(&ico_path)?;
    icon_dir.write(&mut file)?;

    Ok(ico_path)
}
