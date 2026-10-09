fn main() {
    println!("cargo:rerun-if-changed=assets/app-icon.ico");

    #[cfg(windows)]
    winres::WindowsResource::new()
        .set_icon("assets/app-icon.ico")
        .compile()
        .expect("failed to embed Windows app icon");
}
