//! Native presentation fixtures. No model, microphone, downloads, or saved settings.
//! cargo run --example ui-preview -- [ready|empty|models|recording|processing|choose|confirm|download|loading]
//! cargo run --example ui-preview -- --snapshot <dir>   (offscreen PPM renders, no window)
use eframe::egui;
use std::{collections::HashMap, sync::atomic::Ordering, time::Duration};
use whisper_burn::native::{
    config::AppConfig,
    download::{DownloadProgress, ModelVariant},
    hotkey::HotkeyCapture,
    ui::{
        download_screen, loading_screen, main_screen, model_manager_screen,
        status_indicator::AppStatus, theme,
    },
};

const SCREENS: [&str; 9] = [
    "ready",
    "empty",
    "models",
    "recording",
    "processing",
    "choose",
    "confirm",
    "download",
    "loading",
];

struct Preview {
    screen: String,
    config: AppConfig,
    capture: HotkeyCapture,
    progress: DownloadProgress,
    samples: Vec<f32>,
}

impl Preview {
    fn new(screen: &str) -> Self {
        let mut config = AppConfig::default();
        config.language = "fr".into();
        config.hotkey.modifiers = vec!["CONTROL".into(), "SUPER".into()];
        config.hotkey.key = String::new();
        config.auto_paste = true;
        config.auto_mute = true;
        let progress = DownloadProgress::default();
        progress.tokenizer_bytes.store(2_480_617, Ordering::Relaxed);
        progress.tokenizer_total.store(2_480_617, Ordering::Relaxed);
        progress.gguf_bytes.store(283_115_520, Ordering::Relaxed);
        progress.gguf_total.store(1_127_776_064, Ordering::Relaxed);
        let samples = (0..32000)
            .map(|i| {
                let t = i as f32 / 16000.0;
                (t * 800.0).sin() * (0.07 + (t * 8.0).sin().abs() * 0.45)
            })
            .collect();
        Self {
            screen: screen.into(),
            config,
            capture: HotkeyCapture::new(),
            progress,
            samples,
        }
    }

    fn draw(&mut self, ctx: &egui::Context) {
        theme::footer(ctx);
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG).inner_margin(20)).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                match self.screen.as_str() {
                    "models" => {
                        if matches!(model_manager_screen::draw(ui, ModelVariant::LargeV3), model_manager_screen::ModelManagerAction::Back) {
                            self.screen = "ready".into();
                        }
                    }
                    "recording" => main_screen::draw_recording(ui, &self.samples, 16000, Duration::from_millis(8400), "Ctrl + Win"),
                    "choose" => {
                        if matches!(download_screen::draw_choose_model(ui), download_screen::ChooseAction::Select(_)) { self.screen = "confirm".into(); }
                    }
                    "confirm" => match download_screen::draw_confirm(ui, ModelVariant::LargeV3) {
                        download_screen::ConfirmAction::Download => self.screen = "download".into(),
                        download_screen::ConfirmAction::Back => self.screen = "choose".into(),
                        _ => {}
                    },
                    "download" => download_screen::draw_progress(ui, &self.progress, ModelVariant::LargeV3),
                    "loading" => loading_screen::draw(ui, "Loading Whisper Large V3. This may take a minute..."),
                    _ => {
                        let text = if self.screen == "empty" { "" } else {
                            "Les idées viennent plus facilement quand on les dit à voix haute.\n\nPréparer le compte rendu de la réunion, puis partager les prochaines étapes avec l’équipe."
                        };
                        let status = if self.screen == "processing" { AppStatus::Processing } else if self.screen == "empty" { AppStatus::Ready } else { AppStatus::Done };
                        if matches!(main_screen::draw_ready(ui, text, 1270, ModelVariant::LargeV3, &mut self.config, status, &mut self.capture), main_screen::MainAction::OpenModelManager) {
                            self.screen = "models".into();
                        }
                    }
                }
            });
        });
    }
}

impl eframe::App for Preview {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        if !self.capture.listening {
            for (key, screen) in [
                egui::Key::F1,
                egui::Key::F2,
                egui::Key::F3,
                egui::Key::F4,
                egui::Key::F5,
                egui::Key::F6,
                egui::Key::F7,
                egui::Key::F8,
                egui::Key::F9,
            ]
            .into_iter()
            .zip(SCREENS)
            {
                if ctx.input(|input| input.key_pressed(key)) {
                    self.screen = screen.into();
                }
            }
            for (key, size) in [
                (egui::Key::F10, [640.0, 580.0]),
                (egui::Key::F11, [620.0, 540.0]),
                (egui::Key::F12, [780.0, 640.0]),
            ] {
                if ctx.input(|input| input.key_pressed(key)) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(size.into()));
                }
            }
        }
        self.draw(ctx);
    }
}

/// Renders egui's real output on the CPU, matching egui_glow's gamma-space blending.
fn render(screen: &str, size: [f32; 2], ppp: f32, click: Option<&str>) -> (usize, usize, Vec<u8>) {
    let ctx = egui::Context::default();
    theme::apply_dark_theme(&ctx);
    let mut preview = Preview::new(screen);
    // Each texture keeps a mip chain; colour images get box-filtered levels like glGenerateMipmap.
    let mut textures: HashMap<egui::TextureId, Vec<(usize, usize, Vec<egui::Color32>)>> =
        HashMap::new();
    let mut events = Vec::new();
    let mut output = None;
    for frame in 0..8 {
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size.into())),
            time: Some(0.35),
            events: std::mem::take(&mut events),
            ..Default::default()
        };
        input
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default()
            .native_pixels_per_point = Some(ppp);
        let out = ctx.run(input, |ctx| preview.draw(ctx));
        for (id, delta) in &out.textures_delta.set {
            let (w, h, pixels): (usize, usize, Vec<egui::Color32>) = match &delta.image {
                egui::ImageData::Color(image) => {
                    (image.size[0], image.size[1], image.pixels.clone())
                }
                egui::ImageData::Font(image) => (
                    image.size[0],
                    image.size[1],
                    image.srgba_pixels(None).collect(),
                ),
            };
            match delta.pos {
                None => {
                    let mut chain = vec![(w, h, pixels)];
                    while delta.options.mipmap_mode.is_some() && chain.last().unwrap().0 > 1 {
                        let (pw, ph, prev) = chain.last().unwrap();
                        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
                        let level = (0..nw * nh)
                            .map(|i| {
                                let (x, y) = (i % nw * 2, i / nw * 2);
                                let px = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                                    .map(|(x, y)| prev[y.min(ph - 1) * pw + x.min(pw - 1)]);
                                let avg = |c: fn(&egui::Color32) -> u8| {
                                    (px.iter().map(|p| c(p) as u32).sum::<u32>() / 4) as u8
                                };
                                egui::Color32::from_rgba_premultiplied(
                                    avg(|c| c.r()),
                                    avg(|c| c.g()),
                                    avg(|c| c.b()),
                                    avg(|c| c.a()),
                                )
                            })
                            .collect();
                        chain.push((nw, nh, level));
                    }
                    textures.insert(*id, chain);
                }
                Some([x0, y0]) => {
                    let (tw, _, tex) = &mut textures
                        .get_mut(id)
                        .expect("partial update of unknown texture")[0];
                    for y in 0..h {
                        tex[(y0 + y) * *tw + x0..][..w].copy_from_slice(&pixels[y * w..][..w]);
                    }
                }
            }
        }
        if let (2, Some(label)) = (frame, click) {
            let pos = out
                .shapes
                .iter()
                .find_map(|s| match &s.shape {
                    egui::Shape::Text(t) if t.galley.text() == label => {
                        Some(t.visual_bounding_rect().center())
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("no label {label}"));
            let press = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            events = vec![egui::Event::PointerMoved(pos), press(true)];
        } else if let (3, Some(_)) = (frame, click) {
            let pos = egui::pos2(-10.0, -10.0);
            events = vec![
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                },
                egui::Event::PointerGone,
            ];
        }
        output = Some(out);
    }
    let out = output.unwrap();
    let (w, h) = (
        (size[0] * ppp).round() as usize,
        (size[1] * ppp).round() as usize,
    );
    let mut fb = vec![[0.0_f32; 4]; w * h];
    let edge = |a: egui::Pos2, b: egui::Pos2, c: egui::Pos2| {
        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    };
    // Top-left style tie-break so shared edges are drawn exactly once.
    let owns = |a: egui::Pos2, b: egui::Pos2| b.y - a.y > 0.0 || (b.y == a.y && b.x < a.x);
    for prim in ctx.tessellate(out.shapes, out.pixels_per_point) {
        let egui::epaint::Primitive::Mesh(mesh) = prim.primitive else {
            continue;
        };
        let Some(chain) = textures.get(&mesh.texture_id) else {
            continue;
        };
        let clip = prim.clip_rect;
        let (cx0, cy0) = (
            (clip.min.x * ppp).round().max(0.0) as usize,
            (clip.min.y * ppp).round().max(0.0) as usize,
        );
        let (cx1, cy1) = (
            ((clip.max.x * ppp).round().max(0.0) as usize).min(w),
            ((clip.max.y * ppp).round().max(0.0) as usize).min(h),
        );
        for tri in mesh.indices.chunks_exact(3) {
            let mut v = [tri[0], tri[1], tri[2]].map(|i| mesh.vertices[i as usize]);
            let mut p = v.map(|v| egui::pos2(v.pos.x * ppp, v.pos.y * ppp));
            let area = edge(p[0], p[1], p[2]);
            if area.abs() < 1e-9 {
                continue;
            }
            if area < 0.0 {
                p.swap(1, 2);
                v.swap(1, 2);
            }
            let area = area.abs();
            // Pick the mip level from the texel-per-pixel ratio of this triangle.
            let (bw, bh) = (chain[0].0 as f32, chain[0].1 as f32);
            let uv = v.map(|v| egui::pos2(v.uv.x * bw, v.uv.y * bh));
            let ratio = (edge(uv[0], uv[1], uv[2]).abs() / area).sqrt();
            let (tw, th, tex) =
                &chain[(ratio.max(1.0).log2().floor() as usize).min(chain.len() - 1)];
            let texel = |x: i64, y: i64| {
                let c = tex[y.clamp(0, *th as i64 - 1) as usize * tw
                    + x.clamp(0, *tw as i64 - 1) as usize];
                [c.r(), c.g(), c.b(), c.a()].map(|v| v as f32 / 255.0)
            };
            let x0 = (p
                .iter()
                .map(|q| q.x)
                .fold(f32::MAX, f32::min)
                .floor()
                .max(0.0) as usize)
                .max(cx0);
            let y0 = (p
                .iter()
                .map(|q| q.y)
                .fold(f32::MAX, f32::min)
                .floor()
                .max(0.0) as usize)
                .max(cy0);
            let x1 = (p
                .iter()
                .map(|q| q.x)
                .fold(f32::MIN, f32::max)
                .ceil()
                .max(0.0) as usize)
                .min(cx1);
            let y1 = (p
                .iter()
                .map(|q| q.y)
                .fold(f32::MIN, f32::max)
                .ceil()
                .max(0.0) as usize)
                .min(cy1);
            for y in y0..y1 {
                for x in x0..x1 {
                    let q = egui::pos2(x as f32 + 0.5, y as f32 + 0.5);
                    let wts = [
                        edge(p[1], p[2], q),
                        edge(p[2], p[0], q),
                        edge(p[0], p[1], q),
                    ];
                    let edges = [(p[1], p[2]), (p[2], p[0]), (p[0], p[1])];
                    if (0..3)
                        .any(|i| wts[i] < 0.0 || (wts[i] == 0.0 && !owns(edges[i].0, edges[i].1)))
                    {
                        continue;
                    }
                    let b = wts.map(|wt| wt / area);
                    let mix = |f: &dyn Fn(&egui::epaint::Vertex) -> f32| {
                        b[0] * f(&v[0]) + b[1] * f(&v[1]) + b[2] * f(&v[2])
                    };
                    let (u, t) = (
                        mix(&|v| v.uv.x) * *tw as f32 - 0.5,
                        mix(&|v| v.uv.y) * *th as f32 - 0.5,
                    );
                    let (fx, fy) = (u - u.floor(), t - t.floor());
                    let (ix, iy) = (u.floor() as i64, t.floor() as i64);
                    let (s00, s10, s01, s11) = (
                        texel(ix, iy),
                        texel(ix + 1, iy),
                        texel(ix, iy + 1),
                        texel(ix + 1, iy + 1),
                    );
                    let d = &mut fb[y * w + x];
                    let col = [
                        mix(&|v| v.color.r() as f32),
                        mix(&|v| v.color.g() as f32),
                        mix(&|v| v.color.b() as f32),
                        mix(&|v| v.color.a() as f32),
                    ];
                    let src: [f32; 4] = std::array::from_fn(|k| {
                        let s = (s00[k] * (1.0 - fx) + s10[k] * fx) * (1.0 - fy)
                            + (s01[k] * (1.0 - fx) + s11[k] * fx) * fy;
                        col[k] / 255.0 * s
                    });
                    for k in 0..4 {
                        d[k] = src[k] + d[k] * (1.0 - src[3]);
                    }
                }
            }
        }
    }
    let rgb = fb
        .iter()
        .flat_map(|c| [c[0], c[1], c[2]].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8))
        .collect();
    (w, h, rgb)
}

fn snapshot(dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut jobs: Vec<(String, &str, [f32; 2], Option<&str>)> = Vec::new();
    for size in [[780.0, 640.0], [640.0, 580.0], [620.0, 540.0]] {
        for screen in SCREENS {
            jobs.push((
                format!("{screen}-{}x{}", size[0], size[1]),
                screen,
                size,
                None,
            ));
        }
    }
    jobs.push((
        "language-open-780x640".into(),
        "ready",
        [780.0, 640.0],
        Some("Français (fr)"),
    ));
    for (name, screen, size, click) in jobs {
        let (w, h, rgb) = render(screen, size, 1.25, click);
        let mut file = format!("P6\n{w} {h}\n255\n").into_bytes();
        file.extend(rgb);
        std::fs::write(dir.join(format!("{name}.ppm")), file)?;
    }
    Ok(())
}

fn main() -> eframe::Result {
    let mut args = std::env::args().skip(1);
    let screen = args.next().unwrap_or_else(|| "ready".into());
    if screen == "--snapshot" {
        let dir = args.next().expect("usage: ui-preview --snapshot <dir>");
        snapshot(std::path::Path::new(&dir)).expect("write snapshots");
        return Ok(());
    }
    eframe::run_native(
        "Whisper Burn — UI preview",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([780.0, 640.0])
                .with_min_inner_size([620.0, 540.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            theme::apply_dark_theme(&cc.egui_ctx);
            Ok(Box::new(Preview::new(&screen)))
        }),
    )
}
