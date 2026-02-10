use eframe::egui;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::info;

use crate::gguf::loader::load_whisper_from_gguf;
use crate::model::config::WhisperConfig;
use crate::tokenizer::WhisperTokenizer;
use crate::transcribe::InferenceState;
use crate::Language;

use super::audio_capture::AudioCapture;
use super::download::{self, DownloadProgress, ModelVariant};
use super::inference::{
    InferenceHandle, InferenceRequest, InferenceResponse, spawn_inference_thread,
};

pub enum AppScreen {
    CheckModel,
    ChooseModel,
    ConfirmDownload(ModelVariant),
    Downloading(DownloadProgress, ModelVariant),
    LoadingModel(ModelVariant),
    Ready,
    Recording {
        capture: AudioCapture,
        all_samples: Vec<f32>,
        sample_rate: u32,
        started_at: Instant,
    },
    Transcribing,
}

pub struct NativeApp {
    screen: AppScreen,
    inference: Option<InferenceHandle>,
    last_result: String,
    last_inference_ms: u128,
    error_msg: Option<String>,
    selected_variant: ModelVariant,
    selected_lang: Language,
}

impl NativeApp {
    pub fn new() -> Self {
        Self {
            screen: AppScreen::CheckModel,
            inference: None,
            last_result: String::new(),
            last_inference_ms: 0,
            error_msg: None,
            selected_variant: ModelVariant::LargeV3Turbo,
            selected_lang: Language::French,
        }
    }

    fn start_model_load(&mut self, variant: ModelVariant) {
        self.selected_variant = variant;
        self.screen = AppScreen::LoadingModel(variant);
        let gguf_path = download::gguf_path(variant);
        let tokenizer_path = download::tokenizer_path();

        let config = match variant {
            ModelVariant::LargeV3 => WhisperConfig::large_v3(),
            ModelVariant::LargeV3Turbo => WhisperConfig::large_v3_turbo(),
        };

        let (tx, rx) = std::sync::mpsc::channel::<Result<InferenceState, String>>();

        std::thread::spawn(move || {
            let result = (|| -> Result<InferenceState, String> {
                let tokenizer = WhisperTokenizer::from_file(&tokenizer_path)
                    .map_err(|e| format!("Tokenizer load failed: {e}"))?;

                let device = Default::default();
                let model = load_whisper_from_gguf(&gguf_path, &config, &device)
                    .map_err(|e| format!("Model load failed: {e}"))?;

                Ok(InferenceState {
                    model,
                    tokenizer,
                    device,
                })
            })();
            let _ = tx.send(result);
        });

        LOAD_RX.with(|cell| {
            *cell.borrow_mut() = Some(rx);
        });
    }

    fn start_recording(&mut self) {
        match AudioCapture::start() {
            Ok(capture) => {
                let sample_rate = capture.sample_rate;
                info!("Audio capture started at {sample_rate} Hz");
                self.screen = AppScreen::Recording {
                    capture,
                    all_samples: Vec::new(),
                    sample_rate,
                    started_at: Instant::now(),
                };
            }
            Err(e) => {
                self.error_msg = Some(format!("Audio capture error: {e}"));
            }
        }
    }

    fn stop_recording(&mut self) {
        let screen = std::mem::replace(&mut self.screen, AppScreen::Transcribing);
        if let AppScreen::Recording {
            capture,
            mut all_samples,
            sample_rate,
            ..
        } = screen
        {
            let remaining = capture.take_samples();
            all_samples.extend_from_slice(&remaining);
            capture.stop();

            if all_samples.is_empty() {
                self.screen = AppScreen::Ready;
                return;
            }

            if let Some(handle) = &self.inference {
                let _ = handle.tx.send(InferenceRequest::Transcribe {
                    samples: all_samples,
                    sample_rate,
                    language: self.selected_lang,
                });
            }
        }
    }
}

thread_local! {
    static LOAD_RX: std::cell::RefCell<Option<std::sync::mpsc::Receiver<Result<InferenceState, String>>>> =
        std::cell::RefCell::new(None);
}

impl eframe::App for NativeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Always repaint while in dynamic states
        match &self.screen {
            AppScreen::Downloading(_, _)
            | AppScreen::LoadingModel(_)
            | AppScreen::Recording { .. }
            | AppScreen::Transcribing
            | AppScreen::CheckModel => {
                ctx.request_repaint();
            }
            _ => {}
        }

        // Poll inference responses
        if let Some(handle) = &self.inference {
            while let Ok(resp) = handle.rx.try_recv() {
                match resp {
                    InferenceResponse::Result {
                        text,
                        inference_ms,
                    } => {
                        self.last_result = text;
                        self.last_inference_ms = inference_ms;
                        self.screen = AppScreen::Ready;
                    }
                    InferenceResponse::Error(e) => {
                        self.error_msg = Some(e);
                        self.screen = AppScreen::Ready;
                    }
                }
            }
        }

        // State machine transitions
        match &self.screen {
            AppScreen::CheckModel => {
                // Check if any variant is already downloaded
                if download::models_present(ModelVariant::LargeV3Turbo) {
                    self.start_model_load(ModelVariant::LargeV3Turbo);
                } else if download::models_present(ModelVariant::LargeV3) {
                    self.start_model_load(ModelVariant::LargeV3);
                } else {
                    self.screen = AppScreen::ChooseModel;
                }
            }
            _ => {}
        }

        // Check model loading completion
        if matches!(self.screen, AppScreen::LoadingModel(_)) {
            let result = LOAD_RX.with(|cell| {
                let mut rx = cell.borrow_mut();
                if let Some(ref receiver) = *rx {
                    match receiver.try_recv() {
                        Ok(result) => {
                            *rx = None;
                            Some(result)
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => None,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            *rx = None;
                            Some(Err("Model loading thread crashed".into()))
                        }
                    }
                } else {
                    None
                }
            });

            if let Some(result) = result {
                match result {
                    Ok(state) => {
                        info!("Model loaded, starting inference thread");
                        let handle = spawn_inference_thread(state);
                        self.inference = Some(handle);
                        self.screen = AppScreen::Ready;
                    }
                    Err(e) => {
                        self.error_msg = Some(e);
                        self.screen = AppScreen::ChooseModel;
                    }
                }
            }
        }

        // Check download completion
        if let AppScreen::Downloading(ref progress, variant) = self.screen {
            if progress.done.load(Ordering::SeqCst) {
                let err = progress.error.lock().unwrap().clone();
                if let Some(e) = err {
                    self.error_msg = Some(e);
                    self.screen = AppScreen::ConfirmDownload(variant);
                } else {
                    self.start_model_load(variant);
                }
            }
        }

        // Accumulate samples during recording
        if let AppScreen::Recording {
            ref capture,
            ref mut all_samples,
            ..
        } = self.screen
        {
            let new_samples = capture.take_samples();
            if !new_samples.is_empty() {
                all_samples.extend_from_slice(&new_samples);
            }
        }

        // Render UI
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref err) = self.error_msg {
                ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
                ui.separator();
            }

            match &self.screen {
                AppScreen::CheckModel => {
                    super::ui::loading_screen::draw(ui, "Checking model files...");
                }
                AppScreen::ChooseModel => {
                    let action = super::ui::download_screen::draw_choose_model(ui);
                    match action {
                        super::ui::download_screen::ChooseAction::Select(variant) => {
                            self.error_msg = None;
                            self.selected_variant = variant;
                            self.screen = AppScreen::ConfirmDownload(variant);
                        }
                        super::ui::download_screen::ChooseAction::Quit => {
                            std::process::exit(0);
                        }
                        super::ui::download_screen::ChooseAction::None => {}
                    }
                }
                AppScreen::ConfirmDownload(variant) => {
                    let variant = *variant;
                    let action = super::ui::download_screen::draw_confirm(ui, variant);
                    match action {
                        super::ui::download_screen::ConfirmAction::Download => {
                            self.error_msg = None;
                            let progress = DownloadProgress::new();
                            download::spawn_download(&progress, variant);
                            self.screen = AppScreen::Downloading(progress, variant);
                        }
                        super::ui::download_screen::ConfirmAction::Back => {
                            self.screen = AppScreen::ChooseModel;
                        }
                        super::ui::download_screen::ConfirmAction::None => {}
                    }
                }
                AppScreen::Downloading(ref progress, variant) => {
                    super::ui::download_screen::draw_progress(ui, progress, *variant);
                }
                AppScreen::LoadingModel(variant) => {
                    let msg = format!("Loading {} (this may take a minute)...", variant.display_name());
                    super::ui::loading_screen::draw(ui, &msg);
                }
                AppScreen::Ready => {
                    let space_pressed = ctx.input(|i| i.key_down(egui::Key::Space));
                    let new_lang = super::ui::main_screen::draw_ready(
                        ui,
                        &self.last_result,
                        self.last_inference_ms,
                        self.selected_variant,
                        self.selected_lang,
                    );
                    if let Some(lang) = new_lang {
                        self.selected_lang = lang;
                    }
                    if space_pressed {
                        self.start_recording();
                    }
                }
                AppScreen::Recording {
                    all_samples,
                    started_at,
                    sample_rate,
                    ..
                } => {
                    let elapsed = started_at.elapsed();
                    let space_released = ctx.input(|i| i.key_released(egui::Key::Space));
                    super::ui::main_screen::draw_recording(
                        ui,
                        all_samples,
                        *sample_rate,
                        elapsed,
                    );
                    if space_released {
                        self.stop_recording();
                    }
                }
                AppScreen::Transcribing => {
                    super::ui::loading_screen::draw(ui, "Transcribing...");
                }
            }
        });
    }
}
