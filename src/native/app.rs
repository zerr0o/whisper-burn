use eframe::egui;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tracing::info;

use crate::gguf::loader::load_whisper_from_gguf;
use crate::model::config::WhisperConfig;
use crate::tokenizer::WhisperTokenizer;
use crate::transcribe::InferenceState;
use crate::Language;

use super::audio_capture::AudioCapture;
use super::config::{self, AppConfig};
use super::download::{self, DownloadProgress, ModelVariant};
use super::hotkey::{HotkeyEvent, HotkeyState};
use super::inference::{
    InferenceHandle, InferenceRequest, InferenceResponse, spawn_inference_thread,
};
use super::tray::{TrayAction, TrayState};
use super::ui::settings_panel::SettingsState;
use super::ui::status_indicator::AppStatus;

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
    ModelManager,
}

pub struct NativeApp {
    screen: AppScreen,
    inference: Option<InferenceHandle>,
    last_result: String,
    last_inference_ms: u128,
    error_msg: Option<String>,
    selected_variant: ModelVariant,
    selected_lang: Language,
    config: AppConfig,
    hotkey_state: Option<HotkeyState>,
    tray_state: Option<TrayState>,
    settings_state: SettingsState,
    status: AppStatus,
    hwnd: Option<isize>,
    window_visible: bool,
    #[cfg(windows)]
    audio_muter: Option<super::audio_mute::SystemAudioMuter>,
}

impl NativeApp {
    pub fn new(hwnd: Option<isize>) -> Self {
        let config = config::load_config();
        let selected_lang = *Language::from_code(&config.language);

        let selected_variant = match config.model_variant.as_str() {
            "medium" => ModelVariant::Medium,
            "large-v3" => ModelVariant::LargeV3,
            _ => ModelVariant::LargeV3Turbo,
        };

        let hotkey_state = HotkeyState::new(&config);
        let tray_state = TrayState::new();

        Self {
            screen: AppScreen::CheckModel,
            inference: None,
            last_result: String::new(),
            last_inference_ms: 0,
            error_msg: None,
            selected_variant,
            selected_lang,
            config,
            hotkey_state: Some(hotkey_state),
            tray_state: Some(tray_state),
            settings_state: SettingsState::default(),
            status: AppStatus::Ready,
            hwnd,
            window_visible: true,
            #[cfg(windows)]
            audio_muter: None,
        }
    }

    fn save_config(&self) {
        config::save_config(&self.config);
    }

    fn sync_lang_from_config(&mut self) {
        self.selected_lang = *Language::from_code(&self.config.language);
    }

    fn start_model_load(&mut self, variant: ModelVariant) {
        self.selected_variant = variant;
        self.config.model_variant = match variant {
            ModelVariant::Medium => "medium".into(),
            ModelVariant::LargeV3 => "large-v3".into(),
            ModelVariant::LargeV3Turbo => "large-v3-turbo".into(),
        };
        self.save_config();
        self.screen = AppScreen::LoadingModel(variant);
        let gguf_path = download::gguf_path(variant);
        let tokenizer_path = download::tokenizer_path();

        let config = match variant {
            ModelVariant::Medium => WhisperConfig::medium(),
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
                    n_mels: config.n_mels,
                })
            })();
            let _ = tx.send(result);
        });

        LOAD_RX.with(|cell| {
            *cell.borrow_mut() = Some(rx);
        });
    }

    fn start_recording(&mut self) {
        // Auto-mute system audio if enabled
        #[cfg(windows)]
        if self.config.auto_mute {
            match super::audio_mute::SystemAudioMuter::new() {
                Ok(muter) => {
                    let _ = muter.mute();
                    self.audio_muter = Some(muter);
                }
                Err(e) => {
                    tracing::warn!("Auto-mute failed: {e}");
                }
            }
        }

        match AudioCapture::start() {
            Ok(capture) => {
                let sample_rate = capture.sample_rate;
                info!("Audio capture started at {sample_rate} Hz");
                self.status = AppStatus::Recording;
                self.screen = AppScreen::Recording {
                    capture,
                    all_samples: Vec::new(),
                    sample_rate,
                    started_at: Instant::now(),
                };
            }
            Err(e) => {
                self.error_msg = Some(format!("Audio capture error: {e}"));
                // Restore mute on failure
                #[cfg(windows)]
                {
                    if let Some(muter) = self.audio_muter.take() {
                        let _ = muter.restore();
                    }
                }
            }
        }
    }

    fn stop_recording(&mut self) {
        // Restore system audio
        #[cfg(windows)]
        {
            if let Some(muter) = self.audio_muter.take() {
                let _ = muter.restore();
            }
        }

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
                self.status = AppStatus::Ready;
                return;
            }

            self.status = AppStatus::Processing;

            if let Some(handle) = &self.inference {
                let _ = handle.tx.send(InferenceRequest::Transcribe {
                    samples: all_samples,
                    sample_rate,
                    language: self.selected_lang,
                });
            }
        }
    }

    fn toggle_window_visibility(&mut self) {
        #[cfg(windows)]
        if let Some(hwnd) = self.hwnd {
            self.window_visible = !self.window_visible;
            super::tray::set_window_visible(hwnd, self.window_visible);
        }
    }

    #[allow(dead_code)]
    fn hide_window(&mut self) {
        #[cfg(windows)]
        if let Some(hwnd) = self.hwnd {
            self.window_visible = false;
            super::tray::set_window_visible(hwnd, false);
        }
    }
}

thread_local! {
    static LOAD_RX: std::cell::RefCell<Option<std::sync::mpsc::Receiver<Result<InferenceState, String>>>> =
        std::cell::RefCell::new(None);
}

impl eframe::App for NativeApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Periodic repaint for global hotkey + tray polling
        ctx.request_repaint_after(Duration::from_millis(100));

        // Poll global hotkey
        if let Some(ref mut hotkey) = self.hotkey_state {
            match hotkey.poll() {
                HotkeyEvent::Pressed => {
                    if matches!(self.screen, AppScreen::Ready) {
                        self.start_recording();
                    }
                }
                HotkeyEvent::Released => {
                    if matches!(self.screen, AppScreen::Recording { .. }) {
                        self.stop_recording();
                    }
                }
                HotkeyEvent::None => {}
            }
        }

        // Poll tray events
        if let Some(ref tray) = self.tray_state {
            match tray.poll() {
                TrayAction::ToggleWindow => {
                    self.toggle_window_visibility();
                }
                TrayAction::Quit => {
                    std::process::exit(0);
                }
                TrayAction::None => {}
            }
        }

        // Poll inference responses
        if let Some(handle) = &self.inference {
            while let Ok(resp) = handle.rx.try_recv() {
                match resp {
                    InferenceResponse::Result {
                        text,
                        inference_ms,
                    } => {
                        self.last_result = text.clone();
                        self.last_inference_ms = inference_ms;
                        self.status = AppStatus::Done;
                        self.screen = AppScreen::Ready;

                        // Auto-paste if enabled
                        if self.config.auto_paste && !text.is_empty() {
                            let text_clone = text;
                            std::thread::spawn(move || {
                                if let Err(e) = super::auto_paste::auto_paste(&text_clone) {
                                    tracing::warn!("Auto-paste failed: {e}");
                                }
                            });
                        }
                    }
                    InferenceResponse::Error(e) => {
                        self.error_msg = Some(e);
                        self.status = AppStatus::Ready;
                        self.screen = AppScreen::Ready;
                    }
                }
            }
        }

        // State machine transitions
        match &self.screen {
            AppScreen::CheckModel => {
                if download::models_present(self.selected_variant) {
                    self.start_model_load(self.selected_variant);
                } else if download::models_present(ModelVariant::LargeV3Turbo) {
                    self.start_model_load(ModelVariant::LargeV3Turbo);
                } else if download::models_present(ModelVariant::LargeV3) {
                    self.start_model_load(ModelVariant::LargeV3);
                } else if download::models_present(ModelVariant::Medium) {
                    self.start_model_load(ModelVariant::Medium);
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
                        self.status = AppStatus::Ready;
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

        // Draw settings overlay
        let settings_action = super::ui::settings_panel::draw(
            ctx,
            &mut self.config,
            &mut self.settings_state,
        );
        match settings_action {
            super::ui::settings_panel::SettingsAction::HotkeyChanged => {
                if let Some(ref mut hotkey) = self.hotkey_state {
                    hotkey.update_hotkey(&self.config);
                }
                self.save_config();
            }
            super::ui::settings_panel::SettingsAction::Close => {
                self.sync_lang_from_config();
                self.save_config();
            }
            super::ui::settings_panel::SettingsAction::None => {}
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
                AppScreen::Ready | AppScreen::Transcribing => {
                    let hotkey_display = HotkeyState::display_string(&self.config);
                    let action = super::ui::main_screen::draw_ready(
                        ui,
                        &self.last_result,
                        self.last_inference_ms,
                        self.selected_variant,
                        self.selected_lang,
                        &hotkey_display,
                        self.status,
                    );
                    match action {
                        super::ui::main_screen::MainAction::LanguageChanged(lang) => {
                            self.selected_lang = lang;
                            self.config.language = lang.code.unwrap_or("auto").to_string();
                            self.save_config();
                        }
                        super::ui::main_screen::MainAction::OpenSettings => {
                            self.settings_state.open = true;
                        }
                        super::ui::main_screen::MainAction::OpenModelManager => {
                            self.screen = AppScreen::ModelManager;
                        }
                        super::ui::main_screen::MainAction::None => {}
                    }
                }
                AppScreen::Recording {
                    all_samples,
                    started_at,
                    sample_rate,
                    ..
                } => {
                    let elapsed = started_at.elapsed();
                    let hotkey_display = HotkeyState::display_string(&self.config);
                    super::ui::main_screen::draw_recording(
                        ui,
                        all_samples,
                        *sample_rate,
                        elapsed,
                        &hotkey_display,
                    );
                }
                AppScreen::ModelManager => {
                    let action = super::ui::model_manager_screen::draw(
                        ui,
                        self.selected_variant,
                    );
                    match action {
                        super::ui::model_manager_screen::ModelManagerAction::Back => {
                            self.screen = AppScreen::Ready;
                        }
                        super::ui::model_manager_screen::ModelManagerAction::Delete(variant) => {
                            if let Err(e) = super::model_manager::delete_model(variant) {
                                self.error_msg = Some(e);
                            }
                        }
                        super::ui::model_manager_screen::ModelManagerAction::Switch(variant) => {
                            self.inference = None;
                            self.start_model_load(variant);
                        }
                        super::ui::model_manager_screen::ModelManagerAction::Download(variant) => {
                            self.error_msg = None;
                            let progress = DownloadProgress::new();
                            download::spawn_download(&progress, variant);
                            self.screen = AppScreen::Downloading(progress, variant);
                        }
                        super::ui::model_manager_screen::ModelManagerAction::None => {}
                    }
                }
            }
        });
    }
}
