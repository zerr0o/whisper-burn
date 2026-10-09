use std::sync::mpsc;
use tracing::{info, warn};

use crate::audio::AudioBuffer;
use crate::transcribe::{self, InferenceState};
use crate::Language;

pub struct InferenceRequest {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub language: Language,
}

pub type InferenceResponse = Result<(String, u128), String>;

pub struct InferenceHandle {
    pub tx: mpsc::Sender<InferenceRequest>,
    pub rx: mpsc::Receiver<InferenceResponse>,
}

pub fn spawn_inference_thread(state: InferenceState) -> InferenceHandle {
    let (req_tx, req_rx) = mpsc::channel::<InferenceRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<InferenceResponse>();

    std::thread::spawn(move || {
        info!("Inference thread started");
        loop {
            let Ok(InferenceRequest {
                samples,
                sample_rate,
                language,
            }) = req_rx.recv()
            else {
                info!("Inference thread shutting down");
                break;
            };
            let audio = AudioBuffer::new(samples, sample_rate);
            let resp = transcribe::transcribe(&state, audio, language).map_err(|e| {
                warn!("Inference error: {e}");
                e.to_string()
            });
            if resp_tx.send(resp).is_err() {
                break;
            }
        }
    });

    InferenceHandle {
        tx: req_tx,
        rx: resp_rx,
    }
}
