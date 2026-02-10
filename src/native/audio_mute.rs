#[cfg(windows)]
pub struct SystemAudioMuter {
    endpoint_volume: windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume,
    was_muted: bool,
}

#[cfg(windows)]
impl SystemAudioMuter {
    pub fn new() -> Result<Self, String> {
        use windows::Win32::Media::Audio::{
            IMMDeviceEnumerator, MMDeviceEnumerator,
            eRender, eConsole,
        };
        use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
        use windows::Win32::System::Com::{
            CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED,
        };

        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .map_err(|e| format!("CoCreateInstance failed: {e}"))?;

            let device = enumerator
                .GetDefaultAudioEndpoint(eRender, eConsole)
                .map_err(|e| format!("GetDefaultAudioEndpoint failed: {e}"))?;

            let endpoint_volume: IAudioEndpointVolume = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|e| format!("Activate IAudioEndpointVolume failed: {e}"))?;

            let was_muted = endpoint_volume
                .GetMute()
                .map_err(|e| format!("GetMute failed: {e}"))?
                .as_bool();

            Ok(Self {
                endpoint_volume,
                was_muted,
            })
        }
    }

    pub fn mute(&self) -> Result<(), String> {
        unsafe {
            self.endpoint_volume
                .SetMute(true, std::ptr::null())
                .map_err(|e| format!("SetMute failed: {e}"))
        }
    }

    pub fn restore(&self) -> Result<(), String> {
        unsafe {
            self.endpoint_volume
                .SetMute(self.was_muted, std::ptr::null())
                .map_err(|e| format!("SetMute restore failed: {e}"))
        }
    }
}
