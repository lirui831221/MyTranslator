use std::ffi::CString;

#[repr(C)]
struct WhisperVadContextParams {
    n_threads: i32,
    use_gpu: bool,
    gpu_device: i32,
}

enum WhisperVadContext {}

#[link(name = "whisper")]
extern "C" {
    fn whisper_log_set(
        callback: Option<extern "C" fn(i32, *const std::ffi::c_char, *mut std::ffi::c_void)>,
        user_data: *mut std::ffi::c_void,
    );
    fn whisper_vad_init_from_file_with_params(
        path_model: *const std::ffi::c_char,
        params: WhisperVadContextParams,
    ) -> *mut WhisperVadContext;
    fn whisper_vad_detect_speech_no_reset(
        context: *mut WhisperVadContext,
        samples: *const f32,
        n_samples: i32,
    ) -> bool;
    fn whisper_vad_reset_state(context: *mut WhisperVadContext);
    fn whisper_vad_n_probs(context: *mut WhisperVadContext) -> i32;
    fn whisper_vad_probs(context: *mut WhisperVadContext) -> *mut f32;
    fn whisper_vad_free(context: *mut WhisperVadContext);
}

extern "C" fn quiet_whisper_log(
    _level: i32,
    _text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
}

pub struct NeuralVad {
    context: *mut WhisperVadContext,
}

impl NeuralVad {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let path = CString::new(model_path).map_err(|_| "invalid VAD model path")?;
        let params = WhisperVadContextParams {
            n_threads: 1,
            use_gpu: false,
            gpu_device: 0,
        };
        // This CPU-only whisper runtime is used solely for frame-level VAD;
        // suppress its per-frame diagnostic output.
        unsafe { whisper_log_set(Some(quiet_whisper_log), std::ptr::null_mut()) };
        // SAFETY: path is a live, NUL-terminated CString for the duration of
        // the call; whisper.cpp returns an owned context or null on failure.
        let context = unsafe { whisper_vad_init_from_file_with_params(path.as_ptr(), params) };
        if context.is_null() {
            return Err(format!("failed to load Silero VAD model: {}", model_path));
        }
        Ok(Self { context })
    }

    /// Score one 32 ms, 16 kHz mono frame (512 samples). The LSTM state is
    /// intentionally preserved across calls until `reset` is invoked.
    pub fn score(&mut self, samples: &[f32]) -> Result<f32, String> {
        if samples.len() != 512 {
            return Err(format!(
                "Silero VAD expects 512 samples, got {}",
                samples.len()
            ));
        }
        // SAFETY: context is owned by self and used only on the microphone
        // thread; samples points to 512 initialized f32 values.
        let ok = unsafe {
            whisper_vad_detect_speech_no_reset(self.context, samples.as_ptr(), samples.len() as i32)
        };
        if !ok {
            return Err("Silero VAD inference failed".into());
        }
        let count = unsafe { whisper_vad_n_probs(self.context) };
        let probabilities = unsafe { whisper_vad_probs(self.context) };
        if count < 1 || probabilities.is_null() {
            return Err("Silero VAD returned no probability".into());
        }
        let probability = unsafe { *probabilities.add((count - 1) as usize) };
        Ok(probability.clamp(0.0, 1.0))
    }

    pub fn reset(&mut self) {
        unsafe { whisper_vad_reset_state(self.context) };
    }
}

impl Drop for NeuralVad {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe { whisper_vad_free(self.context) };
            self.context = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NeuralVad;

    #[test]
    fn bundled_model_separates_silence_from_speech() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = root.join("models/ggml-silero-v6.2.0.bin");
        let mut vad = NeuralVad::load(model.to_str().unwrap()).unwrap();
        let silence_probability = vad.score(&[0.0; 512]).unwrap();
        vad.reset();

        let mut reader =
            hound::WavReader::open(root.join("benchmarks/audio/smoke/zh1.wav")).unwrap();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / i16::MAX as f32)
            .collect();
        let mut peak_probability = 0.0f32;
        for chunk in samples.chunks(512) {
            let mut frame = [0.0f32; 512];
            frame[..chunk.len()].copy_from_slice(chunk);
            peak_probability = peak_probability.max(vad.score(&frame).unwrap());
        }
        assert!(silence_probability < 0.2, "silence={silence_probability}");
        assert!(peak_probability > 0.5, "speech peak={peak_probability}");
    }
}
