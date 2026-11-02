use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const VAD_WATCHDOG_ZERO_PROBABILITY: f32 = 0.05;
const VAD_WATCHDOG_RECOVERY_PROBABILITY: f32 = 0.52;
const VAD_WATCHDOG_FALLBACK_MS: f32 = 384.0;
const VAD_WATCHDOG_RELOAD_MS: f32 = 4_000.0;
const VAD_WATCHDOG_RELOAD_COOLDOWN_MS: f32 = 30_000.0;
const VAD_WATCHDOG_QUIET_RESET_MS: f32 = 800.0;
const ENDPOINT_POST_ROLL_MS: u32 = 120;
const ENDPOINT_INCOMPLETE_MAX_SILENCE_MS: u32 = 900;
const SYSTEM_AUDIO_ZERO_SIGNAL_TIMEOUT_MS: f32 = 6_000.0;
// ScreenCaptureKit emits exact zeroes when a source is technically READY but
// no usable samples reach the helper. Keep this threshold far below ordinary
// room/player noise so the watchdog does not mistake quiet content for failure.
const SYSTEM_AUDIO_SIGNAL_RMS_THRESHOLD: f32 = 0.000_001;
const ROLLING_INCOMPLETE_IDLE_MAX_MS: u32 = 7_000;
const SPEAKER_CHANGE_MIN_PAUSE_MS: f32 = 96.0;
const SPEAKER_CHANGE_PROBE_WINDOW_MS: f32 = 160.0;
const SPEAKER_CHANGE_CONFIRMATION_STEP_MS: f32 = 80.0;
const SPEAKER_CHANGE_DEBOUNCE_MS: f32 = 250.0;
const SPEAKER_PROFILE_MS: f32 = 360.0;
const SPEAKER_CHANGE_SILENCE_PREROLL_MS: f32 = 64.0;
const CROSS_SEGMENT_PROFILE_MS: f32 = 300.0;
const CROSS_SEGMENT_SAME_MAX_SCORE: f32 = 0.24;
const CROSS_SEGMENT_WINDOW_STABILITY_MAX_SCORE: f32 = 0.18;
static WAV_SEQUENCE: AtomicU64 = AtomicU64::new(1);

fn next_wav_path() -> std::path::PathBuf {
    let sequence = WAV_SEQUENCE.fetch_add(1, Ordering::AcqRel);
    std::env::temp_dir().join(format!("myt-seg-{}-{}.wav", std::process::id(), sequence))
}

#[derive(Default)]
struct VadWatchdog {
    strong_zero_ms: f32,
    quiet_ms: f32,
    reload_cooldown_ms: f32,
    energy_fallback: bool,
}

struct VadWatchdogUpdate {
    activated_fallback: bool,
    reload_neural_vad: bool,
}

#[derive(Default)]
struct SystemAudioSignalWatchdog {
    zero_signal_ms: f32,
    signal_seen: bool,
}

impl SystemAudioSignalWatchdog {
    fn observe(&mut self, rms: f32, chunk_ms: f32) -> bool {
        if self.signal_seen {
            return false;
        }
        if rms > SYSTEM_AUDIO_SIGNAL_RMS_THRESHOLD {
            // Once this capture session has produced real samples, later
            // digital silence is a legitimate pause and must not force a
            // source switch.
            self.signal_seen = true;
            self.zero_signal_ms = 0.0;
            return false;
        }
        self.zero_signal_ms += chunk_ms;
        self.zero_signal_ms >= SYSTEM_AUDIO_ZERO_SIGNAL_TIMEOUT_MS
    }
}

#[derive(Clone, Copy, Debug)]
struct VoiceSignature {
    pitch_hz: f32,
    pitch_confidence: f32,
    zero_crossing_rate: f32,
    log_rms: f32,
    spectral_shape: [f32; 8],
}

struct SpeakerTurnProbe {
    old_signature: VoiceSignature,
    new_speech_start_sample: usize,
    pause_ms: f32,
    new_speech_samples: Vec<f32>,
}

struct CrossSegmentSpeakerProbe {
    old_signature: VoiceSignature,
    new_speech_samples: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CrossSegmentSpeakerDecision {
    Same,
    Different,
    Uncertain,
}

#[derive(Clone, Copy, Debug)]
struct CrossSegmentSpeakerComparison {
    decision: CrossSegmentSpeakerDecision,
    score: f32,
    confirmations: usize,
}

impl VadWatchdog {
    fn observe(
        &mut self,
        neural_probability: Option<f32>,
        rms: f32,
        start_threshold: f32,
        chunk_ms: f32,
    ) -> VadWatchdogUpdate {
        self.reload_cooldown_ms = (self.reload_cooldown_ms - chunk_ms).max(0.0);
        let was_fallback = self.energy_fallback;
        let strong_audio = rms >= watchdog_strong_threshold(start_threshold);
        let neural_zero = neural_probability
            .is_some_and(|probability| probability <= VAD_WATCHDOG_ZERO_PROBABILITY);

        if neural_probability
            .is_some_and(|probability| probability >= VAD_WATCHDOG_RECOVERY_PROBABILITY)
        {
            self.clear_signal();
        } else if neural_zero && strong_audio {
            self.strong_zero_ms += chunk_ms;
            self.quiet_ms = 0.0;
            if self.strong_zero_ms >= VAD_WATCHDOG_FALLBACK_MS {
                self.energy_fallback = true;
            }
        } else if self.energy_fallback && rms < start_threshold {
            self.quiet_ms += chunk_ms;
            if self.quiet_ms >= VAD_WATCHDOG_QUIET_RESET_MS {
                self.clear_signal();
            }
        } else if !self.energy_fallback {
            // The fallback requires sustained strong audio, not a click or a
            // single loud frame.
            self.strong_zero_ms = 0.0;
            self.quiet_ms = 0.0;
        } else {
            self.quiet_ms = 0.0;
        }

        let reload_neural_vad = self.energy_fallback
            && self.strong_zero_ms >= VAD_WATCHDOG_RELOAD_MS
            && self.reload_cooldown_ms <= 0.0;
        if reload_neural_vad {
            self.reload_cooldown_ms = VAD_WATCHDOG_RELOAD_COOLDOWN_MS;
            // Keep the constrained energy fallback active while the fresh
            // Silero context receives its first frames.
            self.strong_zero_ms = VAD_WATCHDOG_FALLBACK_MS;
        }

        VadWatchdogUpdate {
            activated_fallback: !was_fallback && self.energy_fallback,
            reload_neural_vad,
        }
    }

    fn clear_signal(&mut self) {
        self.strong_zero_ms = 0.0;
        self.quiet_ms = 0.0;
        self.energy_fallback = false;
    }
}

/// Simple energy-based VAD + segmenter:
/// - collects mic samples until `end_ms` of continuous silence
/// - emits a WAV (16 kHz mono s16le) path per utterance
pub struct VoicePipeline {
    pub running: Arc<AtomicBool>,
}

/// A medium-model recognition checkpoint. Rolling checkpoints keep the
/// utterance open while `is_final` closes it on silence or the hard limit.
pub struct VoiceChunk {
    pub wav_path: String,
    pub utterance_id: u64,
    pub speaker_turn_id: u64,
    pub is_final: bool,
    pub end_reason: SegmentEndReason,
    pub speaker_boundary_before: bool,
    pub speaker_turn_end: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentEndReason {
    Ongoing,
    ConfirmedSilence,
    StrongSilence,
    RollingLimit,
    SpeakerChange,
}

impl SegmentEndReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ongoing => "ongoing",
            Self::ConfirmedSilence => "confirmed-silence",
            Self::StrongSilence => "strong-silence",
            Self::RollingLimit => "rolling-limit",
            Self::SpeakerChange => "speaker-change",
        }
    }
}

#[derive(Clone, Copy)]
struct VoiceChunkMetadata {
    utterance_id: u64,
    speaker_turn_id: u64,
    is_final: bool,
    end_reason: SegmentEndReason,
    speaker_boundary_before: bool,
    speaker_turn_end: bool,
}

impl VoicePipeline {
    pub fn new() -> Self {
        VoicePipeline {
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run the selected audio capture loop. Calls `on_segment(chunk)` for rolling
    /// medium-model checkpoints and `on_partial(chunk)` for the fast,
    /// provisional base-model preview.
    /// Blocking; spawn it on a dedicated thread.
    pub fn run<F, P>(&self, on_segment: F, on_partial: P) -> Result<(), String>
    where
        F: Fn(VoiceChunk) + Send + 'static,
        P: Fn(VoiceChunk) + Send + 'static,
    {
        let running = self.running.clone();
        let input_mode = crate::voice_input_mode();
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let mut microphone_stream: Option<cpal::Stream> = None;
        let mut system_audio_child: Option<std::process::Child> = None;

        let sample_rate = if input_mode == "system" {
            let helper_path = crate::system_audio_helper_path();
            if !std::path::Path::new(&helper_path).exists() {
                return Err(format!("system audio helper not found: {}", helper_path));
            }
            let mut child = std::process::Command::new(&helper_path)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .map_err(|error| format!("system audio capture failed to start: {}", error))?;
            let stdout = child
                .stdout
                .take()
                .ok_or("system audio capture has no audio output")?;
            let stderr = child.stderr.take();
            let source_tx = tx.clone();
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stdout);
                let mut bytes = [0_u8; 8192];
                let mut pending = Vec::with_capacity(8192);
                loop {
                    match reader.read(&mut bytes) {
                        Ok(0) => break,
                        Ok(count) => {
                            let samples = append_f32le_samples(&mut pending, &bytes[..count]);
                            if !samples.is_empty() && source_tx.send(samples).is_err() {
                                break;
                            }
                        }
                        Err(error) => {
                            crate::log_line(&format!(
                                "[system-audio] failed to read capture stream: {}",
                                error
                            ));
                            break;
                        }
                    }
                }
            });
            let stderr = stderr.ok_or("system audio capture has no status output")?;
            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
            std::thread::spawn(move || {
                let mut readiness = Some(ready_tx);
                for line in BufReader::new(stderr).lines() {
                    match line {
                        Ok(message) => {
                            crate::log_line(&format!("[system-audio] {}", message));
                            if message.starts_with("READY") {
                                if let Some(sender) = readiness.take() {
                                    let _ = sender.send(Ok(()));
                                }
                            } else if message.starts_with("ERROR") {
                                if let Some(sender) = readiness.take() {
                                    let detail = message
                                        .strip_prefix("ERROR")
                                        .unwrap_or(&message)
                                        .trim()
                                        .to_string();
                                    let _ = sender.send(Err(if detail.is_empty() {
                                        "ScreenCaptureKit reported an unknown error".into()
                                    } else {
                                        detail
                                    }));
                                }
                            }
                        }
                        Err(error) => {
                            if let Some(sender) = readiness.take() {
                                let _ = sender.send(Err(format!(
                                    "failed to read ScreenCaptureKit status: {}",
                                    error
                                )));
                            }
                            break;
                        }
                    }
                }
                if let Some(sender) = readiness.take() {
                    let _ =
                        sender.send(Err("ScreenCaptureKit exited before reporting READY".into()));
                }
            });
            let readiness = ready_rx.recv_timeout(std::time::Duration::from_secs(4));
            if let Err(error) = readiness.unwrap_or_else(|error| {
                Err(match error {
                    std::sync::mpsc::RecvTimeoutError::Timeout => {
                        "timed out waiting 4 seconds for ScreenCaptureKit READY; check Screen Recording and System Audio permissions".into()
                    }
                    std::sync::mpsc::RecvTimeoutError::Disconnected => {
                        "ScreenCaptureKit status channel closed before READY".into()
                    }
                })
            }) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
            crate::log_line("[audio] source=system, ScreenCaptureKit 16kHz mono");
            system_audio_child = Some(child);
            16_000.0
        } else {
            let host = cpal::default_host();
            let device = host.default_input_device().ok_or("no microphone found")?;
            let cfg = device
                .default_input_config()
                .map_err(|e| format!("mic config failed: {}", e))?;
            let sample_rate = cfg.sample_rate().0 as f32;
            let channels = cfg.channels() as usize;
            let source_tx = tx.clone();
            let err_fn =
                |error| crate::log_line(&format!("[audio] microphone stream error: {}", error));
            let stream = device
                .build_input_stream(
                    &cfg.into(),
                    move |data: &[f32], _| {
                        let mono: Vec<f32> = if channels > 1 {
                            data.chunks(channels)
                                .map(|channel| channel.iter().sum::<f32>() / channels as f32)
                                .collect()
                        } else {
                            data.to_vec()
                        };
                        let _ = source_tx.send(mono);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("microphone stream build failed: {}", e))?;
            stream
                .play()
                .map_err(|e| format!("microphone stream start failed: {}", e))?;
            crate::log_line(&format!(
                "[audio] source=microphone, sample_rate={:.0}, channels={}",
                sample_rate, channels
            ));
            microphone_stream = Some(stream);
            sample_rate
        };
        // Only producer threads retain senders. If the selected source exits,
        // the receiver disconnects and the capture session can fail visibly.
        drop(tx);

        let neural_vad_path = crate::neural_vad_model_path();
        let mut neural_vad = neural_vad_path.as_deref().and_then(|path| {
            match crate::neural_vad::NeuralVad::load(path) {
                Ok(vad) => {
                    crate::log_line("[vad] Silero v6.2 streaming VAD ready");
                    Some(vad)
                }
                Err(error) => {
                    crate::log_line(&format!(
                        "[vad] Silero unavailable, using energy fallback: {}",
                        error
                    ));
                    None
                }
            }
        });
        let neural_native_frame = (sample_rate * 0.032).round().max(1.0) as usize;
        let mut neural_native_buffer: Vec<f32> = Vec::with_capacity(neural_native_frame * 2);
        let mut neural_probability: Option<f32> = None;
        let mut vad_watchdog = VadWatchdog::default();

        // Adaptive VAD with real hysteresis: speech must cross the higher
        // start threshold, but stays open until it drops below the lower end
        // threshold. Absolute floors keep very quiet rooms from amplifying
        // microphone self-noise into false speech.
        let mut noise_floor: f32 = 0.006;
        let noise_adapt: f32 = 0.98;
        let mut next_partial_ms: f32 = 550.0;
        let mut next_checkpoint_ms: f32 = 1_800.0;
        let mut above_th_frames: u32 = 0;

        let mut segment: Vec<f32> = Vec::new();
        let mut pre_roll: VecDeque<f32> = VecDeque::new();
        // The same buffer protects natural onsets and becomes the overlap at
        // a hard rolling boundary, where there may be no silence at all.
        let pre_roll_samples = (sample_rate * 0.48) as usize;
        let mut silence_ms: f32 = 0.0;
        // Strong neural silence is tracked separately from ordinary endpoint
        // silence. In steady mode this lets an unambiguous sentence pause
        // close promptly while still giving uncertain pauses the full 800ms.
        let mut strong_silence_ms: f32 = 0.0;
        // Fast mode first marks a candidate endpoint, then keeps recording a
        // short post-roll. A recent ASR hint may extend only an unfinished
        // clause; ordinary sentences never fall back to the old 800ms wait.
        let mut endpoint_candidate_ms: Option<u32> = None;
        let mut pause_start_sample: Option<usize> = None;
        let mut speaker_turn_probe: Option<SpeakerTurnProbe> = None;
        let mut cross_segment_probe: Option<CrossSegmentSpeakerProbe> = None;
        let mut previous_natural_signature: Option<VoiceSignature> = None;
        let mut previous_natural_segment_seen = false;
        let mut preserve_speaker_turn_across_rolling_boundary = false;
        let mut current_speaker_boundary_before = false;
        let mut speaker_change_cooldown_ms: f32 = 0.0;
        let mut speech_elapsed_ms: f32 = 0.0;
        let mut voiced_ms: f32 = 0.0;
        let mut peak_rms: f32 = 0.0;
        let mut in_speech = false;
        crate::set_voice_activity(false);
        let mut utterance_id: u64 = 0;
        let mut speaker_turn_id: u64 = 1;
        // diagnostics: log mic RMS every 2s so thresholds can be tuned
        let mut diag_elapsed: f32 = 0.0;
        let mut diag_max_rms: f32 = 0.0;
        let mut diag_samples: u32 = 0;
        let mut system_signal_watchdog = SystemAudioSignalWatchdog::default();

        let mut source_error: Option<String> = None;
        while running.load(Ordering::Relaxed) {
            // Short conversational replies can close more aggressively. Long
            // or pause-heavy speech keeps the proven 420 ms budget so a brief
            // thinking pause does not fragment the sentence.
            let steady_mode = crate::vad_steady_mode();
            let (base_end_ms, max_ms) = endpoint_policy(steady_mode, voiced_ms, speech_elapsed_ms);
            let end_ms = base_end_ms;
            let strong_end_ms = strong_silence_endpoint_ms(steady_mode, end_ms);
            let chunk = match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(c) => c,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    source_error = Some(if input_mode == "system" {
                        "system audio capture stopped; allow Screen Recording/System Audio permission in macOS Settings, then retry".to_string()
                    } else {
                        "microphone capture stopped unexpectedly".to_string()
                    });
                    break;
                }
            };
            let rms = (chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len().max(1) as f32).sqrt();
            let chunk_ms = chunk.len() as f32 / sample_rate * 1000.0;
            if input_mode == "system" && system_signal_watchdog.observe(rms, chunk_ms) {
                let error = "ScreenCaptureKit 已就绪，但系统音频连续 6 秒为数字零信号；播放器可能没有输出、内容受保护，或系统录音权限尚未生效。已自动回退麦克风";
                crate::log_line(&format!("[system-audio] zero-signal watchdog: {}", error));
                source_error = Some(error.to_string());
                break;
            }
            speaker_change_cooldown_ms = (speaker_change_cooldown_ms - chunk_ms).max(0.0);

            if neural_vad.is_some() {
                neural_native_buffer.extend_from_slice(&chunk);
                while neural_native_buffer.len() >= neural_native_frame {
                    let native: Vec<f32> =
                        neural_native_buffer.drain(..neural_native_frame).collect();
                    let frame = resample_fixed(&native, 512);
                    let score = neural_vad.as_mut().unwrap().score(&frame);
                    match score {
                        Ok(probability) => neural_probability = Some(probability),
                        Err(error) => {
                            crate::log_line(&format!(
                                "[vad] Silero inference failed, using energy fallback: {}",
                                error
                            ));
                            neural_vad = None;
                            neural_probability = None;
                            neural_native_buffer.clear();
                            break;
                        }
                    }
                }
            }

            let (watchdog_start_threshold, _) = vad_thresholds(noise_floor);
            let watchdog_update =
                vad_watchdog.observe(neural_probability, rms, watchdog_start_threshold, chunk_ms);
            if watchdog_update.activated_fallback {
                crate::log_line(&format!(
                    "[vad-watchdog] Silero stayed near zero during strong audio; enabling constrained energy fallback (rms={:.4}, threshold={:.4})",
                    rms,
                    watchdog_strong_threshold(watchdog_start_threshold)
                ));
            }
            if watchdog_update.reload_neural_vad {
                if let Some(path) = neural_vad_path.as_deref() {
                    match crate::neural_vad::NeuralVad::load(path) {
                        Ok(reloaded) => {
                            neural_vad = Some(reloaded);
                            neural_native_buffer.clear();
                            neural_probability = None;
                            crate::log_line(
                                "[vad-watchdog] rebuilt Silero context after 4s pathological zero output",
                            );
                        }
                        Err(error) => crate::log_line(&format!(
                            "[vad-watchdog] Silero rebuild failed; keeping constrained fallback: {}",
                            error
                        )),
                    }
                }
            }

            // AEC guard: while reading a translation aloud, discard any open
            // utterance rather than joining pre-TTS and post-TTS speech.
            if crate::is_self_speaking() {
                segment.clear();
                pre_roll.clear();
                in_speech = false;
                crate::set_voice_activity(false);
                above_th_frames = 0;
                silence_ms = 0.0;
                strong_silence_ms = 0.0;
                endpoint_candidate_ms = None;
                pause_start_sample = None;
                speaker_turn_probe = None;
                cross_segment_probe = None;
                previous_natural_signature = None;
                previous_natural_segment_seen = true;
                preserve_speaker_turn_across_rolling_boundary = false;
                current_speaker_boundary_before = false;
                speaker_change_cooldown_ms = 0.0;
                speech_elapsed_ms = 0.0;
                voiced_ms = 0.0;
                peak_rms = 0.0;
                next_partial_ms = 550.0;
                next_checkpoint_ms = 1_800.0;
                neural_native_buffer.clear();
                neural_probability = None;
                vad_watchdog.clear_signal();
                if let Some(vad) = neural_vad.as_mut() {
                    vad.reset();
                }
                continue;
            }

            // ---- diagnostics ----
            diag_elapsed += chunk_ms;
            diag_samples += 1;
            if rms > diag_max_rms {
                diag_max_rms = rms;
            }
            if diag_elapsed >= 2000.0 {
                crate::record_audio_environment(diag_max_rms, noise_floor, neural_probability);
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/myt-voice-test.log")
                {
                    use std::io::Write;
                    let _ = writeln!(
                        f,
                        "[vad-diag] max_rms={:.4} noise_floor={:.4} start_th={:.4} neural_p={:.3} watchdog_fallback={} chunks={} in_speech={}",
                        diag_max_rms, noise_floor, vad_thresholds(noise_floor).0,
                        neural_probability.unwrap_or(-1.0), vad_watchdog.energy_fallback,
                        diag_samples, in_speech
                    );
                }
                diag_elapsed = 0.0;
                diag_max_rms = 0.0;
                diag_samples = 0;
            }
            // ---- end diagnostics ----

            let mut started_now = false;
            if !in_speech {
                // Only learn samples that still look like background. This
                // prevents a soft speech onset from raising the floor until
                // the speaker can no longer cross it.
                if rms < (noise_floor * 1.55).max(0.018) {
                    noise_floor = (noise_adapt * noise_floor + (1.0 - noise_adapt) * rms)
                        .clamp(0.0015, 0.035);
                }
                let (start_th, _) = vad_thresholds(noise_floor);

                pre_roll.extend(chunk.iter().copied());
                while pre_roll.len() > pre_roll_samples {
                    pre_roll.pop_front();
                }

                if is_speech_frame_with_watchdog(
                    neural_probability,
                    rms,
                    start_th,
                    false,
                    vad_watchdog.energy_fallback,
                ) {
                    above_th_frames += 1;
                    if above_th_frames >= 3 {
                        in_speech = true;
                        crate::set_voice_activity(true);
                        started_now = true;
                        segment.extend(pre_roll.drain(..));
                        silence_ms = 0.0;
                        strong_silence_ms = 0.0;
                        endpoint_candidate_ms = None;
                        pause_start_sample = None;
                        speaker_turn_probe = None;
                        speech_elapsed_ms = above_th_frames as f32 * chunk_ms;
                        voiced_ms = speech_elapsed_ms;
                        peak_rms = rms;
                        above_th_frames = 0;
                        next_partial_ms = 550.0;
                        next_checkpoint_ms = if preserve_speaker_turn_across_rolling_boundary {
                            1_200.0
                        } else {
                            1_800.0
                        };
                        utterance_id = utterance_id.wrapping_add(1);
                        current_speaker_boundary_before = false;
                        cross_segment_probe = None;
                        if preserve_speaker_turn_across_rolling_boundary {
                            preserve_speaker_turn_across_rolling_boundary = false;
                            crate::log_line(&format!(
                                "[speaker-cross] preserved turn={} across rolling boundary, utterance={}",
                                speaker_turn_id, utterance_id
                            ));
                        } else if previous_natural_segment_seen {
                            previous_natural_segment_seen = false;
                            if let Some(old_signature) = previous_natural_signature.take() {
                                cross_segment_probe = Some(CrossSegmentSpeakerProbe {
                                    old_signature,
                                    new_speech_samples: Vec::new(),
                                });
                            } else {
                                speaker_turn_id = next_speaker_turn_id(speaker_turn_id);
                                current_speaker_boundary_before = true;
                                crate::log_line(&format!(
                                    "[speaker-cross] uncertain: previous profile unavailable; starting turn={} before utterance={}",
                                    speaker_turn_id, utterance_id
                                ));
                            }
                        }
                    }
                } else {
                    above_th_frames = 0;
                }
            }

            if in_speech {
                if !started_now {
                    segment.extend_from_slice(&chunk);
                    speech_elapsed_ms += chunk_ms;
                }
                peak_rms = peak_rms.max(rms);
                let (_, end_th) = vad_thresholds(noise_floor);
                let speech_frame = is_speech_frame_with_watchdog(
                    neural_probability,
                    rms,
                    end_th,
                    true,
                    vad_watchdog.energy_fallback,
                );
                if !speech_frame {
                    if silence_ms <= f32::EPSILON {
                        pause_start_sample = Some(segment.len().saturating_sub(chunk.len()));
                    }
                    silence_ms += chunk_ms;
                    if neural_probability.is_some_and(|probability| probability <= 0.12) {
                        strong_silence_ms += chunk_ms;
                    } else {
                        strong_silence_ms = 0.0;
                    }
                } else {
                    let resumed_after_ms = silence_ms;
                    if speaker_turn_probe.is_none()
                        && speaker_change_cooldown_ms <= f32::EPSILON
                        && resumed_after_ms >= SPEAKER_CHANGE_MIN_PAUSE_MS
                        && voiced_ms >= 400.0
                    {
                        if let Some(pause_start) = pause_start_sample {
                            let profile_samples =
                                (sample_rate * SPEAKER_PROFILE_MS / 1000.0) as usize;
                            let profile_start = pause_start.saturating_sub(profile_samples);
                            if let Some(old_signature) =
                                voice_signature(&segment[profile_start..pause_start], sample_rate)
                            {
                                speaker_turn_probe = Some(SpeakerTurnProbe {
                                    old_signature,
                                    new_speech_start_sample: segment
                                        .len()
                                        .saturating_sub(chunk.len()),
                                    pause_ms: resumed_after_ms,
                                    new_speech_samples: Vec::new(),
                                });
                            }
                        }
                    }
                    if let Some(probe) = speaker_turn_probe.as_mut() {
                        probe.new_speech_samples.extend_from_slice(&chunk);
                    }
                    if let Some(probe) = cross_segment_probe.as_mut() {
                        probe.new_speech_samples.extend_from_slice(&chunk);
                    }
                    silence_ms = 0.0;
                    strong_silence_ms = 0.0;
                    endpoint_candidate_ms = None;
                    pause_start_sample = None;
                    if !started_now {
                        voiced_ms += chunk_ms;
                    }
                }

                let completed_cross_segment_probe = cross_segment_probe
                    .as_ref()
                    .is_some_and(|probe| {
                        probe.new_speech_samples.len()
                            >= cross_segment_probe_samples_needed(sample_rate)
                    })
                    .then(|| cross_segment_probe.take().unwrap());
                if let Some(probe) = completed_cross_segment_probe {
                    let comparison = compare_cross_segment_speaker(&probe, sample_rate);
                    match comparison.decision {
                        CrossSegmentSpeakerDecision::Same => crate::log_line(&format!(
                            "[speaker-cross] same speaker confirmed turn={} utterance={} score={:.2} confirmations={}/2",
                            speaker_turn_id,
                            utterance_id,
                            comparison.score,
                            comparison.confirmations
                        )),
                        CrossSegmentSpeakerDecision::Different
                        | CrossSegmentSpeakerDecision::Uncertain => {
                            speaker_turn_id = next_speaker_turn_id(speaker_turn_id);
                            current_speaker_boundary_before = true;
                            if comparison.decision == CrossSegmentSpeakerDecision::Different {
                                crate::record_speaker_change(comparison.score);
                            }
                            crate::log_line(&format!(
                                "[speaker-cross] {:?}; starting turn={} before utterance={} score={:.2} confirmations={}/2",
                                comparison.decision,
                                speaker_turn_id,
                                utterance_id,
                                comparison.score,
                                comparison.confirmations
                            ));
                        }
                    }
                }

                // Confirm the new voice twice on overlapping windows. The
                // second window begins 80ms after the first, so a transient
                // onset or plosive cannot force a speaker boundary by itself
                // while the decision still arrives after roughly 240ms.
                let probe_window_samples =
                    (sample_rate * SPEAKER_CHANGE_PROBE_WINDOW_MS / 1000.0) as usize;
                let confirmation_step_samples =
                    (sample_rate * SPEAKER_CHANGE_CONFIRMATION_STEP_MS / 1000.0) as usize;
                let probe_samples_needed = probe_window_samples + confirmation_step_samples;
                let completed_speaker_probe = speaker_turn_probe
                    .as_ref()
                    .is_some_and(|probe| probe.new_speech_samples.len() >= probe_samples_needed)
                    .then(|| speaker_turn_probe.take().unwrap());
                if let Some(probe) = completed_speaker_probe {
                    let first_window = &probe.new_speech_samples[..probe_window_samples];
                    let second_window = &probe.new_speech_samples[confirmation_step_samples
                        ..confirmation_step_samples + probe_window_samples];
                    if let (Some(first_signature), Some(second_signature)) = (
                        voice_signature(first_window, sample_rate),
                        voice_signature(second_window, sample_rate),
                    ) {
                        let (changed, score, confirmations, threshold) = speaker_change_confirmed(
                            probe.old_signature,
                            first_signature,
                            second_signature,
                            probe.pause_ms,
                        );
                        crate::log_line(&format!(
                            "[speaker] probe utterance={} pause={:.0}ms score={:.2} changed={} threshold={:.2} confirmations={}/2 old_pitch={:.0}/{:.2} new_pitch={:.0}/{:.2}",
                            utterance_id,
                            probe.pause_ms,
                            score,
                            changed,
                            threshold,
                            confirmations,
                            probe.old_signature.pitch_hz,
                            probe.old_signature.pitch_confidence,
                            second_signature.pitch_hz,
                            second_signature.pitch_confidence
                        ));
                        if changed {
                            let old_segment_end = probe.new_speech_start_sample.min(segment.len());
                            let old_segment_flushed = old_segment_end > 0;
                            if old_segment_end > 0 {
                                self.flush_chunk(
                                    &segment[..old_segment_end],
                                    sample_rate,
                                    VoiceChunkMetadata {
                                        utterance_id,
                                        speaker_turn_id,
                                        is_final: true,
                                        end_reason: SegmentEndReason::SpeakerChange,
                                        speaker_boundary_before: current_speaker_boundary_before,
                                        speaker_turn_end: true,
                                    },
                                    &on_segment,
                                );
                            }
                            crate::record_speaker_change(score);
                            crate::clear_endpoint_semantic_hint(utterance_id);

                            let silence_preroll_samples =
                                (sample_rate * SPEAKER_CHANGE_SILENCE_PREROLL_MS / 1000.0) as usize;
                            let new_segment_start = probe
                                .new_speech_start_sample
                                .saturating_sub(silence_preroll_samples)
                                .min(segment.len());
                            segment = segment[new_segment_start..].to_vec();
                            utterance_id = utterance_id.wrapping_add(1);
                            speaker_turn_id = next_speaker_turn_id(speaker_turn_id);
                            current_speaker_boundary_before = !old_segment_flushed;
                            cross_segment_probe = None;
                            previous_natural_signature = None;
                            previous_natural_segment_seen = false;
                            preserve_speaker_turn_across_rolling_boundary = false;
                            speech_elapsed_ms = segment.len() as f32 / sample_rate * 1000.0;
                            voiced_ms =
                                probe.new_speech_samples.len() as f32 / sample_rate * 1000.0;
                            peak_rms = signal_rms(&probe.new_speech_samples);
                            silence_ms = 0.0;
                            strong_silence_ms = 0.0;
                            endpoint_candidate_ms = None;
                            pause_start_sample = None;
                            pre_roll.clear();
                            above_th_frames = 0;
                            next_partial_ms = 550.0;
                            next_checkpoint_ms = 1_800.0;
                            speaker_change_cooldown_ms = SPEAKER_CHANGE_DEBOUNCE_MS;
                            crate::log_line(&format!(
                                "[speaker] forced boundary; next utterance={}, turn={}, retained_new_audio={:.0}ms debounce={}ms",
                                utterance_id,
                                speaker_turn_id,
                                speech_elapsed_ms,
                                SPEAKER_CHANGE_DEBOUNCE_MS as u32
                            ));
                            continue;
                        }
                    }
                }

                // The base model returns in ~0.2s on Apple Silicon. A first
                // preview at 550ms keeps perceived latency below Hen while a
                // single-flight guard in main.rs prevents process pile-ups.
                if speech_elapsed_ms >= next_partial_ms {
                    next_partial_ms = speech_elapsed_ms + 1000.0;
                    if segment.len() >= 9600 {
                        self.flush_chunk(
                            &segment,
                            sample_rate,
                            VoiceChunkMetadata {
                                utterance_id,
                                speaker_turn_id,
                                is_final: false,
                                end_reason: SegmentEndReason::Ongoing,
                                speaker_boundary_before: current_speaker_boundary_before,
                                speaker_turn_end: false,
                            },
                            &on_partial,
                        );
                    }
                }

                // Re-recognize the growing utterance at a moderate cadence.
                // Two consecutive hypotheses allow main.rs to commit only a
                // stable prefix instead of publishing every Whisper rewrite.
                if speech_elapsed_ms >= next_checkpoint_ms {
                    next_checkpoint_ms = speech_elapsed_ms + 1_200.0;
                    if segment.len() >= (sample_rate * 1.2) as usize {
                        self.flush_chunk(
                            &segment,
                            sample_rate,
                            VoiceChunkMetadata {
                                utterance_id,
                                speaker_turn_id,
                                is_final: false,
                                end_reason: SegmentEndReason::Ongoing,
                                speaker_boundary_before: current_speaker_boundary_before,
                                speaker_turn_end: false,
                            },
                            &on_segment,
                        );
                        current_speaker_boundary_before = false;
                    }
                }

                let reached_regular_silence = silence_ms >= end_ms as f32;
                let reached_strong_silence = strong_silence_ms >= strong_end_ms as f32;
                if !steady_mode
                    && endpoint_candidate_ms.is_none()
                    && (reached_regular_silence || reached_strong_silence)
                {
                    endpoint_candidate_ms = Some(if reached_strong_silence {
                        strong_end_ms
                    } else {
                        end_ms
                    });
                    let hint = crate::endpoint_semantic_hint(utterance_id);
                    crate::log_line(&format!(
                        "[endpoint] candidate utterance={} silence={:.0}ms hint={:?}",
                        utterance_id, silence_ms, hint
                    ));
                }

                let semantic_hint = crate::endpoint_semantic_hint(utterance_id);
                let rolling_limit_ms =
                    dynamic_rolling_limit_ms(max_ms, semantic_hint, crate::asr_queue_idle());
                let confirmed_fast_endpoint_ms = endpoint_candidate_ms.map(|candidate_ms| {
                    confirmed_endpoint_ms(
                        candidate_ms,
                        semantic_hint,
                        crate::adaptive_vad_extension_ms(),
                    )
                });
                let ended_by_regular_silence = if steady_mode {
                    reached_regular_silence
                } else {
                    confirmed_fast_endpoint_ms
                        .is_some_and(|confirmed_ms| silence_ms >= confirmed_ms as f32)
                };
                let ended_by_strong_silence = steady_mode && reached_strong_silence;
                // If a natural endpoint is already being confirmed, let its
                // short post-roll finish even when the 5s rolling boundary is
                // reached a few milliseconds earlier. Resumed speech clears
                // the candidate and the hard boundary applies immediately.
                let ended_by_hard_limit = speech_elapsed_ms >= rolling_limit_ms as f32
                    && (steady_mode || endpoint_candidate_ms.is_none());
                if ended_by_regular_silence || ended_by_strong_silence || ended_by_hard_limit {
                    let ended_by_silence = ended_by_regular_silence || ended_by_strong_silence;
                    let endpoint_threshold_ms = if ended_by_strong_silence {
                        strong_end_ms
                    } else if let Some(confirmed_ms) = confirmed_fast_endpoint_ms {
                        confirmed_ms
                    } else {
                        end_ms
                    };
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("/tmp/myt-voice-test.log")
                    {
                        use std::io::Write;
                        let _ = writeln!(
                            f,
                            "[vad] segment closed: speech_ms={:.0} silence_ms={:.0} strong_silence_ms={:.0} threshold_ms={} rolling_limit_ms={} mode={} samples={} reason={}",
                            speech_elapsed_ms,
                            silence_ms,
                            strong_silence_ms,
                            endpoint_threshold_ms,
                            rolling_limit_ms,
                            if steady_mode { "steady" } else { "fast" },
                            segment.len(),
                            if ended_by_strong_silence {
                                "strong-silence"
                            } else if ended_by_regular_silence {
                                if semantic_hint == crate::EndpointSemanticHint::Incomplete {
                                    "semantic-extension"
                                } else {
                                    "confirmed-silence"
                                }
                            } else {
                                "rolling-max"
                            }
                        );
                    }
                    let (start_th, _) = vad_thresholds(noise_floor);
                    if cross_segment_probe.take().is_some() {
                        speaker_turn_id = next_speaker_turn_id(speaker_turn_id);
                        current_speaker_boundary_before = true;
                        crate::log_line(&format!(
                            "[speaker-cross] Uncertain; probe ended before two confirmations, starting turn={} before utterance={}",
                            speaker_turn_id, utterance_id
                        ));
                    }
                    if ended_by_silence {
                        let speech_end = pause_start_sample.unwrap_or(segment.len());
                        previous_natural_signature = tail_voice_signature(
                            &segment,
                            sample_rate,
                            speech_end,
                            CROSS_SEGMENT_PROFILE_MS,
                        );
                        previous_natural_segment_seen = true;
                        preserve_speaker_turn_across_rolling_boundary = false;
                    } else {
                        previous_natural_signature = None;
                        previous_natural_segment_seen = false;
                        preserve_speaker_turn_across_rolling_boundary = true;
                    }
                    if should_keep_segment(voiced_ms, peak_rms, start_th) {
                        self.flush_chunk(
                            &segment,
                            sample_rate,
                            VoiceChunkMetadata {
                                utterance_id,
                                speaker_turn_id,
                                is_final: true,
                                end_reason: if ended_by_strong_silence {
                                    SegmentEndReason::StrongSilence
                                } else if ended_by_regular_silence {
                                    SegmentEndReason::ConfirmedSilence
                                } else {
                                    SegmentEndReason::RollingLimit
                                },
                                speaker_boundary_before: current_speaker_boundary_before,
                                speaker_turn_end: false,
                            },
                            &on_segment,
                        );
                        current_speaker_boundary_before = false;
                    } else {
                        crate::log_line(&format!(
                            "[vad] dropped weak segment: voiced={:.0}ms peak={:.4} start={:.4}",
                            voiced_ms, peak_rms, start_th
                        ));
                    }
                    crate::clear_endpoint_semantic_hint(utterance_id);
                    // At a continuous-speech hard boundary, seed the next
                    // segment with 480ms of overlap so words straddling the
                    // cut are not lost. Stable-prefix logic removes repeats.
                    let overlap = if ended_by_silence {
                        Vec::new()
                    } else {
                        segment[segment.len().saturating_sub(pre_roll_samples)..].to_vec()
                    };
                    segment.clear();
                    in_speech = false;
                    crate::set_voice_activity(false);
                    pre_roll.clear();
                    pre_roll.extend(overlap);
                    above_th_frames = 0;
                    silence_ms = 0.0;
                    strong_silence_ms = 0.0;
                    endpoint_candidate_ms = None;
                    pause_start_sample = None;
                    speaker_turn_probe = None;
                    speech_elapsed_ms = 0.0;
                    voiced_ms = 0.0;
                    peak_rms = 0.0;
                    next_partial_ms = 550.0;
                    // After a rolling cut, produce two medium-model views of
                    // the continuation before the 4.2s semantic carry limit.
                    // Natural utterances keep the lower-load 1.8s cadence.
                    next_checkpoint_ms = if ended_by_silence { 1_800.0 } else { 1_200.0 };
                    neural_native_buffer.clear();
                    neural_probability = None;
                    vad_watchdog.clear_signal();
                    if let Some(vad) = neural_vad.as_mut() {
                        vad.reset();
                    }
                }
            }
        }

        if let Some(stream) = microphone_stream {
            let _ = stream.pause();
        }
        if let Some(mut child) = system_audio_child {
            let _ = child.kill();
            let _ = child.wait();
        }
        crate::set_voice_activity(false);
        if let Some(error) = source_error {
            Err(error)
        } else {
            Ok(())
        }
    }

    /// Write the segment to a WAV and invoke the callback with its path.
    /// Does NOT clear the segment (callers decide).
    fn flush_with<P>(&self, segment: &[f32], sample_rate: f32, cb: &P)
    where
        P: Fn(String),
    {
        if segment.is_empty() {
            return;
        }
        // resample to 16 kHz (linear) and write WAV
        let target = 16000.0f32;
        let ratio = target / sample_rate;
        let out_len = (segment.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f32 / ratio;
            let i0 = src.floor() as usize;
            let i1 = (i0 + 1).min(segment.len() - 1);
            let frac = src - i0 as f32;
            resampled.push(segment[i0] * (1.0 - frac) + segment[i1] * frac);
        }

        // A checkpoint and a final boundary can be emitted inside the same
        // millisecond. A process-local sequence gives every queued owner a
        // distinct path so coalescing one job cannot delete another's WAV.
        let path = next_wav_path();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        if let Ok(mut writer) = hound::WavWriter::create(&path, spec) {
            for s in &resampled {
                let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                let _ = writer.write_sample(v);
            }
            let _ = writer.finalize();
            cb(path.to_string_lossy().into_owned());
        }
    }

    fn flush_chunk<F>(
        &self,
        segment: &[f32],
        sample_rate: f32,
        metadata: VoiceChunkMetadata,
        on_segment: &F,
    ) where
        F: Fn(VoiceChunk),
    {
        self.flush_with(segment, sample_rate, &|wav_path| {
            on_segment(VoiceChunk {
                wav_path,
                utterance_id: metadata.utterance_id,
                speaker_turn_id: metadata.speaker_turn_id,
                is_final: metadata.is_final,
                end_reason: metadata.end_reason,
                speaker_boundary_before: metadata.speaker_boundary_before,
                speaker_turn_end: metadata.speaker_turn_end,
            });
        });
    }
}

fn signal_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

fn voice_signature(samples: &[f32], sample_rate: f32) -> Option<VoiceSignature> {
    if samples.len() < (sample_rate * 0.10) as usize {
        return None;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let centered: Vec<f32> = samples.iter().map(|sample| sample - mean).collect();
    let rms = signal_rms(&centered);
    if rms < 0.0025 {
        return None;
    }

    let zero_crossings = centered
        .windows(2)
        .filter(|pair| (pair[0] >= 0.0) != (pair[1] >= 0.0))
        .count();
    let zero_crossing_rate = zero_crossings as f32 / centered.len().saturating_sub(1).max(1) as f32;

    let target_rate = 8_000.0_f32.min(sample_rate);
    let step = (sample_rate / target_rate).round().max(1.0) as usize;
    let downsampled: Vec<f32> = centered.iter().step_by(step).copied().collect();
    let effective_rate = sample_rate / step as f32;
    let min_lag = (effective_rate / 360.0).floor().max(1.0) as usize;
    let max_lag = (effective_rate / 75.0)
        .ceil()
        .min((downsampled.len() / 2) as f32) as usize;
    let mut best_lag = 0usize;
    let mut best_correlation = 0.0_f32;
    for lag in min_lag..=max_lag.max(min_lag) {
        let mut cross = 0.0_f64;
        let mut left_energy = 0.0_f64;
        let mut right_energy = 0.0_f64;
        for index in lag..downsampled.len() {
            let left = downsampled[index] as f64;
            let right = downsampled[index - lag] as f64;
            cross += left * right;
            left_energy += left * left;
            right_energy += right * right;
        }
        let denominator = (left_energy * right_energy).sqrt();
        let correlation = if denominator > 1e-12 {
            (cross / denominator) as f32
        } else {
            0.0
        };
        if correlation > best_correlation {
            best_correlation = correlation;
            best_lag = lag;
        }
    }
    let pitch_hz = if best_lag > 0 {
        effective_rate / best_lag as f32
    } else {
        0.0
    };

    let band_frequencies = [
        125.0, 250.0, 500.0, 750.0, 1_000.0, 1_500.0, 2_200.0, 3_200.0,
    ];
    let mut band_powers = [0.0_f32; 8];
    for (index, frequency) in band_frequencies.iter().enumerate() {
        let omega = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        let coefficient = 2.0 * omega.cos();
        let mut previous = 0.0_f32;
        let mut previous_two = 0.0_f32;
        for sample in &centered {
            let current = *sample + coefficient * previous - previous_two;
            previous_two = previous;
            previous = current;
        }
        band_powers[index] = (previous_two * previous_two + previous * previous
            - coefficient * previous * previous_two)
            .max(0.0);
    }
    let total_band_power = band_powers.iter().sum::<f32>().max(1e-12);
    for power in &mut band_powers {
        *power /= total_band_power;
    }

    Some(VoiceSignature {
        pitch_hz,
        pitch_confidence: best_correlation.clamp(0.0, 1.0),
        zero_crossing_rate,
        log_rms: rms.max(1e-6).ln(),
        spectral_shape: band_powers,
    })
}

fn speaker_change_threshold(pause_ms: f32) -> f32 {
    if pause_ms < 200.0 {
        0.55
    } else if pause_ms < 400.0 {
        0.42
    } else {
        0.34
    }
}

fn speaker_change_confirmed(
    old: VoiceSignature,
    first: VoiceSignature,
    second: VoiceSignature,
    pause_ms: f32,
) -> (bool, f32, usize, f32) {
    let threshold = speaker_change_threshold(pause_ms);
    let first_result = speaker_signature_changed(old, first, threshold);
    let second_result = speaker_signature_changed(old, second, threshold);
    let confirmations = usize::from(first_result.0) + usize::from(second_result.0);
    (
        first_result.0 && second_result.0,
        first_result.1.min(second_result.1),
        confirmations,
        threshold,
    )
}

fn cross_segment_probe_samples_needed(sample_rate: f32) -> usize {
    let window = (sample_rate * SPEAKER_CHANGE_PROBE_WINDOW_MS / 1000.0) as usize;
    let step = (sample_rate * SPEAKER_CHANGE_CONFIRMATION_STEP_MS / 1000.0) as usize;
    window + step
}

fn compare_cross_segment_speaker(
    probe: &CrossSegmentSpeakerProbe,
    sample_rate: f32,
) -> CrossSegmentSpeakerComparison {
    let window = (sample_rate * SPEAKER_CHANGE_PROBE_WINDOW_MS / 1000.0) as usize;
    let step = (sample_rate * SPEAKER_CHANGE_CONFIRMATION_STEP_MS / 1000.0) as usize;
    if probe.new_speech_samples.len() < window + step {
        return CrossSegmentSpeakerComparison {
            decision: CrossSegmentSpeakerDecision::Uncertain,
            score: 1.0,
            confirmations: 0,
        };
    }
    let first = voice_signature(&probe.new_speech_samples[..window], sample_rate);
    let second = voice_signature(&probe.new_speech_samples[step..step + window], sample_rate);
    let (Some(first), Some(second)) = (first, second) else {
        return CrossSegmentSpeakerComparison {
            decision: CrossSegmentSpeakerDecision::Uncertain,
            score: 1.0,
            confirmations: 0,
        };
    };

    let change_threshold = speaker_change_threshold(800.0);
    let first_change = speaker_signature_changed(probe.old_signature, first, change_threshold);
    let second_change = speaker_signature_changed(probe.old_signature, second, change_threshold);
    let change_confirmations = usize::from(first_change.0) + usize::from(second_change.0);
    if change_confirmations == 2 {
        return CrossSegmentSpeakerComparison {
            decision: CrossSegmentSpeakerDecision::Different,
            score: first_change.1.min(second_change.1),
            confirmations: 2,
        };
    }

    let (_, window_stability_score) = speaker_signature_changed(first, second, change_threshold);
    let same_confirmations = usize::from(first_change.1 <= CROSS_SEGMENT_SAME_MAX_SCORE)
        + usize::from(second_change.1 <= CROSS_SEGMENT_SAME_MAX_SCORE);
    let signatures_reliable = probe.old_signature.pitch_confidence >= 0.30
        && first.pitch_confidence >= 0.30
        && second.pitch_confidence >= 0.30;
    if signatures_reliable
        && same_confirmations == 2
        && window_stability_score <= CROSS_SEGMENT_WINDOW_STABILITY_MAX_SCORE
    {
        CrossSegmentSpeakerComparison {
            decision: CrossSegmentSpeakerDecision::Same,
            score: first_change.1.max(second_change.1),
            confirmations: 2,
        }
    } else {
        CrossSegmentSpeakerComparison {
            decision: CrossSegmentSpeakerDecision::Uncertain,
            score: first_change.1.max(second_change.1),
            confirmations: same_confirmations,
        }
    }
}

fn tail_voice_signature(
    samples: &[f32],
    sample_rate: f32,
    speech_end: usize,
    profile_ms: f32,
) -> Option<VoiceSignature> {
    let speech_end = speech_end.min(samples.len());
    let profile_samples = (sample_rate * profile_ms / 1000.0) as usize;
    let start = speech_end.saturating_sub(profile_samples);
    voice_signature(&samples[start..speech_end], sample_rate)
}

fn next_speaker_turn_id(current: u64) -> u64 {
    current.wrapping_add(1).max(1)
}

fn speaker_signature_changed(
    old: VoiceSignature,
    new: VoiceSignature,
    score_threshold: f32,
) -> (bool, f32) {
    let pitch_distance = if old.pitch_confidence >= 0.34
        && new.pitch_confidence >= 0.34
        && old.pitch_hz > 0.0
        && new.pitch_hz > 0.0
    {
        (old.pitch_hz / new.pitch_hz).log2().abs().min(1.0)
    } else {
        0.0
    };
    let spectral_distance = old
        .spectral_shape
        .iter()
        .zip(new.spectral_shape.iter())
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        .mul_add(0.5, 0.0)
        .min(1.0);
    let zcr_distance = ((old.zero_crossing_rate - new.zero_crossing_rate).abs() / 0.18).min(1.0);
    let energy_distance = ((old.log_rms - new.log_rms).abs() / 1.4).min(1.0);
    let score = 0.44 * pitch_distance
        + 0.36 * spectral_distance
        + 0.12 * zcr_distance
        + 0.08 * energy_distance;
    let changed = score >= score_threshold
        && (pitch_distance >= 0.28 || spectral_distance >= 0.30)
        && (spectral_distance >= 0.18 || zcr_distance >= 0.25);
    (changed, score)
}

fn append_f32le_samples(pending: &mut Vec<u8>, input: &[u8]) -> Vec<f32> {
    pending.extend_from_slice(input);
    let complete_bytes = pending.len() / 4 * 4;
    let (complete, _) = pending[..complete_bytes].as_chunks::<4>();
    let samples = complete
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    pending.drain(..complete_bytes);
    samples
}

fn resample_fixed(samples: &[f32], output_len: usize) -> Vec<f32> {
    if samples.is_empty() || output_len == 0 {
        return Vec::new();
    }
    if samples.len() == 1 {
        return vec![samples[0]; output_len];
    }
    let scale = (samples.len() - 1) as f32 / (output_len - 1).max(1) as f32;
    (0..output_len)
        .map(|index| {
            let source = index as f32 * scale;
            let left = source.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = source - left as f32;
            samples[left] * (1.0 - fraction) + samples[right] * fraction
        })
        .collect()
}

fn is_speech_frame(
    neural_probability: Option<f32>,
    rms: f32,
    energy_threshold: f32,
    already_speaking: bool,
) -> bool {
    match neural_probability {
        Some(probability) => {
            let probability_threshold = if already_speaking { 0.35 } else { 0.52 };
            probability >= probability_threshold && rms >= 0.003
        }
        None => rms >= energy_threshold,
    }
}

fn watchdog_strong_threshold(start_threshold: f32) -> f32 {
    (start_threshold * 2.5).max(0.035)
}

fn is_speech_frame_with_watchdog(
    neural_probability: Option<f32>,
    rms: f32,
    energy_threshold: f32,
    already_speaking: bool,
    energy_fallback: bool,
) -> bool {
    if energy_fallback {
        let threshold = if already_speaking {
            energy_threshold.max(0.010)
        } else {
            watchdog_strong_threshold(energy_threshold)
        };
        rms >= threshold
    } else {
        is_speech_frame(neural_probability, rms, energy_threshold, already_speaking)
    }
}

fn vad_thresholds(noise_floor: f32) -> (f32, f32) {
    let start = (noise_floor * 1.8).clamp(0.012, 0.055);
    let end = (noise_floor * 1.3).clamp(0.008, 0.035).min(start * 0.82);
    (start, end)
}

fn should_keep_segment(voiced_ms: f32, peak_rms: f32, start_threshold: f32) -> bool {
    voiced_ms >= 320.0 || (voiced_ms >= 120.0 && peak_rms >= start_threshold * 1.25)
}

fn endpoint_policy(steady: bool, voiced_ms: f32, elapsed_ms: f32) -> (u32, u32) {
    if steady {
        (800, 5_000)
    } else if voiced_ms <= 2_000.0 && elapsed_ms <= 3_000.0 {
        (320, 5_000)
    } else {
        (420, 5_000)
    }
}

fn dynamic_rolling_limit_ms(
    base_limit_ms: u32,
    semantic_hint: crate::EndpointSemanticHint,
    asr_queue_idle: bool,
) -> u32 {
    if semantic_hint == crate::EndpointSemanticHint::Incomplete && asr_queue_idle {
        ROLLING_INCOMPLETE_IDLE_MAX_MS.max(base_limit_ms)
    } else {
        base_limit_ms
    }
}

fn strong_silence_endpoint_ms(steady: bool, regular_endpoint_ms: u32) -> u32 {
    if steady {
        480
    } else {
        regular_endpoint_ms
    }
}

fn confirmed_endpoint_ms(
    candidate_ms: u32,
    hint: crate::EndpointSemanticHint,
    adaptive_extension_ms: u32,
) -> u32 {
    if hint == crate::EndpointSemanticHint::Incomplete {
        return ENDPOINT_INCOMPLETE_MAX_SILENCE_MS;
    }
    candidate_ms
        .saturating_add(ENDPOINT_POST_ROLL_MS)
        .saturating_add(adaptive_extension_ms.min(160) / 2)
}

#[cfg(test)]
mod tests {
    use super::{
        append_f32le_samples, compare_cross_segment_speaker, confirmed_endpoint_ms,
        dynamic_rolling_limit_ms, endpoint_policy, is_speech_frame, is_speech_frame_with_watchdog,
        next_wav_path, resample_fixed, should_keep_segment, speaker_change_confirmed,
        speaker_change_threshold, speaker_signature_changed, strong_silence_endpoint_ms,
        vad_thresholds, voice_signature, CrossSegmentSpeakerDecision, CrossSegmentSpeakerProbe,
        SystemAudioSignalWatchdog, VadWatchdog,
    };

    #[test]
    fn queued_wav_paths_are_unique_even_within_one_millisecond() {
        let first = next_wav_path();
        let second = next_wav_path();
        assert_ne!(first, second);
        assert!(first
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with(&format!("myt-seg-{}-", std::process::id())));
    }

    #[test]
    fn system_audio_decoder_preserves_partial_float_bytes() {
        let first = 0.25_f32.to_le_bytes();
        let second = (-0.5_f32).to_le_bytes();
        let mut pending = Vec::new();
        assert!(append_f32le_samples(&mut pending, &first[..3]).is_empty());
        let mut remainder = Vec::from(&first[3..]);
        remainder.extend_from_slice(&second);
        assert_eq!(
            append_f32le_samples(&mut pending, &remainder),
            vec![0.25, -0.5]
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn endpoint_is_aggressive_only_for_short_conversational_replies() {
        assert_eq!(endpoint_policy(false, 1_800.0, 2_100.0), (320, 5_000));
        assert_eq!(endpoint_policy(false, 2_100.0, 2_400.0), (420, 5_000));
        assert_eq!(endpoint_policy(false, 1_800.0, 3_100.0), (420, 5_000));
        assert_eq!(endpoint_policy(true, 400.0, 600.0), (800, 5_000));
        assert_eq!(strong_silence_endpoint_ms(true, 800), 480);
        assert_eq!(strong_silence_endpoint_ms(false, 420), 420);
    }

    #[test]
    fn rolling_boundary_extends_only_for_incomplete_idle_speech() {
        assert_eq!(
            dynamic_rolling_limit_ms(5_000, crate::EndpointSemanticHint::Incomplete, true),
            7_000
        );
        assert_eq!(
            dynamic_rolling_limit_ms(5_000, crate::EndpointSemanticHint::Incomplete, false),
            5_000
        );
        assert_eq!(
            dynamic_rolling_limit_ms(5_000, crate::EndpointSemanticHint::Complete, true),
            5_000
        );
        assert_eq!(
            dynamic_rolling_limit_ms(5_000, crate::EndpointSemanticHint::Unknown, true),
            5_000
        );
    }

    #[test]
    fn fast_endpoint_adds_post_roll_only_when_the_clause_is_not_incomplete() {
        assert_eq!(
            confirmed_endpoint_ms(320, crate::EndpointSemanticHint::Unknown, 0),
            440
        );
        assert_eq!(
            confirmed_endpoint_ms(420, crate::EndpointSemanticHint::Complete, 80),
            580
        );
        assert_eq!(
            confirmed_endpoint_ms(320, crate::EndpointSemanticHint::Incomplete, 0),
            900
        );
    }

    #[test]
    fn system_audio_zero_signal_watchdog_falls_back_only_before_first_signal() {
        let mut silent = SystemAudioSignalWatchdog::default();
        assert!(!silent.observe(0.0, 5_900.0));
        assert!(silent.observe(0.0, 100.0));

        let mut recovered = SystemAudioSignalWatchdog::default();
        assert!(!recovered.observe(0.0, 3_000.0));
        assert!(!recovered.observe(0.000_01, 32.0));
        assert!(!recovered.observe(0.0, 30_000.0));
    }

    #[test]
    fn speaker_signature_separates_distinct_voices_without_reacting_to_volume() {
        fn voice_fixture(sample_rate: f32, pitch: f32, amplitude: f32) -> Vec<f32> {
            let samples = (sample_rate * 0.36) as usize;
            (0..samples)
                .map(|index| {
                    let time = index as f32 / sample_rate;
                    amplitude
                        * ((2.0 * std::f32::consts::PI * pitch * time).sin()
                            + 0.35 * (2.0 * std::f32::consts::PI * pitch * 2.0 * time).sin()
                            + 0.18 * (2.0 * std::f32::consts::PI * pitch * 3.0 * time).sin())
                })
                .collect()
        }

        let sample_rate = 16_000.0;
        let first = voice_signature(&voice_fixture(sample_rate, 120.0, 0.08), sample_rate).unwrap();
        let same_louder =
            voice_signature(&voice_fixture(sample_rate, 120.0, 0.16), sample_rate).unwrap();
        let other = voice_signature(&voice_fixture(sample_rate, 220.0, 0.08), sample_rate).unwrap();
        assert!(!speaker_signature_changed(first, same_louder, 0.34).0);
        let (changed, score) = speaker_signature_changed(first, other, 0.34);
        assert!(changed, "distinct synthetic voices scored only {score:.3}");

        let (confirmed, _, confirmations, _) =
            speaker_change_confirmed(first, other, same_louder, 450.0);
        assert!(!confirmed);
        assert_eq!(confirmations, 1);

        let (confirmed, _, confirmations, _) = speaker_change_confirmed(first, other, other, 450.0);
        assert!(confirmed);
        assert_eq!(confirmations, 2);
    }

    #[test]
    fn speaker_change_threshold_tightens_short_pauses() {
        assert_eq!(speaker_change_threshold(120.0), 0.55);
        assert_eq!(speaker_change_threshold(199.0), 0.55);
        assert_eq!(speaker_change_threshold(200.0), 0.42);
        assert_eq!(speaker_change_threshold(399.0), 0.42);
        assert_eq!(speaker_change_threshold(400.0), 0.34);
        assert_eq!(speaker_change_threshold(800.0), 0.34);
    }

    #[test]
    fn cross_segment_tracking_confirms_same_different_and_uncertain_voices() {
        fn voice_fixture(sample_rate: f32, pitch: f32, amplitude: f32, seconds: f32) -> Vec<f32> {
            let samples = (sample_rate * seconds) as usize;
            (0..samples)
                .map(|index| {
                    let time = index as f32 / sample_rate;
                    amplitude
                        * ((2.0 * std::f32::consts::PI * pitch * time).sin()
                            + 0.35 * (2.0 * std::f32::consts::PI * pitch * 2.0 * time).sin()
                            + 0.18 * (2.0 * std::f32::consts::PI * pitch * 3.0 * time).sin())
                })
                .collect()
        }

        let sample_rate = 16_000.0;
        let old =
            voice_signature(&voice_fixture(sample_rate, 120.0, 0.08, 0.30), sample_rate).unwrap();
        let same = CrossSegmentSpeakerProbe {
            old_signature: old,
            new_speech_samples: voice_fixture(sample_rate, 120.0, 0.13, 0.24),
        };
        assert_eq!(
            compare_cross_segment_speaker(&same, sample_rate).decision,
            CrossSegmentSpeakerDecision::Same
        );

        let different = CrossSegmentSpeakerProbe {
            old_signature: old,
            new_speech_samples: voice_fixture(sample_rate, 220.0, 0.08, 0.24),
        };
        assert_eq!(
            compare_cross_segment_speaker(&different, sample_rate).decision,
            CrossSegmentSpeakerDecision::Different
        );

        let uncertain = CrossSegmentSpeakerProbe {
            old_signature: old,
            new_speech_samples: vec![0.0; (sample_rate * 0.24) as usize],
        };
        assert_eq!(
            compare_cross_segment_speaker(&uncertain, sample_rate).decision,
            CrossSegmentSpeakerDecision::Uncertain
        );
    }

    #[test]
    fn vad_has_real_hysteresis_across_noise_levels() {
        for floor in [0.002, 0.006, 0.012, 0.025, 0.05] {
            let (start, end) = vad_thresholds(floor);
            assert!(start > end, "start={start}, end={end}, floor={floor}");
        }
    }

    #[test]
    fn short_loud_replies_survive_but_weak_clicks_do_not() {
        assert!(should_keep_segment(150.0, 0.030, 0.018));
        assert!(!should_keep_segment(150.0, 0.019, 0.018));
        assert!(should_keep_segment(400.0, 0.019, 0.018));
    }

    #[test]
    fn neural_vad_rejects_loud_non_speech_and_has_hysteresis() {
        assert!(!is_speech_frame(Some(0.10), 0.20, 0.02, false));
        assert!(is_speech_frame(Some(0.70), 0.01, 0.02, false));
        assert!(is_speech_frame(Some(0.40), 0.01, 0.02, true));
        assert!(!is_speech_frame(Some(0.40), 0.01, 0.02, false));
    }

    #[test]
    fn vad_resampler_produces_one_silero_frame() {
        let input: Vec<f32> = (0..1536).map(|value| value as f32).collect();
        let output = resample_fixed(&input, 512);
        assert_eq!(output.len(), 512);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[511], 1535.0);
    }

    #[test]
    fn watchdog_requires_sustained_strong_audio_before_fallback() {
        let mut watchdog = VadWatchdog::default();
        for _ in 0..11 {
            let update = watchdog.observe(Some(0.0), 0.08, 0.02, 32.0);
            assert!(!update.activated_fallback);
            assert!(!watchdog.energy_fallback);
        }
        let update = watchdog.observe(Some(0.0), 0.08, 0.02, 32.0);
        assert!(update.activated_fallback);
        assert!(watchdog.energy_fallback);
        assert!(is_speech_frame_with_watchdog(
            Some(0.0),
            0.08,
            0.02,
            false,
            watchdog.energy_fallback
        ));
    }

    #[test]
    fn watchdog_rejects_clicks_and_quiet_false_positives() {
        let mut watchdog = VadWatchdog::default();
        watchdog.observe(Some(0.0), 0.20, 0.02, 32.0);
        for _ in 0..30 {
            watchdog.observe(Some(0.0), 0.025, 0.02, 32.0);
        }
        assert!(!watchdog.energy_fallback);
        assert!(!is_speech_frame_with_watchdog(
            Some(0.0),
            0.025,
            0.02,
            false,
            false
        ));
    }

    #[test]
    fn watchdog_rebuilds_after_four_seconds_and_has_cooldown() {
        let mut watchdog = VadWatchdog::default();
        let mut reloads = 0;
        for _ in 0..125 {
            if watchdog
                .observe(Some(0.0), 0.08, 0.02, 32.0)
                .reload_neural_vad
            {
                reloads += 1;
            }
        }
        assert_eq!(reloads, 1);
        for _ in 0..125 {
            if watchdog
                .observe(Some(0.0), 0.08, 0.02, 32.0)
                .reload_neural_vad
            {
                reloads += 1;
            }
        }
        assert_eq!(reloads, 1);
    }
}
