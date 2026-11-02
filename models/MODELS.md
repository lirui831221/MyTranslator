# MyTranslator managed models

All runtime model files in this directory are regular files owned by the
MyTranslator project. External symlinks are intentionally rejected by
`tools/verify_models.sh` so development builds and release builds do not
depend on another application being installed.

| File | Upstream source | Revision / note | Bytes |
|---|---|---|---:|
| `Hy-MT2-1.8B-Q4_K_M.gguf` | `tencent/Hy-MT2-1.8B-GGUF` | `a0c709d9fac510f2c807aa3af52872340dc37a4a` | 1,133,080,448 |
| `Index-Translate-2B.Q4_K_M.gguf` | `IndexTeam/Index-Translate-2B-GGUF` | `449c9e6457b3632d328c6cbb78ae8e8e0c8059a5` | 1,312,164,352 |
| `ggml-base-q5_1.bin` | whisper.cpp model | streaming preview | 59,707,625 |
| `ggml-medium-q5_0.bin` | whisper.cpp model | final recognition | 539,212,467 |
| `ggml-large-v3-turbo-q5_0.bin` | `ggerganov/whisper.cpp` | selective high-accuracy review | 574,041,195 |
| `ggml-silero-v6.2.0.bin` | whisper.cpp Silero VAD | VAD v6.2 | 885,098 |
| `qwen3-asr-1.7b/` | `mlx-community/Qwen3-ASR-1.7B-8bit` | revision `a8379a2e2f9e313c9292cdf1af4055ab56d50d55`; optional background review | 2,467,857,511 |
| `qwen3.5-2b-mlx-4bit/` | `mlx-community/Qwen3.5-2B-MLX-4bit` | conservative, selective background translation review | 1,749,081,964 |

`SHA256SUMS` is the release integrity authority. Run:

```bash
./tools/verify_models.sh
```

The two translation model files were migrated on 2026-10-23 from the locally
installed Hen offline package after their hashes were matched against Hen's
upstream manifest. They are now independent copies; deleting or upgrading Hen
does not change these files.

The Qwen3-ASR directory was migrated on 2026-10-25 after its 2,463,307,541-byte
`model.safetensors` file matched the SHA-256 published by the Hugging Face API
for `mlx-community/Qwen3-ASR-1.7B-8bit`. MyTranslator runs it through its own
bundled Apache-2.0 MLX worker and never reads Hen's application directory at
runtime. Whisper Medium remains the foreground recognizer; Qwen is lazy-loaded
only for gated background review and exits after an idle timeout.

The Qwen3.5 directory was migrated on 2026-10-25 as an independent managed
copy of the Apache-2.0 MLX 4-bit conversion. Its 1,722,271,785-byte weight file
has SHA-256 `713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5`.
It never replaces Hy-MT2 on the foreground path: MyTranslator invokes it only
for gated, asynchronous review and unloads the worker after 75 idle seconds.

