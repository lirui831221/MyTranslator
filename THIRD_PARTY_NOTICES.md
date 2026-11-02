# Third-party notices

MyTranslator includes an optional offline Qwen3-ASR final-recognition path.

- `mlx-qwen3-asr` 0.4.4, copyright its contributors, Apache License 2.0:
  <https://github.com/moona3k/mlx-qwen3-asr>
- Apple MLX and `mlx-metal`, copyright Apple Inc., MIT License:
  <https://github.com/ml-explore/mlx>
- `mlx-vlm`, copyright its contributors, MIT License:
  <https://github.com/Blaizzy/mlx-vlm>
- Qwen3.5-2B, copyright Alibaba Cloud, Apache License 2.0:
  <https://huggingface.co/Qwen/Qwen3.5-2B>
- Qwen3-ASR-1.7B and the MLX 8-bit conversion, Apache License 2.0:
  <https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit>

The corresponding license and package metadata are retained inside the
bundled runtime and model directories. Hy-MT2 is the foreground translation
model; Qwen3.5 performs selective asynchronous review. Whisper remains active
for rolling previews and automatic fallback, while Qwen3-ASR handles final
utterance recognition in the balanced and quality profiles. MLX inference is
serialized when Qwen3-ASR and Qwen3.5 are both selected.
