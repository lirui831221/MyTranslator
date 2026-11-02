# FLEURS attribution

The `fleurs200` benchmark is a deterministic subset of the Google FLEURS
test split. It contains 100 Mandarin Chinese (`cmn_hans_cn`) recordings and
100 US English (`en_us`) recordings selected from shared sentence IDs.

- Dataset: `google/fleurs`
- Authors: Google Research
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Dataset card: <https://huggingface.co/datasets/google/fleurs>
- Paper: <https://arxiv.org/abs/2205.12446>
- Changes: only a subset is extracted; files are not modified.

The generated JSONL records the original config, split, sentence ID, and
SHA-256 for every selected recording so the benchmark can be reproduced and
audited.
