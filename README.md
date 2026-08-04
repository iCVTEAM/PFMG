# PFMG
Code implement of paper "Towards Unified Co-Speech Gesture Generation via Hierarchical Implicit Periodicity Learning"

This repository contains the PyTorch implementation of PFMG, an audio-driven co-speech gesture generation method built on BEAT-style body, expression, audio, text, emotion, speaker, and PAE motion features.

The public release focuses on the PFMG-PAE generation path. Given speech-related conditions and PAE features, the model generates upper-body gesture motion and writes BVH-compatible result files for visualization.

## Contents

- `train.py`, `test.py`: common training and motion-generation entry points.
- `audio2face_trainer.py`: trains the Audio2Face auxiliary model used to produce `face.bin`.
- `models/`: PFMG-PAE, PFMG-TCN, audio-to-face, and PAE-related model code.
- `dataloaders/`: BEAT-style LMDB dataloader, vocabulary builder, and motion utilities.
- `configs/`: release configs for Audio2Face, PFMG-PAE, and PAE feature preparation.
- `scripts/prepare_beat.py`: builds statistics, vocabulary, and LMDB caches from prepared BEAT feature folders.

## Installation

We recommend Python `==3.8`. The release smoke tests were run with Python 3.8, CUDA, PyTorch 1.12, and `pyarrow==4.0.0`.

```shell
conda create -n pfmg python=3.8
conda activate pfmg
pip install -r requirements-legacy.txt
```

If you use local HuggingFace Wav2Vec2 directories, set:

```shell
export PFMG_WAV2VEC2_MODEL=/path/to/wav2vec2-large-xlsr-53-english
export PFMG_WAV2VEC2_EMOTION_MODEL=/path/to/wav2vec-english-speech-emotion-recognition
```

## Data Preparation

This release does not redistribute BEAT data or generated caches. After obtaining BEAT data and converting it into the feature folders described in `docs/DATA.md`, run:

```shell
python scripts/prepare_beat.py \
  --cache-root ./data/beat_cache/beat_4english_15_141
```

The expected feature folders are:

```text
train/val/test
|-- bvh_rot
|-- wave16k
|-- facial52
|-- pae
|-- text
|-- emo
`-- sem
```

## Training and Motion Generation

Commands below assume they are run from this repository root.

### Train Audio2Face

PFMG-TCN loads `face.bin` to generate facial conditions before body-motion generation. To train this auxiliary model from the prepared BEAT cache, run:

```shell
python train.py \
  -c configs/audio2face_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled
```

The best validation checkpoint is saved as:

```text
outputs/audio2pose/custom/<audio2face_exp_name>/rec_val.bin
```

Rename or copy this checkpoint to:

```text
data/beat_cache/beat_4english_15_141/weights/face.bin
```

### Train PFMG-PAE

PFMG-PAE expects the auxiliary `face.bin` and `pfmg_tcn.bin` checkpoints under `data/beat_cache/beat_4english_15_141/weights/`. Then run:

```shell
python train.py \
  -c configs/pfmg_tcn_pae_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled
```

### Generate motion with pretrained PFMG-PAE

```shell
python test.py \
  -c configs/pfmg_tcn_pae_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled \
  --gpus 0
```

The script writes raw generated motion to `result_raw_*.bvh` files and then converts them to `res_*.bvh` files under:

```text
outputs/audio2pose/custom/<exp_name>/9999/
```

These generated BVH files can be loaded into Blender with the same visualization flow used by BEAT.

## Acknowledgements

This codebase builds on BEAT / PantoMatrix audio-to-gesture code and the BEAT 2022 CaMN baseline. We thank the authors of BEAT, CaMN, HuggingFace Transformers, timm, pymo, and related open-source projects. See `NOTICE.md` for details.

## License

This release is for non-commercial research use. See `LICENSE.md` and `NOTICE.md` for license and third-party notices.
