# ReAlign-VLA

<p align="center">
  <img src="realign_overview.png" alt="FORGE overview" width="100%">
</p>

**Realign perception, not policy.** ReAlign-VLA is a lightweight post-encoder module for improving visual robustness in frozen vision-language-action policies. It predicts a token-adaptive low-rank residual for corrupted third-person visual features while leaving the visual encoder, wrist stream, and downstream VLA unchanged.

## Abstract

Vision-language-action (VLA) policies can fail sharply under visual corruption despite strong nominal performance. ReAlign-VLA addresses these failures by restoring corrupted policy-facing visual representations rather than relearning the policy. Given corrupted agent-view tokens, ReAlign predicts token-specific coefficients and frame-conditioned low-rank correction directions, then adds the resulting residual before the frozen visual projector. The module is trained from paired clean and synthetically corrupted source-domain observations using clean visual features as supervision, without action labels, policy gradients, or test-time adaptation. In our experiments with OpenVLA-OFT and LIBERO-Plus, ReAlign substantially improves robustness under held-out visual corruption while preserving clean-task performance and transferring from LIBERO-Spatial to LIBERO-Object without Object-specific adaptation.

## Installation

### Core package

Use Python 3.10. The core package is sufficient for cached-feature training, synthetic smoke tests, and unit tests.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# Optional: tests
python -m pip install -r requirements-dev.txt
python scripts/smoke_test.py
python -m pytest -q
```

For CUDA training, install the PyTorch build appropriate for your system before installing the package if needed.

### OpenVLA-OFT / LIBERO integration

Feature extraction and closed-loop evaluation require OpenVLA-OFT and either LIBERO or LIBERO-Plus. The versions used in our experiments are:

| Component | Revision |
|---|---|
| OpenVLA-OFT | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| OFT Transformers fork | `bc339d9ad707454c0c115970db43c260067c61ab` |
| LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus | `4976dc30028e805ff8094b55501d532c48fec182` |

Install the optional policy/simulator dependencies:

```bash
python -m pip install -r requirements-oft.txt

mkdir -p upstreams

git clone https://github.com/moojink/openvla-oft.git upstreams/openvla-oft
git -C upstreams/openvla-oft checkout e4287e94541f459edc4feabc4e181f537cd569a8
python -m pip install --no-deps -e upstreams/openvla-oft
```

Install **one** LIBERO backend in the same environment because LIBERO and LIBERO-Plus expose the same `libero` Python package. For the controlled corruption experiments, use the pinned LIBERO-Plus checkout and set the matching path in the YAML config. Model weights and simulator assets are obtained separately from their upstream releases.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream repositories and licensing notes.

## Training

ReAlign is trained in two stages: first extract frozen clean/corrupted visual features, then optimize only the ReAlign adapter on the cached feature pairs.

### 1. Extract clean/corrupted features

The default training config uses successful LIBERO-Spatial trajectories from initialization states 1--8 for training and 9--10 for validation. Agent-view frames are sampled every four environment steps, and three synthetic corruptions are generated per clean frame.

```bash
python scripts/extract_features.py \
  --config configs/train_realign.yaml \
  --output data/feature_cache \
  --device cuda
```

The training corruption bank is implemented independently with OpenCV/NumPy and contains:

- Gaussian blur
- motion blur
- zoom blur
- glass blur
- Gaussian noise

LIBERO-Plus perturbation operators are **not** used for training. Fog, camera-viewpoint perturbations, and lighting perturbations are also excluded from the training bank.

### 2. Train ReAlign

```bash
python scripts/train_realign.py \
  --config configs/train_realign.yaml \
  --features data/feature_cache \
  --output runs/realign \
  --device cuda
```

The released defaults use the final rank-16 NoMixer architecture with 2.28M trainable parameters. Training uses AdamW for 15K steps with learning rate `5e-4`, weight decay `0.01`, 200 warmup steps, cosine decay, gradient clipping at `1.0`, and batches containing 16 corrupted pairs plus 3 clean identity pairs. 

Only ReAlign parameters are optimized. The VLA and visual encoders remain frozen. `best.pt` is selected by held-out validation token cosine similarity, while `last.pt` can be used to resume training:

```bash
python scripts/train_realign.py \
  --config configs/train_realign.yaml \
  --features data/feature_cache \
  --output runs/realign \
  --device cuda \
  --resume runs/realign/last.pt
```

## Evaluation

`scripts/eval_realign.py` evaluates either the frozen OpenVLA-OFT policy or the same frozen policy with a trained ReAlign adapter.

```bash
# Frozen OpenVLA-OFT
python scripts/eval_realign.py \
  --config configs/eval_libero.yaml \
  --method frozen \
  --output runs/eval_frozen \
  --device cuda

# OpenVLA-OFT + ReAlign
python scripts/eval_realign.py \
  --config configs/eval_libero.yaml \
  --method realign \
  --adapter runs/realign/best.pt \
  --output runs/eval_realign \
  --device cuda
```

The paper setting uses dual RGB inputs, proprioception, center cropping, eight-action chunks, and an agent-view-only ReAlign intervention. The wrist tokens bypass ReAlign unchanged. For the controlled LIBERO-Plus experiments, `use_clean_instruction: true` removes perturbation-specific language suffixes so the evaluation isolates visual shift. This controlled protocol is distinct from the official LIBERO-Plus leaderboard protocol.

The bundled evaluation config is a small integration example rather than the exact paper benchmark manifest. A reviewed manifest can be supplied with `--manifest`; Hard and corruption-family results are derived from the corresponding Noise episodes rather than requiring additional rollouts. The same Spatial-trained ReAlign adapter can be evaluated on LIBERO-Object without Object-specific adapter training.

Detailed hyperparameters, seeds, precision settings, and software versions are listed in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
