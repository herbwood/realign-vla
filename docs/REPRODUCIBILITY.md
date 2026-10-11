# Reproducibility

## Final ReAlign configuration

| Item | Setting |
|---|---|
| Backbone | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` |
| Backbone revision | `638918f3d1c2e43a39a8a20772bdb8b91835e4b7` |
| Adapter | NoMixer token-adaptive low-rank residual |
| Rank / hidden width | 16 / 256 |
| Feature dimension | 2176 |
| Agent / wrist tokens | 256 / 256 |
| Trainable parameters | 2,279,968 |
| Adapter init / training seed | 0 |
| Evaluation seed | 7 |
| Environment seed | 0 |
| Training init states | 1--8 |
| Validation init states | 9--10 |
| Sample interval | every 4 environment steps |
| Corruptions per clean frame | 3 |
| Batch | 16 corrupted + 3 clean identity pairs |
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Weight decay | `0.01` |
| Training steps / warmup | 15,000 / 200 |
| Gradient clipping | 1.0 |
| Reconstruction weight | 1.0 |
| Residual regularization | `1e-4` |
| Action chunk | 8 |
| Settling steps | 10 |
| Spatial / Object horizon | 220 / 280 |
| Feature cache / backbone / adapter precision | FP16 / BF16 / FP32 |

## Software

The core package is tested with Python 3.10, PyTorch 2.2.0, NumPy 1.26.4, OpenCV 4.10.0.84, and PyYAML 6.0.3. Policy integration additionally uses torchvision 0.17.0, the OpenVLA-OFT Transformers fork, timm 0.9.10, TensorFlow 2.15.0, robosuite 1.4.1, and MuJoCo 2.3.7. See `requirements-oft.txt` for the complete optional integration stack.

## Evaluation convention

The controlled paper evaluation uses the same task variant, initial state, instruction, and fixed seed across paired methods. ReAlign modifies only the 256 agent-view visual tokens before the projector; the 256 wrist tokens are preserved exactly. The controlled LIBERO-Plus setting uses clean base-task instructions rather than perturbation suffixes, so its results should not be conflated with the official LIBERO-Plus leaderboard protocol.

## Runtime notes

Cached-feature adapter optimization and feature extraction are separate stages. Feature extraction requires the frozen 7B OpenVLA-OFT model and simulator; cached-feature optimization does not load the VLA. Real rollout outcomes can depend on simulator assets, image-processing libraries, CUDA kernels, and device-specific numerics, so record upstream revisions, checkpoint identity, config files, and generated cache hashes for reproduction runs.
