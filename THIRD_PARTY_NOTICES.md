# Third-party notices

This repository contains the ReAlign-VLA adapter, corruption bank, training code, and integration utilities. It does **not** redistribute OpenVLA/OpenVLA-OFT source trees, third-party model weights, LIBERO/LIBERO-Plus assets, or datasets.

## Upstream projects

| Component | Repository / revision |
|---|---|
| OpenVLA | https://github.com/openvla/openvla |
| OpenVLA-OFT | https://github.com/moojink/openvla-oft @ `e4287e94541f459edc4feabc4e181f537cd569a8` |
| OFT Transformers fork | https://github.com/moojink/transformers-openvla-oft @ `bc339d9ad707454c0c115970db43c260067c61ab` |
| dlimp | https://github.com/moojink/dlimp_openvla @ `040105d256bd28866cc6620621a3d5f7b6b91b46` |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO @ `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-Plus | https://github.com/sylvestf/LIBERO-plus @ `4976dc30028e805ff8094b55501d532c48fec182` |

OpenVLA/OpenVLA-OFT, LIBERO, their pretrained components, datasets, model weights, and simulator assets remain subject to their respective licenses and terms. LIBERO-Plus is obtained separately and is not redistributed by this repository; users should review its repository terms before use or redistribution.

The fused OpenVLA-OFT vision stack includes pretrained DINOv2 and SigLIP components, which are likewise obtained through their upstream distributions. Core Python dependencies such as PyTorch, NumPy, OpenCV, and PyYAML retain their own licenses.
