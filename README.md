## CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving

This repository is organized around the paper **“CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving”**, providing a concise overview plus pointers to the code/docs so the project can be understood and reproduced from a GitHub landing page.

- **Paper**: [`arXiv:2508.13485`](https://arxiv.org/abs/2508.13485)
- **arXiv submission date**: 2025-08-19
- **Venue**: IROS 2025 (marked as *Accepted to IROS 2025* on the paper page)
- **Authors**: Fuyang Liu, Jilin Mei, Fangyuan Mao, Chen Min, Yan Xing, Yu Hu


---

## Motivation

4D radar point clouds are attractive for autonomous driving due to robustness in adverse weather and the ability to provide spatial cues across diverse scenarios. However, 4D radar point clouds are often **sparse and noisy**, which can significantly hurt downstream perception (e.g., voxel-based detection).

CORENet targets this issue by leveraging **LiDAR supervision during training** to learn noise patterns and improve feature quality, while keeping **radar-only inference** for deployment.

---

## What CORENet does

CORENet is a **cross-modal denoising framework** with the following key properties (as stated in the abstract on [`arXiv:2508.13485`](https://arxiv.org/abs/2508.13485)):

- **Cross-modal supervision**: uses LiDAR supervision during training to identify noise patterns and learn discriminative features from raw 4D radar.
- **Plug-and-play**: designed to integrate into voxel-based detection frameworks without modifying existing pipelines.
- **Radar-only inference**: LiDAR is used only for supervision during training; inference operates with radar data only.
- **Validated on Dual-Radar**: extensive evaluation is reported on the challenging Dual-Radar dataset with elevated noise levels.

---

## How to think about it

Practically, you can view CORENet as a **learned denoising / feature enhancement module** for radar point clouds:

- **Training**: uses LiDAR as a supervision signal to teach the model what “clean / reliable geometry” looks like.
- **Inference**: removes the LiDAR dependency and runs in **radar-only** mode.

---

## Repository entry points

Start here for usage and reproduction:

- **Installation**: `docs/INSTALL.md`
- **Getting started**: `docs/GETTING_STARTED.md`
- **Demo**: `docs/DEMO.md` and `tools/demo.py`

Common scripts (examples):

- `tools/train.py`: training
- `tools/test.py`: evaluation
- `tools/cfgs/`: model & dataset configs

---

## Citation (BibTeX)

If you find this work useful, please cite the paper (metadata from [`arXiv:2508.13485`](https://arxiv.org/abs/2508.13485)):

```bibtex
@article{liu2025corenet,
  title        = {CORENet: Cross-Modal 4D Radar Denoising Network with LiDAR Supervision for Autonomous Driving},
  author       = {Liu, Fuyang and Mei, Jilin and Mao, Fangyuan and Min, Chen and Xing, Yan and Hu, Yu},
  year         = {2025},
  eprint       = {2508.13485},
  archivePrefix= {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2508.13485},
  url          = {https://arxiv.org/abs/2508.13485}
}
```

---

## License

See `LICENSE`.


