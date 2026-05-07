# Pretrained Checkpoints

Download all required weights in one command:

```bash
python tools/blv/download_checkpoints.py
```

This creates `checkpoints/pretrained/` with the following files.

---

## Required (used by Final A/B/C/D configs)

| Model              | Filename                                                            | Source    |
| ------------------ | ------------------------------------------------------------------- | --------- |
| Mask2Former Swin-L | `mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth` | OpenMMLab |
| SegFormer MIT-B5   | `segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth` | OpenMMLab |
| SAN ViT-B/16       | `san-vit-b16_20230906-fd0a7684.pth`                                 | OpenMMLab |

Direct URLs:

```
https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth

https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_640x640_160k_ade20k/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth

https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-b16_20230906-fd0a7684.pth
```

## Optional (SAN large variant + CLIP backbone)

```
https://download.openmmlab.com/mmsegmentation/v0.5/san/san-vit-l14_20230907-a11e098f.pth
https://download.openmmlab.com/mmsegmentation/v0.5/san/clip_vit-base-patch16-224_3rdparty-d08f8887.pth
https://download.openmmlab.com/mmsegmentation/v0.5/san/clip-vit-large-patch14_3rdparty-d08f8887.pth
```

The `san_clip_b16` backbone is also auto-fetched by mmseg during the first SAN
inference if not already present.
