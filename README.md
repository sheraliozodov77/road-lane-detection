# Road Lane Detection using U-Net++

This project implements a deep learning pipeline for **road lane segmentation** using the **U-Net++ architecture**. The model is trained to predict lane markings from dashcam-style road images. It includes preprocessing, training, validation, evaluation, and visualization steps, all executed in a GPU environment.

## Dataset Description

This project uses a subset of the [CULane Dataset](https://xingangpan.github.io/projects/CULane.html), specifically the `driver_161_90frame` images and the corresponding `laneseg_label_w16` segmentation masks. These were used to train a binary lane segmentation model, focusing on distinguishing lane markings from the background in complex urban driving scenarios.

- **Input Size:** 256 × 448 (resized for training)
- **Channels:** RGB (3)
- **Mask Type:** Single-channel binary mask
- **Total Samples:** ~18,000
- **Format:** PNG images and masks stored under structured folders:
  ```
  /train/images/
  /train/masks/
  /val/images/
  /val/masks/
  /test/images/
  /test/masks/
  ```

## Model Architecture

The core architecture used is **U-Net++**, a powerful nested encoder-decoder structure with dense skip connections.

### Configuration:
- **Backbone:** Custom (built from scratch, not pretrained)
- **Input Shape:** (256, 448, 3)
- **Output:** (256, 448, 1) binary mask with sigmoid activation
- **Total Parameters:** 2,297,441
- **Loss Function:** BCE + Dice Loss (`bce_dice_loss`)
- **Optimizer:** Adam with Mixed Precision
- **Learning Rate:** 1e-4 with ReduceLROnPlateau
- **Callbacks:**
  - `ModelCheckpoint` (val_loss)
  - `EarlyStopping` (patience=3)
  - `ReduceLROnPlateau` (factor=0.5, patience=2)
  - Resume logic enabled

## Training Pipeline

Training was conducted on Kaggle with GPU (Tesla T4).

### Hyperparameters:
- **Epochs:** 20
- **Batch Size:** 4
- **Precision:** Mixed (`float16`)
- **Callbacks:** Best model checkpoint saved as `unetpp_best.keras`

## Evaluation Metrics (on Test Set)

| Metric              | Value         |
|---------------------|---------------|
| **Test Accuracy**   | 0.9844        |
| **Dice Coefficient**| 0.6451        |
| **Loss**            | 0.4178        |


## Challenges Faced

- **GPU Memory Limitations:** Required tuning batch size
- **Large Model Depth:** Needed mixed precision and efficient checkpointing

## Future Work

- Try **Attention U-Net** or **U-Net++ with pretrained backbones**
- Experiment with **Tversky Loss** and **Focal Loss**

## Environment

- **Framework:** TensorFlow 2.x
- **Hardware:** Kaggle (Tesla T4, 16GB RAM)
- **Mixed Precision:** Enabled for memory optimization
