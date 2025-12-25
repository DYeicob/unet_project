# U-Net Image Segmentation Project

This project implements an image segmentation model based on the U-Net architecture using PyTorch.

## Project Structure

```
project/
│── data/
│   ├── images/ (Place your images here)
│   ├── masks/  (Place your masks here)
│
│── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── train.py
│   ├── utils.py
│   ├── metrics.py
│
│── notebooks/
│   ├── visualize_training.ipynb
│
│── requirements.txt
│── README.md
```

## Installation

1. Clone this repository or download the source files.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1.  Place your original images in `data/images/`.
2.  Place the corresponding segmentation masks in `data/masks/`.
    *   Masks must have the exact same filename as their corresponding images.
    *   Masks should be grayscale images where the pixels of interest have a value > 0.

## Training

To train the model, run the `train.py` script:

```bash
python src/train.py --epochs 20 --batch_size 8 --lr 0.001
```

Available Arguments:
*   `--epochs`: Number of training epochs (default: 20).
*   `--batch_size`: Batch size (default: 4).
*   `--lr`: Learning rate (default: 1e-4).
*   `--img_dir`: Directory for images (default: `data/images`).
*   `--mask_dir`: Directory for masks (default: `data/masks`).
*   `--checkpoint_dir`: Directory to save model checkpoints (default: `checkpoints`).

The best model will be automatically saved to `checkpoints/best_model.pth`.

## Visualization

Open the `notebooks/visualize_training.ipynb` notebook to visualize the performance and results of the trained model.

