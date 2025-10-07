# Rotational Motion Deblurring using Deep Learning in Polar Coordinates

This repository provides the official PyTorch implementation for the paper "[**CAR-Net: A Cascade Refinement Network for Rotational Motion Deblurring under Angle Information Uncertainty**]". We present a deep learning framework for removing rotational motion blur from images by leveraging polar coordinate transforms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We propose a new neural network architecture called CAR-net (CAscade Refinement Network) to deblur images that are subject to rotational motion blur. Our approach behind this architecture is designed for the situations where only noisy information of the rotational motion blur angle is available. This uncertainty is taken into account in angle detection modules of our architecture. In these modules we iteratively refine the initial deblurred estimate obtained from frequency-domain inversion; each stage takes the current deblurred image to predict the residual correction amount, which is added to the current estimate, progressively suppressing artifacts and restoring fine details. Our architecture can also accommodate an optional angle detection module which can be trained together with other modules. We provide a detailed description of our architecture and illustrate its efficiency through experiments using both synthetic and real-world images.

## Model Architecture

Our framework utilizes a novel non-blind iterative model for deblurring in the polar domain. This advanced iterative model is designed to jointly estimate the blur angle and restore the sharp image, making it suitable for real-world scenarios where the blur parameters are unknown. It progressively refines the deblurred result over several stages.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- Kornia
- NumPy
- Matplotlib

## Installation

1.  **Clone the repository:**
    ```bash
    git clone XXXX
    cd CAR-Net
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

## Datasets

The datasets used for training and evaluation can be downloaded from the following links.

-   **Simple Pattern Dataset:** [Download Link](https://figshare.com/s/ee3d9104d6e67499a71e)
-   **Real-World Dataset:** [Download Link](https://figshare.com/s/a016eae74e099b32eaa7)

### Training

To train the model from scratch, configure the settings in `config.py` and run the training script.

1.  **Edit Configuration:**
    Open `config.py` and set the following:
    -   Dataset paths for training and validation.
    -   Enable or disable specific model components or features.
    -   Training hyperparameters (learning rate, batch size, epochs).
    -   Device settings (`'cuda'` or `'cpu'`).

2.  **Run Training:**
    ```bash
    python main.py
    ```
    Checkpoints and logs will be saved to the output directory specified in the configuration file.

### Inference with Pre-trained Models

Use the `test.py` script to deblur images. You can process a single image, a directory of images, or files matching a glob pattern.

**1. Batch Processing from CSV**

Process images specified in a CSV file. This is useful for reproducing paper results.
```bash
python test.py --csv_path ./Real-World Rotational_Motion_Blur_Datasets/test/parameters_with_noise_5.csv --output ./output/
```

**Command Line Arguments:**

-   `--input`: Path to the input image, directory, or glob pattern. Required if `--csv_path` is not used.
-   `--csv_path`: Path to a CSV file containing image paths and parameters.
-   `--output`: Path for the output file or directory (required).
-   `--model_path`: Path to trained model weights (`.pth`). Defaults to the path in `config.py`.
-   `--blur_angle`: Estimated blur angle in degrees (default: `30.0`). Required for the model.

## File Structure

```
Rotational-Deblur/
├── models/                 # Neural network architecture
├── utils/                  # Utility functions (e.g., coordinate transforms)
├── core/                   # Core training and evaluation logic
├── data/                   # Dataset loading and preprocessing
├── test.py                 # Inference script for testing models
├── train.py                # Main training script
├── config.py               # Central configuration file
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Citation

If you find this work useful for your research, please consider citing our paper.

*Note: This paper is currently under review for AAIML 2026. The BibTeX entry will be updated upon acceptance.*
```bibtex
@inproceedings{Author2026carnet,
  title     = {CAR-Net: A Cascade Refinement Network for Rotational Motion Deblurring under Angle Information Uncertainty},
  author    = {Author1,Author 2},
  booktitle = {Submitted to International Conference on Advances in Artificial Intelligence and Machine Learning (AAIML)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file