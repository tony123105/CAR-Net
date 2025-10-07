# Rotational Motion Deblurring using Deep Learning in Polar Coordinates

This repository provides the official PyTorch implementation for the paper "[**CAR-Net: A Cascade Refinement Network for Rotational Motion Deblurring under Angle Information Uncertainty**]". We present a deep learning framework for removing rotational motion blur from images by leveraging polar coordinate transforms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Rotational motion blur is a common image degradation that is challenging to address in the Cartesian domain. This work proposes a framework that transforms the blurred image into a polar coordinate system, where the rotational blur manifests as a simpler, spatially-invariant blur. We then apply a novel non-blind iterative deep neural network to perform deblurring in this transformed domain before converting the image back to Cartesian coordinates. This approach simplifies the deconvolution task and allows our model to jointly estimate blur parameters and restore the sharp image.

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
    git clone https://github.com/tony123105/CAR-Net.git
    cd CAR-Net
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Inference with Pre-trained Models

Use the `test.py` script to deblur images. You can process a single image, a directory of images, or files matching a glob pattern.

**1. Batch Processing from CSV**

Process images specified in a CSV file. This is useful for reproducing paper results.
```bash
python test.py --csv_path ./data/Real-World Rotational_Motion_Blur_Datasets/test/parameters_with_noise_5.csv --output ./output/
```

**Command Line Arguments:**

-   `--input`: Path to the input image, directory, or glob pattern. Required if `--csv_path` is not used.
-   `--csv_path`: Path to a CSV file containing image paths and parameters.
-   `--output`: Path for the output file or directory (required).
-   `--model_path`: Path to trained model weights (`.pth`). Defaults to the path in `config.py`.
-   `--blur_angle`: Estimated blur angle in degrees (default: `30.0`). Required for the model.

## Datasets

The datasets used for training and evaluation can be downloaded from the following links.

-   **Simple Pattern Dataset:** [Download Link](https://drive.google.com/file/d/17DTjfxXfxL2narTW_ZTQzRuE8VIe9an_/view?usp=sharing)
-   **Real-World Dataset:** [Download Link](https://drive.google.com/file/d/1iG-AaW0Vd1-Y3sKZexAUTHKWa2acazkT/view?usp=sharing)

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
@inproceedings{lai2026carnet,
  title     = {CAR-Net: A Cascade Refinement Network for Rotational Motion Deblurring under Angle Information Uncertainty},
  author    = {Lai, Ka Chung and Cetinkaya, Ahmet},
  booktitle = {Submitted to International Conference on Advances in Artificial Intelligence and Machine Learning (AAIML)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file