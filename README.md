# ML-models-MCML
Machine learning models and source code for the Multiscale Computational Mechanics Laboratory at Vanderbilt University.

## LSTM Model for Predicting Micro Displacement, Velocity, and Acceleration

This project involves training an LSTM model to predict microscale displacement, velocity, and acceleration values based on given macroscale and microscale homogeneous and heterogeneous data. The model is trained on multiple datasets and evaluated for its performance.

## Project Structure

- `main.py`: Script for training models for individual datasets
- `single_model.py`: Script for training a model on multiple datasets using Strategy 1 (training on first 80% of every dataset and testing on last 20% of every dataset)
- `single_model_random.py`: Script for training a model on multiple datasets using Strategy 2 (training on 8 randomly selected datasets, testing on remaining 3 datasets--assumes that there are 11 datasets total)
- `predict.py`: Script for loading a trained model and making predictions on new data
- `l2_norm_normalized.py`: Script for computing and tabulating the normalized L2 norm of LSTM predictions vs actual data
- `animate.py`: Script for processing and animating the data in a 1D configuration, with respect to time

## Requirements

To run the scripts in this project, you need to have the following dependencies installed:

- Python 3.8 or higher
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Joblib
- FFmpeg (for creating animations in `animate.py`)

## Installation

1. **Install the required Python packages:**

    ```sh
    pip install numpy matplotlib scikit-learn torch joblib
    ```

2. **Install FFmpeg:**

    - Download FFmpeg from the [official website](https://ffmpeg.org/download.html) and follow the installation instructions for your operating system.
    - Ensure that the `ffmpeg` executable is in your system's PATH.
      
## Usage

### Training Models for Individual Datasets

To train models for individual datasets, run the `main.py` script. This script will prepare the data, train the model, and evaluate its performance.

### Training a Single Model

To train a model for multiple datasets using Strategy 1, run the `single_model.py` script.
To train a model for multiple datasets using Strategy 2, run the `single_model_random.py` script.

### Making Predictions

To make predictions using a trained model, run the `predict.py` script. This script will load the trained model and scalers, prepare the input data, and output the predictions.

### Computing Normalized L2 Norm

To compute the normalized L2 norm of the predictions, run the `l2_norm_normalized.py` script. This script will compare the actual and predicted values and save the results to a PDF file.

### Animating the Data

To create animations of the data, run the `animate.py` script. This script will process data and save the animations as MP4 files.
