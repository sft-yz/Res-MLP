# Residual MLP for SuperDARN Electric Potential Prediction

This repository contains the source code for training the Res-MLP model to predict ionospheric convection electric potential maps from OMNI and spatial feature inputs.

## Overview

The code:
- Loads matched OMNI–SuperDARN potential `.mat` files
- Builds point-wise training samples from each potential frame
- Constructs geographic and Fourier-encoded position features
- Normalizes inputs using `MinMaxScaler`
- Trains the Res-MLP model in TensorFlow/Keras
- Saves the best model, scaler, training history, and training curves

## Main Features
- Frame-wise train/validation split
- Fourier feature encoding for:
  - geographic longitude
  - magnetic local time (MLT)
  - latitude
- Residual MLP architecture with:
  - Dense layers
  - Layer Normalization
  - GELU activations
  - Dropout
- Automatic saving of:
  - trained model
  - input scaler
  - global output range
  - training history
  - loss/MAE curves
