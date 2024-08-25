# EEG Data Processing and Cognitive Load Recognition

This repository contains resources for EEG data processing and cognitive load recognition using a Multi-Head Attention EEGNet model. It includes original EEG data, MATLAB code for preprocessing, and Python code for classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Requirements](#requirements)

## Project Overview

This project aims to process EEG data and perform cognitive load recognition using advanced neural network techniques. The repository is organized into three main components:

1. **EEG Data**: Raw EEG data collected for cognitive load studies.
2. **MATLAB Code**: Scripts for preprocessing EEG data to prepare it for analysis.
3. **Python Code**: Implementation of a Multi-Head Attention EEGNet model for cognitive load classification.

## Data

The repository includes the following data files:

- `eeg_data/`: Directory containing raw EEG data files in [specify format, e.g., .set or .edf].

## Preprocessing

MATLAB code for preprocessing the EEG data is available in the `MATLAB` directory. 

To run the preprocessing scripts:

1. Open MATLAB.
2. Navigate to the `MATLAB` directory.
3. Run the preprocessing script:

   ```matlab
   pre-processing.m

## Model

The project uses a Multi-Head Attention EEGNet model for cognitive load recognition. The implementation is in Python and is located in the `eegnet` directory.

### Requirements

- Python 3.x
- pyTorch 2.x
- numpy
- mne
- scikit-learn

You can install the required Python packages using:

```bash
pip install -r requirements.txt
