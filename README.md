# GPT Training AutoTuner

This repository contains a PyTorch implementation of a GPT-like language model, designed for character-level language modeling with various customizable parameters. The model includes features like multi-head attention, positional embeddings, and layer normalization.
Features

    Customizable Parameters: Easily modify model hyperparameters such as block size, batch size, learning rate, embedding size, number of layers, number of heads, and dropout rate.
    Checkpoint Saving: Automatically saves the best model during training based on validation loss.
    VRAM Limits: Checks VRAM usage to ensure the model fits within GPU memory limits, useful for avoiding out-of-memory errors.
    Hyperparameter Search: Perform grid search over various hyperparameter combinations to find the best model configuration.
    Text Generation: Generate text sequences based on the trained model.
    Utilizes Tensor Cores on RTX GPUs when available.

# Installation

First, create a new Conda environment and install the required packages:

conda create -n gpt-env 
conda activate gpt-env
conda install pickle
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install pylzma

Make sure you have the necessary training data files in place (training_data/vocab.txt, training_data/train_split.txt, training_data/val_split.txt).

# Usage

Model Training

To train the model with hyperparameter search, simply run:

python train.py

This script will evaluate multiple combinations of hyperparameters and save the best model to best_model.pt.
Text Generation

After training, you can load the best model into your LLM application and use it.

There is an included model in the class GPTLanguageModel(nn.Module). However you should be able to define your own model for auto-tuning as long as the syntax is consistent with the hyperparameter_search() function

# Code Overview

    Model Definition: The core model is defined in gpt_model.py using PyTorch's nn.Module. It includes classes for multi-head attention, feedforward layers, and blocks.
    Training Script: The main training logic is in train.py, which includes data loading, model training, and hyperparameter search.
    Data Loading: Data is loaded using memory-mapped files for efficient access and batching.
    Hyperparameter Search: The hyperparameter_search function in train.py tests various hyperparameter combinations and saves the best model based on validation loss.
    Text Generation: The generate method in the GPTLanguageModel class allows for generating new text sequences based on a trained model.

# Customization

Adjust the following parameters in train.py to customize the model and training process:

    block_size: Size of each training block.
    batch_size: Number of samples per batch.
    max_iters: Maximum number of training iterations.
    eval_interval: Interval at which to evaluate the model on validation data.
    learning_rate: Learning rate for the optimizer.
    eval_iters: Number of evaluation iterations.
    n_embd: Size of the embeddings.
    n_layer: Number of transformer layers.
    n_head: Number of attention heads.
    dropout: Dropout rate for regularization.

# VRAM Limits

To avoid running out of memory, the training script includes a check for VRAM usage. If the VRAM usage exceeds a predefined limit (e.g., 9.7GB I chose for an RTX 3080), the current hyperparameter combination is skipped.


