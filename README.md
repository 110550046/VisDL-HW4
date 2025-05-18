# NYCU CV 2025 Spring HW4

StudentID: 110550046  
Name: 吳孟謙

## Introduction

This is the implementation for Homework 4 of NYCU Computer Vision (Spring 2025).  
The task is **Image Restoration**, which involves recovering clean images from degraded inputs (Rain and Snow types).  

I implement a custom model based on **PromptIR**, a unified restoration framework utilizing learnable prompts to represent degradation types. my version integrates channel & spatial attention mechanisms to enhance feature discrimination.

### Methods & Hyperparameters

- **Model:** PromptIR with PromptBlock + Attention Modules  
- **Backbone:** Shallow CNN with 16 stacked PromptBlocks  
- **PromptBlock:** Learnable prompt + Channel Attention + Spatial Attention  
- **Loss:** MSE Loss  
- **Batch Size:** 8  
- **Optimizer:** Adam (lr = 1e-4)  
- **LR Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)

## How to install & run  

### dependencies  
Run the following instruction at terminal with pip
```bash
pip install torch torchvision Pillow numpy tqdm scikit-image
```
### Train & Predict
Just run the PIR_train.py & PIR_pred.py, make sure hw4_realse_dataset folder is at the same directory.

## Performance Snapshot
![image](https://github.com/user-attachments/assets/b84105c5-c931-480b-adb8-2c7ac2b08058)

