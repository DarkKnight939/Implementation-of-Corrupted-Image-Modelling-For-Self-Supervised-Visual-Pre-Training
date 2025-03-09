# Implementation-of-Corrupted-Image-Modelling-For-Self-Supervised-Visual-Pre-Training

This repository contains a deep learning framework for **Corrupted Image Modeling (CIM)** using a generator-enhancer approach. The **Generator (BEiT-based model)** reconstructs missing image patches, while the **Enhancer (ResNet-based model)** improves image quality and detects manipulated regions.

## 📁 Project Structure
```
/your_project
│── /models       # Generator and Enhancer models
│── /training     # Training scripts and dataset handling
│── /utils        # Learning rate scheduling, loss functions
│── /tests        # Unit tests for model validation
│── /checkpoints  # Trained model weights
│── /test_results # Outputs from model testing
│── main.py       # Entry point for training
│── test_model.py # Script for testing the trained models
│── requirements.txt  # Dependencies
│── README.md     # Documentation
```

## 🚀 Installation
To set up the project, install the dependencies:
```sh
pip install -r requirements.txt
```

## 🔧 Training
To train the Generator and Enhancer models:
```sh
python main.py
```

## 🛠️ Testing
To test the trained models:
```sh
python test_model.py
```

## 📊 Model Components
- **Generator (BEiT-based model)**: Uses a **Masked Image Modeling (MIM)** approach to reconstruct missing patches.
- **Enhancer (ResNet-based model)**: Enhances image quality using pixel-wise prediction and token replacement detection.

## 📌 Checkpoints
Trained models are saved in the `checkpoints/` directory after every few epochs.

## 📝 License
MIT License
