# Real-Time American Sign Language (ASL) Recognition System
This project implements a **real-time American Sign Language (ASL) recognition system** using deep learning and computer vision.
The system recognizes **ASL alphabets (A–Z)** and **digits (0–9)** from a live webcam feed.
This work is designed as an **academic and research-oriented project**, and also serves as a foundation for a complete **sign-language-to-text communication system**.

## Project Motivation
Communication barriers faced by deaf and mute individuals inspired this project.  
The goal is to help bridge the communication gap by enabling computers to understand sign language in real time.
This project is especially meaningful as it is motivated by real-life challenges and aims to evolve into a full-featured assistive system.

## Features (Phase 1 – Implemented)
- Real-time ASL recognition using a webcam
- Recognition of:
  - ASL Alphabets (A–Z)
  - ASL Digits (0–9)
- MediaPipe-based hand detection and cropping
- CNN-based classification for alphabets and digits
- Prediction smoothing for improved stability
- Runs fully on CPU (no GPU required)
  
## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy

## System Architecture
1. Webcam captures live video frames  
2. MediaPipe detects and localizes the hand  
3. Hand region is cropped and preprocessed  
4. CNN models classify the hand sign  
5. Output is displayed in real time  

MediaPipe is used only for hand detection, while CNN models handle sign classification.

## Project Structure
ASL/
├── interface/
│ └── interface.py
├── models/
│ ├── asl_alphabets_model.h5
│ └── asl_digits_model.h5
├── lables/
│ ├── alphabet_labels.json
│ └── digits_labels.json
├── training/
│ ├── train_alphabets.py
│ └── train_digits.py
│
└── datasets/
└── README.txt # Dataset information only

## Dataset Information
- **ASL Alphabet Dataset**
  - Source: Public ASL alphabet image datasets (Kaggle)
  - Contains labeled hand sign images for A–Z
- **ASL Digit Dataset**
  - Source: Public ASL digit datasets (Kaggle)
  - Contains labeled hand sign images for digits 0–9

Note: Full datasets are **not included** in this repository due to size constraints.

## How to Run the Project
1️. Install Dependencies
```bash
pip install tensorflow opencv-python mediapipe numpy
2. Run the interface
python ASL/interface/interface.py
3. Usage
Show your hand clearly in front of the webcam
Keep the hand steady for better predictions
Press Q to exit

## Future Work
### Phase 2 (Planned):
Word-level ASL recognition
Landmark-based models using MediaPipe + LSTM
Improved temporal modeling for dynamic signs
### Phase 3 (Planned):
Sentence-level recognition
Text-to-speech output
Support for Pakistani Sign Language (PSL)
Mobile application deployment

## Limitations
Accuracy may vary due to:
Lighting conditions
Background noise
Camera quality
Some ASL alphabets and digits have visually similar hand shapes
Designed as a prototype, not a production-ready system

These limitations are common in real-time vision-based recognition systems.

## Author
Zoya Rasool
Software Engineering Student
