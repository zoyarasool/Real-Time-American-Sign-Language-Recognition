# Real-Time American Sign Language (ASL) Recognition System

## Overview
This project implements a real-time American Sign Language (ASL) recognition system using deep learning and computer vision.
The system recognizes ASL alphabets (A–Z) and digits (0–9) from a live webcam feed.
This work is designed as an academic and research-oriented project, and also serves as a foundation for a complete sign-language-to-text communication system.

## Project Motivation
Communication barriers faced by deaf and mute individuals inspired this project.
The goal is to help bridge the communication gap by enabling computers to understand sign language in real time.
This project is especially meaningful as it is motivated by real-life challenges and aims to evolve into a full-featured assistive system.

## Features (Phase 1 – Implemented)
- Real-time ASL recognition using a webcam
- Recognition of:
ASL Alphabets (A–Z)
ASL Digits (0–9)
- MediaPipe-based hand detection and cropping
- CNN-based classification for alphabets and digits
- Runs fully on CPU (no GPU required)

## Technologies Used
- Webcam captures live video frames
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy

## System Architecture
- Webcam captures live video frames
- MediaPipe detects and localizes the hand
- Hand region is cropped and preprocessed
- CNN models classify the hand sign
- Output is displayed in real time

Note: MediaPipe is used only for hand detection, while CNN models handle sign classification.

## Project Structure
ASL/
├── interface/
│   └── interface.py
├── models/
│   ├── asl_alphabets_model.h5
│   └── asl_digits_model.h5
├── lables/
│   ├── alphabet_labels.json
│   └── digits_labels.json
├── training/
│   ├── train_alphabets.py
│   └── train_digits.py
└── datasets/
    └── README.txt   # Dataset information only

## Dataset Information
- ASL Alphabet Dataset
Source: Public ASL alphabet image datasets (Kaggle)
Contains labeled hand sign images for A–Z
- ASL Digit Dataset
Source: Public ASL digit datasets (Kaggle)
Contains labeled hand sign images for digits 0–9

Note: Full datasets are not included in this repository due to size constraints.C

## How to Run the Project
- Install Dependencies
  pip install tensorflow opencv-python mediapipe numpy
- Run the Interface
  python ASL/interface/interface.py
- Usage
  Show your hand clearly in front of the webcam
  Keep the hand steady for better predictions
  Press Q to exit

## Future Work
- Phase 2 (Planned):
  Word-level ASL recognition
  Landmark-based models using MediaPipe + LSTM
  Improved temporal modeling for dynamic signs
- Phase 3 (Planned):
  Sentence-level recognition
  Text-to-speech output
  Support for Pakistani Sign Language (PSL)
  Mobile application deployment
  
## Author
Zoya Rasool
Software Engineering Student
National University of Technology (NUTECH), Islamabad
