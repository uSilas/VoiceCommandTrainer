# VoiceCommandTrainer

This repository presents an experimental study on speech command recognition for robotic navigation. The project investigates the performance of **MLP** and **SVM** classifiers using a **collaborative learning approach**, where discriminative audio representations are extracted through a **supervised LSTM network**.

The objective is to classify voice commands such as navigation instructions (e.g., left, right, forward, back, and stop) from audio recordings and evaluate the effectiveness of different machine learning models on the extracted feature space.

## Dataset

The audio dataset used in this study is publicly available at:

**[https://drive.google.com/drive/folders/1kojJ3HZkAvbN08krQImQ2ekQtoLPOipA?usp=sharing]**

## Repository Contents

- Training of supervised LSTM models for feature extraction.
- Training and evaluation of MLP and SVM classifiers.
- Performance comparison between classification approaches.
- Experimental scripts and notebooks used throughout the study.

## Application

The developed models are intended for voice-controlled virtual-car systems, enabling navigation through spoken commands.
