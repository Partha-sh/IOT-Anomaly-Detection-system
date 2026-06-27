# Industrial IoT Anomaly Detection System using Autoencoders

## Overview

This project implements an Industrial IoT Anomaly Detection System using a TensorFlow Autoencoder. The objective is to identify abnormal machine behavior and potential failures using sensor data collected from manufacturing equipment.

Unlike traditional supervised classification approaches, this project uses an unsupervised learning technique. The model is trained only on healthy machine data and learns normal operating patterns. When a machine behaves differently from what the model has learned, the reconstruction error increases and the system flags the observation as an anomaly.

The project also includes a real-time monitoring simulation and an interactive Streamlit dashboard for anomaly detection.

---

## Problem Statement

In industrial environments, machine failures can result in production downtime, maintenance costs, and operational losses. Since failure events are relatively rare compared to normal machine operations, training a traditional classification model can be challenging.

To address this issue, an Autoencoder-based anomaly detection approach was implemented. The model learns normal machine behavior and identifies unusual patterns without relying heavily on labeled failure examples.

---

## Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset

Sensor Features Used:

* Air Temperature [K]
* Process Temperature [K]
* Rotational Speed [rpm]
* Torque [Nm]
* Tool Wear [min]

Target:

* Machine Failure (used only for evaluation)

Columns such as UDI and Product ID were excluded because they do not contribute to machine behavior analysis.

---

## Project Workflow

### 1. Data Preprocessing

* Loaded and explored the dataset
* Removed non-informative columns
* Selected machine sensor features
* Separated normal and failure records

### 2. Training Data Preparation

* Trained only on normal machine records
* Split normal data into training and validation sets
* Applied StandardScaler for feature normalization

### 3. Autoencoder Architecture

Architecture:

Input (5)
↓
Dense (4)
↓
Bottleneck (2)
↓
Dense (4)
↓
Output (5)

The bottleneck layer compresses machine information into a lower-dimensional representation, forcing the network to learn meaningful patterns.

---

## How the Autoencoder Works

The Autoencoder receives machine sensor readings as input and attempts to reconstruct the same readings at the output.

Input:

Air Temp
Process Temp
RPM
Torque
Tool Wear

↓

Encoder

↓

Bottleneck Representation

↓

Decoder

↓

Reconstructed Input

If the machine behaves normally, reconstruction is accurate and the reconstruction error remains low.

If the machine behaves abnormally, reconstruction becomes difficult and the reconstruction error increases significantly.

---

## Reconstruction Error

Reconstruction Error is calculated using Mean Squared Error (MSE):

Error = Mean((Original - Reconstructed)^2)

Decision Rule:

* Error > Threshold → Anomaly
* Error ≤ Threshold → Normal

Threshold values were selected using validation-set reconstruction errors and further tuned to balance precision and recall.

---

## Model Evaluation

Metrics Used:

* Confusion Matrix
* Precision
* Recall
* F1 Score
* Accuracy

Final Model Results:

* Accuracy: ~84%
* Normal Class Precision: 0.91
* Normal Class Recall: 0.90
* Failure Class Precision: 0.47
* Failure Class Recall: 0.50

The project also included threshold tuning experiments to study the tradeoff between false positives and false negatives.

---

## Real-Time Monitoring Simulation

A real-time anomaly detection pipeline was implemented where machine sensor readings are processed one at a time.

Workflow:

Sensor Reading
↓
StandardScaler
↓
Autoencoder
↓
Reconstruction Error
↓
Threshold Check
↓
Normal / Anomaly

This simulates how an anomaly detection system would operate in a production environment.

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Joblib
* Streamlit

---

## What I Learned

This project significantly improved my understanding of Machine Learning and Deep Learning concepts.

### Machine Learning Concepts

* Data preprocessing and feature selection
* Training and validation data splitting
* Feature scaling using StandardScaler
* Model evaluation using confusion matrices and classification metrics
* Precision, Recall, F1 Score, and Accuracy analysis
* Threshold tuning and decision boundary selection

### Deep Learning Concepts

* Neural Networks and Dense Layers
* Autoencoders
* Encoder and Decoder architecture
* Bottleneck representations
* Reconstruction Error
* Weight updates through backpropagation
* Loss minimization using Adam Optimizer

### Anomaly Detection Concepts

* Unsupervised learning
* Learning normal behavior instead of failure patterns
* Detecting anomalies through reconstruction difficulty
* Tradeoffs between false alarms and missed failures
* Real-world predictive maintenance systems

### Project Development Skills

* Building end-to-end ML pipelines
* Saving and loading trained models
* Deploying machine learning solutions
* Creating interactive dashboards with Streamlit
* Presenting technical results through visualizations and reports

---

## Future Improvements

* Experiment with deeper Autoencoder architectures
* Hyperparameter optimization
* Live sensor data integration
* Time-series anomaly detection
* Cloud deployment and monitoring
* Explainable AI for anomaly interpretation

---

## Conclusion

This project demonstrates how Autoencoders can be used for Industrial IoT anomaly detection by learning normal machine behavior and identifying unusual operating conditions through reconstruction error analysis. The system provides a practical foundation for predictive maintenance applications and real-time industrial monitoring.
