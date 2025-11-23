#  **README.md – End-to-End Structured Data Modeling with ANN**

# **End-to-End Structured Data Modeling with Artificial Neural Networks (ANN)**

This repository contains an end-to-end deep learning workflow for binary classification on structured data using an Artificial Neural Network (ANN).
The project includes data preprocessing, scaling, dataset preparation with `tf.data`, model building, training with callbacks, evaluating results, and generating predictions.

---

##  **Project Overview**

This project demonstrates a complete machine learning pipeline using the Diabetes dataset.
It includes:

* Data preprocessing
* Standardization using `StandardScaler`
* Creating TensorFlow datasets
* Building a neural network with Keras
* Training the model using Early Stopping and ModelCheckpoint
* Plotting training curves
* Inspecting weights and biases
* Making predictions using both NumPy arrays and TensorFlow datasets

---

##  **Objective**

To build a clean, reproducible deep learning workflow for structured tabular data using TensorFlow & Keras.

---

##  **Technologies & Libraries**

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* joblib

---
##  **Main Steps**

### **1. Data Preprocessing**

* Load dataset
* Separate features and target
* Apply `StandardScaler`
* Save scaler for future predictions

### **2. Train/Validation Split**

Uses an 80/20 split with a fixed random seed.

### **3. TensorFlow Dataset Pipeline**

* Convert NumPy arrays to `tf.data.Dataset`
* Apply batching and shuffling

### **4. Model Architecture**

```python
model = Sequential([
    Input(shape=(8,)),
    Dense(50, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### **5. Callbacks**

* EarlyStopping (with best weights restored)
* ModelCheckpoint (save best model to disk)

### **6. Evaluation**

* Plot training/validation loss
* Plot accuracy & AUC
* Extract best epoch metrics
* Inspect weight statistics

### **7. Predictions**

* Load saved model + saved scaler
* Predict on a new row or dataset

---

## **Training Visualizations**

Training curves for:

* Loss
* Accuracy
* AUC

(Plots are generated inside the script.)

---

## **Note on Educational Context**

This project was implemented by following a hands-on coding session from the **Deep Learning** course.
The implementation is therefore a **learning exercise**, not an original research project.


Hazır!
İstersen daha sade, daha teknik veya daha uzun bir README formatı da yazabilirim.
