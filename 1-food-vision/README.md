# 🍔 Food Vision - TensorFlow Food Classification

Welcome to the **Food Vision** project! 🚀 This repository contains my implementation of a deep learning model for food classification using TensorFlow and the **Food101** dataset. The project is based on the Udemy course **"TensorFlow for Deep Learning Bootcamp"** by **Andrei Neagoie** and **Daniel Bourke (Zero to Mastery)**.

---

## 📜 Table of Contents

- [📌 Overview](#-overview)
- [📂 Project Structure](#-project-structure)
- [⚙️ Prerequisites](#️-prerequisites)
- [🛠️ Usage](#%EF%B8%8F-usage)
- [💡 Code Walkthrough](#-code-walkthrough)
  - [1️⃣ Setup & Dataset Loading](#1%EF%B8%8F-setup--dataset-loading)
  - [2️⃣ Data Exploration & Preprocessing](#2%EF%B8%8F-data-exploration--preprocessing)
  - [3️⃣ Model Creation & Compilation](#3%EF%B8%8F-model-creation--compilation)
  - [4️⃣ Training, Fine-Tuning & Evaluation](#4%EF%B8%8F-training-fine-tuning--evaluation)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)

---

## 📌 Overview

This project builds a deep learning model to classify food images using the **Food101** dataset. The workflow includes:

✅ Loading and preprocessing the dataset 🍕🍔🥗  
✅ Training a **TensorFlow EfficientNetB0** model using feature extraction 🎯  
✅ Fine-tuning the model for better accuracy 🔍  
✅ Saving and reloading the trained model for reuse 💾  

---

## 📂 Project Structure

```
1-food-vision/
├── helper_functions.py    # Helper functions for callbacks and visualization
├── model.ipynb            # Jupyter Notebook containing the complete code
└── README.md              # This README file
```

---

## ⚙️ Prerequisites

Before running the project, ensure you have the following installed:

- 🐍 **Python 3.10 or earlier**
- 🤖 **TensorFlow 2.8.0**
- 📦 **TensorFlow Datasets (tfds)**
- 📌 **matplotlib, scikit-learn and opencv-python**

---

## 📂 Dataset: Food-101 🍛
The **Food-101** dataset is a large-scale food image dataset introduced by **Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool**.

### 📥 Download Food-101 Dataset
To download the dataset, run the following commands in your terminal:

```bash
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvzf food-101.tar.gz
```

### Alternatively, download it manually from the official sources:

- 🔗 [Food-101 Dataset Official Page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

- 🔗 [Food-101 on Kaggle](https://www.kaggle.com/datasets/kmader/food41)

## 🛠️ Usage

To run the project, open the Jupyter Notebook (`food_vision.ipynb`) and execute the cells step by step. If using **Google Colab**, ensure GPU acceleration is enabled for faster training. 🚀

---

## 💡 Code Walkthrough

### 1️⃣ Setup & Dataset Loading

The project starts by verifying the TensorFlow version and loading the dataset:

```python
import tensorflow as tf
print(f"Tensorflow version: {tf.__version__}")

import tensorflow_datasets as tfds
(train_data, test_data), ds_info = tfds.load(
    name="food101",
    split=["train", "validation"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
class_names = ds_info.features["label"].names
print(class_names[:10])
```

---

### 2️⃣ Data Exploration & Preprocessing

A sample image is displayed along with its class name:

```python
import matplotlib.pyplot as plt
plt.imshow(image)
plt.title(class_names[label.numpy()])
plt.axis(False)
```

Images are preprocessed for model training:

```python
def preprocess_img(image, label, img_shape=224):
    image = tf.image.resize(image, [img_shape, img_shape])
    return tf.cast(image, tf.float32), label
```

---

### 3️⃣ Model Creation & Compilation

An **EfficientNetB0** model is created and compiled:

```python
from tensorflow.keras import layers

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(len(class_names))(x)
outputs = layers.Activation("softmax", dtype=tf.float32)(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
```

---

### 4️⃣ Training, Fine-Tuning & Evaluation

Model is trained with feature extraction first:

```python
history = model.fit(train_data, epochs=3, validation_data=test_data)
```

Fine-tuning is enabled for better performance:

```python
for layer in base_model.layers:
    layer.trainable = True

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(0.0001),
              metrics=["accuracy"])

history_fine_tune = model.fit(train_data, epochs=10, validation_data=test_data)
```

Model is saved and reloaded for evaluation:

```python
model.save("food_vision_model")
loaded_model = tf.keras.models.load_model("food_vision_model")
results = loaded_model.evaluate(test_data)
print(results)
```

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/Vardhan1303/tensorflow/blob/main/LICENSE) file for more details.

---

## 🙏 Acknowledgements

- 💡 **Andrei Neagoie** & **Daniel Bourke** for the **Zero to Mastery** TensorFlow course.  
- 🍽️ Food-101 dataset creators: Lukas Bossard, Matthieu Guillaumin, & Luc Van Gool.
- 🚀 TensorFlow & TensorFlow Datasets teams for providing great tools & datasets.  
- 🌟 Open-source contributors for their continued support & inspiration.  

🔥 **Happy Coding!** 🚀🍔
