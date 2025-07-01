# 🧠 EEG-Based Emotion Classification using a Hybrid GRU-CNN Model 🧠

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A deep learning project to classify human emotions from EEG brainwave data using a powerful hybrid model combining Convolutional Neural Networks (CNN) and Gated Recurrent Units (GRU).

---

## 📜 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [Data Preprocessing](#-data-preprocessing)
  - [Model Architecture](#-model-architecture)
- [How to Run](#-how-to-run)
- [Results](#-results)
- [Dependencies](#-dependencies)

---

## 🚀 Project Overview

This project explores the fascinating intersection of neuroscience and artificial intelligence. The primary goal is to build and train a robust deep learning model capable of discerning human emotions—specifically **Positive**, **Negative**, and **Neutral** states—by analyzing electroencephalography (EEG) signals.

We leverage a hybrid architecture that capitalizes on the strengths of both CNNs for spatial feature extraction from the EEG channels and GRUs for capturing temporal dependencies within the signal data.

---

## 📊 Dataset

The model is trained on the **EEG Brainwave Dataset: Feeling Emotions**, which contains pre-processed EEG data.

- **Source**: [Kaggle EEG Brainwave Dataset](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data)
- **Files Used**: `emotions.csv`
- **Labels**: The dataset is categorized into three emotional states:
  - `POSITIVE`
  - `NEGATIVE`
  - `NEUTRAL`

The distribution of the emotional labels in the dataset is as follows:


---

## 🛠️ Methodology

The project follows a structured approach from data preparation to model training and evaluation.

### 🧹 Data Preprocessing

1.  **Loading Data**: The EEG signals are loaded from the `features_raw.csv` and `emotions.csv` files.
2.  **Normalization**: To ensure the model converges effectively, the signal data is standardized by subtracting the mean and dividing by the standard deviation. This centers the data around zero and scales it to unit variance.
3.  **Segmentation**: The continuous EEG signals are segmented into smaller, fixed-size windows (e.g., 0.3 seconds). This creates uniform input samples for the model. The `EEGDataset` class in the notebook handles this process efficiently.

### 🤖 Model Architecture

The core of this project is a hybrid deep learning model that intelligently combines two powerful architectures:

1.  **Convolutional Neural Network (CNN)**:
    -   **`Conv1D` Layers**: These layers act as feature extractors. They apply convolutional filters to the input EEG signals to automatically learn and identify relevant spatial patterns and local features across the EEG channels.
    -   **`MaxPooling1D`**: This layer downsamples the feature maps, reducing dimensionality and making the learned features more robust.
    -   **`BatchNormalization`**: Used to stabilize and accelerate the training process.

2.  **Gated Recurrent Unit (GRU)**:
    -   **`GRU` Layers**: After feature extraction by the CNN, the GRU layers process the sequences of features. GRUs are a type of recurrent neural network (RNN) adept at learning temporal patterns, making them perfect for understanding how EEG signals evolve over time.

This combination allows the model to learn **"what to look for"** (via CNN) and **"when to look for it"** (via GRU) in the EEG data to make an accurate emotion classification.

---

## 🏃 How to Run

To replicate this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rudrakshmohanty/EEG-Based-Emotion-Classification-using-Hybrid-GRU-CNN-Model.git
    cd EEG-Based-Emotion-Classification-using-Hybrid-GRU-CNN-Model
    ```

2.  **Download the dataset:**
    -   Download the `features_raw.csv` and `emotions.csv` files from the [Kaggle dataset page](https://www.kaggle.com/datasets/shashwatwork/eeg-brainwave-dataset-feeling-emotions).
    -   Place them in a directory accessible by the notebook (e.g., a `data` folder).

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created containing the packages listed in the [Dependencies](#-dependencies) section).*

4.  **Run the Jupyter Notebook:**
    -   Open and run the `BIO_Final_2.ipynb` notebook in a Jupyter environment or Google Colab.
    -   Make sure to update the file paths for the dataset in the notebook cells.

---

## 📈 Results

The model is trained to classify the three emotional states. The notebook includes visualizations of:
-   The raw feature data from the EEG signals.
-   Sample EEG waveforms corresponding to each emotion (Positive, Negative, Neutral).
-   The final classification performance (e.g., accuracy, confusion matrix), which can be seen by running the notebook.

This structured approach provides a clear and effective way to tackle the complex task of emotion recognition from brainwave data.

---

## 📦 Dependencies

The project relies on the following Python libraries:

-   `tensorflow`
-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`
-   `pyod` (for outlier detection)

You can install them using pip:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn pyod
```
