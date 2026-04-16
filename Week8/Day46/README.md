# Assignment
# Day46

## Project Title

Transfer Learning Experiment: MNIST CNN Features for Social Media Post Classification

---

## Overview

This project tests whether a Convolutional Neural Network (CNN) trained on the MNIST handwritten digit dataset can transfer useful learned representations to a social media post classification task.

The goal is to evaluate if filters learned from clean digit images (edges, curves, strokes) can help classify noisy, imbalanced, multilingual social media data after converting posts into image-like representations.

This experiment focuses on **representation learning**, domain mismatch, and practical transfer learning performance.

---

## Problem Statement

MNIST contains:

* Balanced classes
* Clean centered grayscale images
* Simple visual patterns

Social media data contains:

* Imbalanced classes
* Mixed languages
* Noise, slang, emojis
* Semantic meaning instead of digit shapes

The key question:

**Do features learned on MNIST transfer effectively to social media classification?**

---

## Experiment Setup

Three models are compared:

### 1. Baseline: TF-IDF + Logistic Regression

Traditional NLP pipeline using text vectorization.

### 2. CNN Trained from Scratch

CNN trained directly on generated image thumbnails of text.

### 3. Transfer Learning Model

Uses convolution layers from pretrained MNIST CNN as a frozen feature extractor and trains a new classifier head.

---

## Input Representation

### Text Thumbnail Images

Each post is converted into a grayscale image by rendering text and resizing it to 28×28.

This allows image CNN filters to process text visually.

---

## Technologies Used

* Python
* Pandas
* NumPy
* OpenCV
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---

## Project Structure

```bash
project/
│── social_media_posts.csv
│── mnist_cnn.h5
│── main.py
│── README.md
```

---

## How to Run

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow opencv-python
```

### Run Project

```bash
python main.py
```

---

## Expected Results

Typical outcome:

| Model                        | Performance   |
| ---------------------------- | ------------- |
| TF-IDF + Logistic Regression | Strong        |
| CNN from Scratch             | Medium        |
| MNIST Transfer CNN           | Low to Medium |

Exact scores depend on dataset quality and label distribution.

---

## Key Findings

### Transfer Learning Works Only Partially

The first CNN layers may transfer generic visual features such as:

* edges
* corners
* stroke patterns
* texture density

### Why It Mostly Fails

MNIST features are learned for digit recognition, while social media classification depends on:

* language understanding
* sentiment
* context
* multilingual tokens
* semantics

There is strong domain mismatch between source and target tasks.

### Important Insight

Using a pretrained model is not always enough. Transfer learning works best when source representations match the target problem.

---

## Future Improvements

Better models for this task:

* Multilingual BERT
* XLM-R
* FastText
* Character-level CNN
* Text + Metadata Hybrid Models

---

## Conclusion

This project demonstrates that MNIST CNN filters provide limited benefit for social media classification. Early visual filters may transfer slightly, but deeper learned representations are task-specific.

The experiment highlights a core machine learning principle:


