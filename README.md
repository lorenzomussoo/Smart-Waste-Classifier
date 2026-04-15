# Smart Waste Classifier ♻️🤖

An AI-powered computer vision tool designed to help users correctly classify and dispose of waste, aiming to reduce environmental impact and improve recycling efficiency. This project utilizes Deep Learning to classify waste images into 7 distinct categories.

## 🎯 Overview
Proper waste sorting is a critical environmental challenge. This tool provides an intuitive graphical interface where users can upload an image of a waste item. The system runs the image through two parallel Machine Learning models and outputs the correct disposal category along with a confidence score.

### 🗂️ The 7 Waste Categories
1. Battery
2. Biological
3. Cardboard / Paper
4. Clothes
5. Glass
6. Metal
7. Plastic / Trash

## 🧠 Machine Learning Models
To evaluate different approaches to image classification, two distinct models were trained and compared:
1. **CNN (Convolutional Neural Network):** A deep learning approach tailored for image feature extraction and classification. (Fine-tuned for optimal accuracy).
2. **LR-SGD (Logistic Regression with Stochastic Gradient Descent):** A linear classifier serving as a baseline baseline to compare computational efficiency and accuracy against the CNN.

## 🛠️ Tech Stack & Methodology
* **Language:** Python
* **Deep Learning Framework:** PyTorch (for the CNN architecture and fine-tuning).
* **Machine Learning:** Scikit-Learn (for `SGDClassifier`).
* **User Interface:** Gradio (for building an interactive, browser-based web UI).
* **Data Preprocessing:** Custom scripts (`Merge.py`, `Clean.py`) were developed to merge and clean two distinct open-source datasets (including the well-known TrashNet dataset) to ensure a robust training pipeline.
* **Evaluation:** Confusion matrices, learning curves, and classification reports were generated to analyze model uncertainty and loss metrics.

## 📁 Repository Structure
* `/Code`: Contains the core Python scripts for data cleaning, model training, fine-tuning, and the Gradio UI application (`Progetto.py`).
* `/Test`: A collection of sample images to easily test the application's predictions.
* `/Utils/Analysis`: Evaluation reports, precision/recall CSVs, and visualization plots comparing the CNN and LR-SGD performance.
* `Report_AI_Lab.pdf`: The full academic report detailing the dataset composition, network architectures, and final evaluation metrics.

## 🚀 How to Run Locally

### Prerequisites
* Python 3.10+
* Install required dependencies (PyTorch, Scikit-Learn, Gradio, Pandas, Matplotlib, OpenCV/Pillow).

### Execution
1. Clone this repository:
   ```bash
   git clone [https://github.com/yourusername/Smart-Waste-Classifier.git](https://github.com/yourusername/Smart-Waste-Classifier.git)

2. (Note: The raw datasets and .pth model weights are excluded from this repository due to GitHub size constraints. You must train the models locally using Train.py and Fine-Tune.py after downloading the appropriate datasets).

3. To launch the user interface, run:
   ```bash
   python Code/Progetto.py

4. Access the web interface via the localhost link provided in your terminal, upload a test image from the /Test folder, and view the prediction.

---
*_Developed as a Machine Learning and Artificial Intelligence laboratory project. Focuses on data engineering, CNN architecture, and HCI via Gradio._
