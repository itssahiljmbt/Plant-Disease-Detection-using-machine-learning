# 🌱 Plant Disease Detection Using Machine Learning

## 📌 Overview

Plant diseases significantly affect agricultural productivity and food security. Early and accurate detection of plant diseases helps farmers take timely action, reduce crop loss, and improve yield quality.

This project presents a **machine learning–based plant disease detection system** that identifies plant diseases from leaf images using **deep learning techniques**. The system classifies plant leaf images into healthy or diseased categories with high accuracy, enabling fast and reliable disease diagnosis.

---

## 🎯 Objectives

* To detect plant diseases automatically using leaf images
* To reduce dependency on manual inspection by experts
* To provide a fast, cost-effective, and accurate disease prediction system
* To demonstrate the practical application of machine learning in agriculture

---

## 🧠 Methodology

1. **Data Collection**

   * A labeled dataset of plant leaf images containing healthy and diseased samples was used.
   * Images were divided into training and testing sets.

2. **Data Preprocessing**

   * Image resizing and normalization
   * Noise reduction and image enhancement
   * Data augmentation to improve model generalization

3. **Model Development**

   * A **Convolutional Neural Network (CNN)** was implemented for image classification.
   * The CNN automatically extracts features such as texture, color, and shape from leaf images.

4. **Training & Evaluation**

   * The model was trained on the preprocessed dataset.
   * Performance was evaluated using accuracy and confidence scores.

5. **Prediction**

   * The trained model predicts the disease class of a given plant leaf image along with prediction confidence.

---

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Machine Learning:** TensorFlow / Keras
* **Image Processing:** OpenCV, PIL
* **Numerical Computing:** NumPy
* **Backend (Optional):** FastAPI
* **Frontend (Optional):** ReactJS
* **Development Environment:** VS Code

---

## 📂 Project Structure

```
Plant-Disease-Detection-using-machine-learning/
│
├── model/
│   └── cnn_model.py
│
├── utils/
│   └── image_preprocessing.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Results

* The CNN model achieved **high accuracy (~90%+)** on the test dataset.
* The system successfully identifies plant diseases from unseen images.
* Provides confidence scores for each prediction to indicate reliability.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/itssahiljmbt/Plant-Disease-Detection-using-machine-learning.git
cd Plant-Disease-Detection-using-machine-learning
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python app.py
```

---

## 📁 Dataset & Model

* **Dataset:** Not included in the repository due to size constraints
* **Trained Model:** Not included in the repository

📌 **Dataset and trained model can be accessed via external links:**

* Dataset: *(Add Google Drive / Kaggle link here)*
* Trained Model: *(Add Google Drive link here)*

---

## 🔍 Applications

* Smart agriculture systems
* Crop health monitoring
* Farmer decision-support systems
* Agricultural research and automation

---

## 🌟 Advantages

* Reduces human effort and error
* Fast and accurate disease detection
* Scalable for multiple plant species
* Can be integrated with mobile or web applications

---

## 🚧 Limitations

* Requires good quality images for accurate prediction
* Performance depends on dataset size and diversity
* Limited to diseases present in the training dataset

---

## 🔮 Future Scope

* Support for more plant species and diseases
* Mobile application integration
* Real-time disease detection using camera input
* Cloud deployment for large-scale usage

---

## 👤 Author

**Sahil Rathor**
Computer Science Engineering Student
Machine Learning & Full-Stack Development Enthusiast

📌 GitHub: [https://github.com/itssahiljmbt](https://github.com/itssahiljmbt)

---

## 📜 License

This project is intended for **educational and research purposes only**.
