# 🚗 Equitable AI on the Road

### Robust Object Detection and Segmentation Under Adverse Conditions with Bias Mitigation

---

## 🌐 Overview

Autonomous vehicles must function reliably under all circumstances — especially during **fog**, **frost**, and **snow**. However, most current object detection systems falter in these adverse weather conditions.

This project improves the resilience of object detection and segmentation models by:

* Enhancing detection robustness under environmental noise.
* Addressing **bias** using fairness-aware evaluations.
* Adding **explainability** for model transparency.

---

## ✨ Key Features

### 🔍 Object Detection

* **Model**: Faster R-CNN with a ResNet-50 backbone

### 🧩 Segmentation Techniques

* K-means Clustering
* GrabCut
* Superpixel Segmentation (SLIC)
* Watershed Segmentation
* U-Net (Deep Learning-based segmentation)

### 📦 Datasets Used

* COCO-C
* Pascal-C
* Cityscapes-C

### 🧠 Explainability & Fairness Analysis

* **Explainable AI**: SHAP & LIME
* **Fairness Metrics**:

  * Demographic (Geographic) Parity
  * Equalized / Levelled Odds
  * Counterfactual Fairness

---

## 🔬 Methodology

### 📁 Dataset Preparation

* Use corrupted dataset variants (COCO-C, Pascal-C, Cityscapes-C) to simulate adverse conditions.

### 🛠 Object Detection & Segmentation

* Train and test Faster R-CNN for object detection.
* Evaluate segmentation using:

  * Classical methods (e.g., GrabCut, Watershed)
  * Deep learning (U-Net) for performance in fog/snow.

### ⚖️ Bias Detection & Mitigation

* Apply fairness metrics to detect inequity in predictions.
* Use SHAP and LIME to explain and adjust biased behavior.

### 📊 Evaluation & Interpretation

* Measure:

  * Detection accuracy in adverse conditions.
  * Segmentation precision.
  * Bias mitigation success.
* Visualize model decisions to ensure transparency.

---

## ➡️ Results & Impact

This project demonstrates:

* 📈 **Improved detection**: 89% accuracy under snow and fog using Faster R-CNN
* 🔍 **Best segmentation**: U-Net performs strongest under harsh conditions
* ⚖️ **Fairness gains**: 10–15% bias reduction using synthetic augmentation and fairness constraints
* 🧠 **Interpretability**: SHAP/LIME explained 85–90% of critical predictions

These contributions promote **safer and more equitable autonomous navigation systems**, especially important for **developing regions and global AI adoption**.

---

## ⚙️ Installation & Requirements

### 📦 Prerequisites

* Python 3.x
* TensorFlow or PyTorch
* OpenCV
* NumPy, Pandas, Matplotlib
* SHAP & LIME

### 📥 Installation

```bash
pip install tensorflow torch torchvision opencv-python numpy pandas matplotlib shap lime
```

---

## 🚀 Usage

### 🔧 Train Object Detection Model

```bash
python train_detector.py --dataset COCO-C
```

### 🎨 Perform Segmentation Analysis

```bash
python segment_images.py --method U-Net
```

### 📏 Run Fairness Evaluation

```bash
python evaluate_fairness.py --metric Levelled_Odds
```

---

## 🙌 Acknowledgments

Special thanks to the open-source communities and contributors behind:

* COCO, Pascal, and Cityscapes datasets
* SHAP, LIME, and Fairlearn libraries
* Researchers advancing ethical AI for real-world use

---

## 👩‍💻 Author

**K. Anusha**
A passionate advocate for ethical, robust, and human-centered AI systems.
📫 Connect: [GitHub](https://github.com/kothapallianusha987)
📧 Email: [kothapallianusha987@gmail.com](mailto:kothapallianusha987@gmail.com)

---

## 🤝 Open for Collaboration

This project is open to contributors who are excited about:

* Ethical AI
* Autonomous systems
* Computer vision
* Social good through technology

Let’s build together — a future where AI not only drives vehicles but also drives equity, safety, and progress for every nation, especially developing countries.

> *“Together, we can engineer intelligence that empowers — not divides.”*
