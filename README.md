# Equitable AI on the Road: Robust Object Detection and Segmentation Under Adverse Conditions with Bias Mitigation

## Overview
The rise of autonomous vehicles has underscored the need for robust and equitable object detection systems that ensure safe navigation, especially under adverse weather conditions. These systems often struggle in challenging scenarios such as fog, frost, and snow, where visibility is compromised. This project enhances object detection and segmentation models to improve their resilience in difficult environments while simultaneously addressing fairness concerns.

## Features
- **Object Detection Model:** Faster R-CNN with a ResNet-50 backbone
- **Segmentation Techniques:**
  - K-means Clustering
  - GrabCut
  - Superpixel Segmentation (SLIC)
  - Watershed Segmentation
  - U-Net (Deep Learning-based Segmentation)
- **Datasets Used:**
  - COCO-C
  - Pascal-C
  - Cityscapes-C
- **Explainability & Fairness Analysis:**
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (Shapley Additive Explanations)
  - Fairness Metrics: Geographic Parity, Levelled Odds, and Counterfactual Fairness

## Methodology
1. **Dataset Preparation:**
   - Use corrupted versions of standard datasets (COCO-C, Pascal-C, Cityscapes-C) to simulate adverse weather conditions.
2. **Object Detection & Segmentation:**
   - Implement Faster R-CNN for object detection.
   - Apply five different segmentation techniques for refining object recognition.
3. **Bias Detection & Mitigation:**
   - Evaluate model fairness using fairness metrics.
   - Use SHAP and LIME to interpret and mitigate bias.
4. **Evaluation & Interpretation:**
   - Analyze segmentation performance and detection reliability.
   - Improve model transparency and decision confidence.

## Results & Impact
This project contributes to the development of:
- More **reliable** object detection systems for autonomous vehicles in adverse weather conditions.
- **Fairer AI models** that ensure equitable performance across demographic groups.
- Increased **trust and transparency** in AI-driven decision-making.

## Installation & Requirements
### Prerequisites
- Python 3.x
- TensorFlow / PyTorch
- OpenCV
- NumPy, Pandas, Matplotlib
- SHAP & LIME

### Installation
```bash
pip install tensorflow torch torchvision opencv-python numpy pandas matplotlib shap lime
```

## Usage
1. **Train Object Detection Model:**
   ```bash
   python train_detector.py --dataset COCO-C
   ```
2. **Perform Segmentation Analysis:**
   ```bash
   python segment_images.py --method U-Net
   ```
3. **Run Fairness Evaluation:**
   ```bash
   python evaluate_fairness.py --metric Levelled_Odds
   ```





## Acknowledgments
Special thanks to the open-source communities and datasets (COCO, Pascal, Cityscapes) that made this research possible.
