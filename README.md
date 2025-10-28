# **Section 1: Introduction to the Problem/Task**
## **1.1 Problem Statement**

Ensuring the structural integrity of welded joints is a critical aspect of industrial manufacturing and construction, as even minor flaws in welds can compromise safety, durability, and overall performance. Traditional manual inspection methods are often time-consuming, subjective, and prone to human error, highlighting the need for automated solutions that can accurately classify weld quality.

## **1.2 Task Definition**

This project focuses on developing a deep learning-based approach using the **YOLOv12 object detection framework** to automatically detect and segment critical weld features from image data. The model is trained to identify four key categories:

* **Welding Line** - Represents the seam continuity
* **Porosity** - Indicates trapped gas voids or bubbles within the weld
* **Spatters** - Scattered molten droplets around the weld area
* **Cracks** - Critical discontinuities that threaten structural integrity

## **1.3 Real-World Significance**

The implementation of automated weld classification has significant implications for:

* **Productivity and Cost-Efficiency** - Automated systems provide rapid and consistent evaluations across large volumes of welds
* **Real-Time Monitoring** - Enables quality control within production pipelines
* **Enhanced Reliability** - Deep learning models can detect subtle variations not visible to the human eye
* **Safety Compliance** - Reduces risks of structural failure and improves compliance with engineering safety standards

## **1.4 Project Objectives**

1. Explore advanced annotation techniques (mask-based segmentation) that comprehensively cover welded metal and surrounding areas
2. Build predictive models using the YOLOv12 deep learning algorithm
3. Analyze and interpret detection outputs to evaluate weld integrity
4. Deploy the model via an interactive web application for real-world usability

# **Section 2: Dataset Description**
## **2.1 Dataset Overview**

The weld quality dataset used in this study comprises **17,063 images** in total, generated through dataset augmentation (x3) from an initial collection of **7,109 images**. The dataset was specifically curated for training deep learning models to detect and segment weld defects in industrial applications.

**Dataset Split:**

| Set | Count | Percentage |
|-----|-------|-----------|
| Train | 14,931 | 87.5% |
| Validation | 1,066 | 6.25% |
| Test | 1,066 | 6.25% |
| **Total** | **17,063** | **100%** |

## **2.2 Data Source and Collection**

* **Source:** Kaggle Welding Defect Dataset (adapted and enhanced)
* **Collection Method:** Images captured from industrial welding environments under varied lighting and angle conditions
* **Annotation Tool:** Roboflow (for consistent labeling and segmentation)
* **Annotation Format:** YOLO format with polygon-based segmentation masks

## **2.3 Class Distribution**

The dataset exhibits noticeable class imbalance, with spatters and porosity being more prevalent than cracks:

| Class | Instance Count | Percentage |
|-------|---------------|-----------|
| Spatters | 47,600 | 50.1% |
| Porosity | 32,136 | 33.8% |
| Welding Line | 12,153 | 12.8% |
| Cracks | 3,084 | 3.3% |
| **Total** | **94,973** | **100%** |

**Key Observations:**
* Spatters represent the largest portion of annotated instances
* Cracks are relatively scarce, presenting a class imbalance challenge
* Each class provides distinct and visually meaningful examples for model training

## **2.4 Preprocessing Steps**

Three critical preprocessing transformations were applied using Roboflow:

### **2.4.1 Auto-Orient**
* Standardizes image orientation across all samples
* Ensures consistent positional alignment of welds
* Prevents misinterpretation due to varying camera angles

### **2.4.2 Resize (640×640)**
* Fixed dimensions required by YOLOv12 architecture
* Balances computational efficiency and feature resolution
* Retains fine-grained details (cracks, spatters) while reducing memory consumption

### **2.4.3 Auto-Adjust Contrast (Adaptive Equalization)**
* Enhances visibility in blurred or poorly lit images
* Redistributes pixel intensity values in localized regions
* Makes subtle features (surface irregularities, small cracks) more distinguishable
* Mitigates impact of uneven lighting conditions

## **2.5 Data Augmentation Strategy**

To improve model robustness and generalization, the following augmentations were applied:

| Augmentation | Range/Value | Purpose |
|--------------|-------------|---------|
| Output per Training | 3x | Triple dataset size |
| Rotation | -15° to +15° | Simulate camera angle variations |
| Brightness | 0% to +15% | Handle lighting inconsistencies |
| Exposure | -15% to +15% | Adapt to under/overexposed conditions |
| Blur | Up to 1.5px | Handle out-of-focus images |
| Noise | Up to 0.1% pixels | Prevent overfitting to clean samples |
| Shear | ±15° H/V | Simulate perspective distortions |

**Impact:** These augmentations reduce overfitting and enhance the model's ability to generalize by simulating real-world variations in weld imaging conditions.

# **Section 3: Requirements and Dependencies**
## **3.1 Hardware Requirements**

* **GPU:** NVIDIA A100 80GB (or equivalent CUDA-compatible GPU)
* **RAM:** Minimum 16GB system memory
* **Storage:** At least 10GB for dataset and model checkpoints

## **3.2 Software Environment**

* **Platform:** Google Colab Pro (with GPU runtime)
* **Python Version:** 3.12+
* **CUDA:** 12.6

## **3.3 Core Libraries**

### **Deep Learning Frameworks**
* `torch==2.2.2` - PyTorch deep learning framework
* `torchvision==0.17.2` - Computer vision utilities
* `ultralytics==8.3.176` - YOLOv12 implementation

### **Computer Vision**
* `opencv-python==4.9.0.80` - Image processing
* `albumentations==2.0.4` - Advanced data augmentation
* `supervision==0.22.0` - Vision AI utilities

### **Model Optimization**
* `onnx==1.16.2` - Model export and optimization
* `onnxslim==0.1.31` - ONNX model compression
* `onnxruntime-gpu==1.18.0` - GPU-accelerated inference

### **Scientific Computing**
* `numpy==1.26.4` - Numerical operations
* `scipy==1.13.0` - Scientific computing
* `pandas>=1.1.4` - Data manipulation
* `matplotlib>=3.3.0` - Visualization
* `seaborn>=0.11.0` - Statistical visualization

### **Utilities**
* `PyYAML==6.0.1` - Configuration files
* `tqdm>=4.64.0` - Progress bars
* `psutil==5.9.8` - System monitoring
* `py-cpuinfo==9.0.0` - CPU information

## **3.4 Web Deployment**
* `streamlit` - Interactive web application framework -> https://css181-2-deep-learning-project-guk8t4cy3kjzwzmrhctcr5.streamlit.app/
* `gradio==4.44.1` - Alternative UI framework

## **3.5 Installation Commands**

All dependencies are installed automatically in the notebook setup cells. Key installation steps include:
```python
# Install core dependencies
!pip install -r requirements.txt

# Install YOLOv12
!pip install -e .

# Upgrade Ultralytics
!pip install -U ultralytics==8.3.176
```

**Get the dataset here:** https://drive.google.com/drive/folders/1zdEEtExt9etOzJGcaz9hV5lINFtqiDVY?usp=sharing
