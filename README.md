# PPE Detection with YOLO and Pose Estimation

This project focused on detecting **Personal Protective Equipment (PPE)** such as helmets, masks, and safety vests.  
The system combines **YOLO-based object detection** with **pose estimation** to validate PPE usage on humans in images and videos.

---

## Project Structure
- `train_ppe_model.ipynb` → Jupyter Notebook for training a YOLO model on a custom PPE dataset.  
- `ppe_detection_pipeline.ipynb` → Jupyter Notebook for running the detection pipeline on **images** or **videos**, annotating detections, and filtering results using human pose keypoints.

---

## Usage

### 1. Train the PPE model
Open **`train_ppe_model.ipynb`** in Jupyter Notebook or JupyterLab, run all cells, and train the model.

### 2. Run the detection pipeline
Open **`ppe_detection_pipeline.ipynb`**, set the path to your input image or video, and run all cells.

---

## Models
- **YOLOv11x** → fine-tuned on a custom PPE dataset with **13 classes**.  
- **YOLO11-Pose-n** → used to detect humans and pose keypoints (head, shoulders, torso, etc.).  
 
### Trained PPE classes (13 total):
```python
ppe_classes_all = [
    'Gloves',
    'Goggles',
    'Helmet',
    'Mask',
    'NO-Goggles',
    'NO-helmet',
    'NO-mask',
    'NO-safety Vest',
    'Work clothes',
    'hand',
    'safety vest'
]

```
 Note: In the final detection pipeline, only 6 core PPE-related classes were used for evaluation and visualization:
- Helmet
- Mask
- Safety Vest
- NO-helmet
- NO-mask
- NO-safety Vest

## Training Environment

- Trained on **an NVIDIA A100 GPU**.

- Framework: Ultralytics **YOLO11** .

- Training performed using Jupyter Notebook.

---

## YOLO Training Results
YOLO automatically logs metrics during training. Below are the sample plots:

### Training Loss
![Loss Curve](docs/loss_curve.png)

### Precision, Recall, and mAP
![Metrics Curve](docs/metrics_curve.png)

### Final Evaluation (Test Set)
```text
  Class               Images   Instances    Box(P       R        mAP50     mAP50-95):
  all                  163       1882      0.818      0.743      0.797      0.598
  Gloves               32         73       0.78       0.493      0.601      0.484
  Goggles              37         54       0.705      0.426      0.565      0.4
  Helmet               134        317      0.955      0.943      0.983      0.936
  Mask                 28         168      0.889      0.917      0.932      0.501
  NO-Goggles           113        229      0.799      0.887      0.879      0.639
  NO-helmet            41         234      0.714      0.675      0.68       0.339
  NO-mask              132        265      0.955      0.951      0.982      0.867
  NO-safety Vest       44         99       0.662      0.596      0.667      0.423
  hand                 93         205      0.802      0.751      0.784      0.58
  safety vest          113        238      0.922      0.789      0.897      0.812
```

---

## Sample Outputs

Image Example

Video Example


---

##  Features

- Detects whether PPE is correctly worn.

- Resolves conflicts between **positive** and **negative** classes (e.g., Helmet vs NO-helmet).

- Filters false detections by validating bounding boxes against **pose keypoints**.

- Supports **images**, **videos**, and **directories** as input.

--- 

## Future Work

- Improve dataset with more PPE types (Gloves, Goggles, etc.).

- Optimize inference speed for real-time detection.

- Develop a simple GUI or web app for easier usage.
---
Developed by Marjan Norouzi