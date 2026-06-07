# Single-Object Classification and Localization Benchmark

A comprehensive research guide and architectural blueprint for concurrent image classification and bounding box regression, utilizing the Oxford-IIIT Pet Dataset.

## 1. Project Motivation & Experimental Design

Object detection pipelines often rely on region proposals and clustering filters like Non-Maximum Suppression (NMS). These mechanisms introduce algorithmic overhead that can obscure the direct analysis of a convolutional backbone's true feature-extraction performance.

This project isolates the environment by ensuring every image contains exactly one primary object. The network is constrained to output two simultaneous predictions per forward pass:

* A discrete class distribution vector (pet breed).
* A continuous geometric coordinate vector (bounding box).

This multi-task optimization framework systematically evaluates how different feature extraction strategies and inductive biases handle the conflicting gradient demands of classification and spatial localization.

---

## 2. Comparative Model Architectures

To establish a thorough baseline, the project evaluates three distinct architectures to isolate feature extraction efficiency:

* **ResNet34 (~21.8M Parameters):** Serves as the primary heavy-duty baseline, utilizing standard residual connections to build deep feature hierarchies and discover macro-geometric shapes.
* **EfficientNet-B0 (~5.3M Parameters):** Represents structured scaling. It optimizes computational efficiency via depthwise separable convolutions to test if sparse architectures can preserve spatial localization.
* **MobileNetV3-Small (~2.5M Parameters):** An extremely lightweight architecture designed for strict mobile/edge hardware constraints, testing the absolute lower bounds of parameter capacity for multi-task learning.

---

## 3. ResNet34 and the Residual Unit

The defining mechanism of ResNet34 is the Residual Unit, which bypasses the standard sequential flow of data to prevent the vanishing gradient problem in deep networks.

Instead of forcing a stacked layer to learn an underlying mapping $h(x)$, the network explicitly lets the layers fit a residual mapping $f(x)=h(x)-x$. A shortcut connection directly adds the input $x$ to the output of the stacked layers. This enables the safe backpropagation of strong gradient signals across 34 layers, allowing for the discovery of fine-grained visual patterns required for accurate bounding box regression.

<img width="263" height="191" alt="image" src="https://github.com/user-attachments/assets/a2816712-8d58-4cea-a6e6-cc755cbc6131" />



---

## 4. Dataset Pipeline & Input Augmentation

The Oxford-IIIT Pet Dataset provides the foundation, featuring 37 distinct breeds of cats and dogs paired with absolute pixel coordinates defining the animal's head region.

To enable uniform GPU batch processing, images were scaled to 224x224 pixels, inherently forcing the network to downsample inputs into a 7x7 global pooling feature matrix. To prevent the model from memorizing the training set, specific augmentations were applied:

* **Random Grayscale & ColorJitter:** Forces the model to learn the structural and geometric shape of the pet rather than memorizing specific fur colors or lighting biases.
* **Blur:** Simulates out-of-focus captures, preventing the network from hyper-focusing on sharp, high-frequency background artifacts.

---

## 5. Unified Multi-Task Architecture

To ensure a fair benchmark, a dynamic `UnifiedMultiTaskModel` class was engineered. This allows seamless swapping of the convolutional backbone while keeping the custom classification and localization heads strictly consistent across all experiments.

```python
import torch
import torch.nn as nn
import torchvision

class UnifiedMultiTaskModel(nn.Module):
    def __init__(self, backbone_type="efficientnet", num_classes=37):
        super().__init__()
        backbone_type = backbone_type.lower()

        # 1. Dynamically select the backbone and map its final output channel count
        if backbone_type == "efficientnet":
            base_model = torchvision.models.efficientnet_b0(
                weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
            )
            self.feature_extractor = base_model.features
            backbone_channels = 1280

        elif backbone_type == "mobilenet":
            base_model = torchvision.models.mobilenet_v3_small(
                weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            )
            self.feature_extractor = base_model.features
            backbone_channels = 576
       elif backbone_type == "resnet":
            base_model = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.DEFAULT
            )
            # Custom sequential to keep the 7x7 grid context for ResNet
            self.feature_extractor = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4
            )
            backbone_channels = 512
            
        else:
            raise ValueError(f"Backbone '{backbone_type}' is not supported.")

        # 2. Classification Path Layers
        self.class_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.class_flatten = nn.Flatten()
        self.classification_head = nn.Linear(in_features=backbone_channels, out_features=num_classes)

        # 3. Localization Path Layers (Preserving spatial grid context)
        self.loc_conv = nn.Sequential(
            nn.Conv2d(in_channels=backbone_channels, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.loc_flatten = nn.Flatten()
        self.localization_head = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=4),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # Shared feature extraction
        spatial_features = self.feature_extractor(inputs)

        # Path A: Classification (Collapse spatial maps to 1x1)
        class_features = self.class_flatten(self.class_pool(spatial_features))
        class_logits = self.classification_head(class_features)

        # Path B: Localization (Flatten the grid to track coordinate locations)
        loc_features = self.loc_conv(spatial_features)
        loc_features = self.loc_flatten(loc_features)
        bbox_coords = self.localization_head(loc_features)

        return class_logits, bbox_coords
```

---

## 6. Pretraining & Weight Initialization Tactics

Training deep networks from scratch on this dataset resulted in rapid overfitting. Pretrained backbones (ImageNet-1k) were adopted to inherit pre-developed feature hierarchies and improve convergence.

However, the newly initialized parallel dense layers required careful setup. **Kaiming He initialization** was implemented for the custom heads. Standard initialization often fails with ReLU activations, which aggressively zero out negative values. To maintain a stable variance of activations across deep layers, the weights are drawn from a distribution with a variance mathematically defined as:

$$Var(W)=\frac{2}{n_{in}}$$

---

## 7. Optimization & Learning Rate Scheduling

* **AdamW Optimizer:** Standard Adam couples weight decay with gradient updates. AdamW decouples the weight decay penalty, leading to superior generalization on unseen validation data.
* **ReduceLROnPlateau Scheduler:** This scheduler dynamically monitors the validation metric and reduces the learning rate when progress stalls. This allows the optimizer to settle into finer local minima given the conflicting gradient demands of classification vs. localization.

---

## 8. The Two-Stage Multi-Task Loss Criterion

CrossEntropyLoss governed the discrete classification task. For the continuous bounding box geometry, a strategic two-stage loss pipeline was implemented:

### Phase 1: Smooth L1 Loss (Epochs 1-15)

Initially, coordinate regression gradients are highly volatile. Mean Squared Error (MSE) treats dimensions independently and lacks geometric context. Smooth L1 provides a steady, bounded gradient, allowing the classification head to learn without being disrupted by the localization head's early chaotic updates.

### Phase 2: Generalized Intersection over Union (GIoU) (Epochs 16+)

Standard Intersection over Union (IoU) cannot serve as a raw loss function due to the vanishing gradient problem if predicted and target boxes have zero overlap. GIoU introduces $C$, the absolute smallest convex hull that completely wraps both boxes. This ensures a continuous gradient signal across the 2D plane:

$$IoU=\frac{|A\cap B|}{|A\cup B|}$$

$$GIoU=IoU-\frac{|C\setminus(A\cup B)|}{|C|}$$

$$L_{GIoU}=1-GIoU$$

---

## 9. Evaluation Metrics

To rigorously evaluate performance, three core benchmarks are utilized:

* **Classification Accuracy:** Percentage of correct discrete breed classifications.
* **mAP@50:** Average precision where the predicted breed is correct AND the bounding box achieves an Intersection over Union (IoU) of at least 50%.
* **mAP@75:** A significantly stricter spatial localization threshold requiring 75% geometric overlap.

---

## 10. Validation Results & Analysis

| Model Architecture | Classification Accuracy | Bounding Box mAP@50 | Bounding Box mAP@75 |
| --- | --- | --- | --- |
| **ResNet34** | 91.29% | 86.64% | 51.58% |
| **EfficientNet-B0** | 90.61% | 79.82% | 26.80% |
| **MobileNetV3-Small** | 77.69% | 68.20% | 27.05% |

**Analysis:** ResNet34 exhibits superior spatial awareness, decisively dominating the strict mAP@75 metric. EfficientNet-B0 is highly competitive in raw classification but struggles with precise boundary regression, indicating that sparse depthwise convolutions may discard critical spatial resolution. MobileNetV3 trades raw precision for parameter efficiency, lacking the capacity to successfully balance both tasks.

---

## 11. Addressing Overfitting Dynamics

As observed in the training logs, the transition to GIoU at epoch 16 triggered a significant shift in the loss scale. While this forced the model to learn true geometric proportions, a divergence between training and validation loss became apparent around epoch 25. Under the strict GIoU metric, the model began memorizing the specific spatial layouts of the training set rather than learning generalizable boundaries.

**Future Mitigation Strategies:**

* **Dataset Expansion:** Utilizing more training examples to force the model to learn universally applicable features.
* **Dynamic Scalar Weighting:** Adjusting the scalar weight hyperparameter to better balance the classification and GIoU loss gradients.
* **Stricter Regularization:** Introducing Layer Normalization before the task heads or increasing dropout rates.
* **Advanced Geometric Augmentation:** Introducing CutMix, rotation, or shearing to force the network to generalize under heavy distortion.
* **Strict Early Stopping:** Halting the training loop precisely at epoch 25 before validation degradation begins.
