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
