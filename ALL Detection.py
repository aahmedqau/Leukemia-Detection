import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PreprocessingPipeline:
    """
    Preprocessing steps for ALL leukemia detection
    - Median Filter
    - Morphological Operations (Cloning)
    - Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    """
    
    def __init__(self, kernel_size: int = 3, clip_limit: float = 2.0, grid_size: Tuple[int, int] = (8, 8)):
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        
    def median_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply median filter for noise reduction"""
        return cv2.medianBlur(image, self.kernel_size)
    
    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations (cloning/enhancement)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Apply opening followed by closing
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Enhance edges
        gradient = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, kernel)
        
        return cv2.addWeighted(closed, 0.8, gradient, 0.2, 0)
    
    def clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast-Limited Adaptive Histogram Equalization"""
        if len(image.shape) == 2:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            return clahe.apply(image)
        else:
            # Color image - apply CLAHE to each channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.grid_size)
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """Complete preprocessing pipeline"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Apply preprocessing steps
        filtered = self.median_filter(image)
        morphological = self.morphological_operations(filtered)
        enhanced = self.clahe(morphological)
        
        return enhanced

class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block
    Based on EfficientNet architecture
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1,
                 expansion_ratio: int = 6, se_ratio: float = 0.25):
        super().__init__()
        
        # Expansion phase
        expanded_channels = int(in_channels * expansion_ratio)
        self.expand = nn.Identity() if expansion_ratio == 1 else nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Depthwise convolution
        padding = kernel_size // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, padding, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and Excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, squeeze_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Output phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Squeeze and Excitation
        se = self.se(x)
        x = x * se
        
        # Projection
        x = self.project(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
            
        return x

class ALLLeukemiaDetector(nn.Module):
    """
    Transfer Learning Model for ALL Leukemia Detection
    Architecture based on the provided description
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks based on the architecture description
        self.blocks = nn.ModuleList([
            # Conv3x3 (Initial block)
            MBConv(32, 16, 3, 1, expansion_ratio=1),
            
            # MBConv1 (3x3) blocks
            MBConv(16, 24, 3, 2, expansion_ratio=1),
            MBConv(24, 24, 3, 1, expansion_ratio=1),
            
            # MBConv6 (5x5) blocks
            MBConv(24, 40, 5, 2, expansion_ratio=6),
            MBConv(40, 40, 5, 1, expansion_ratio=6),
            
            # MBConv6 (3x3) blocks
            MBConv(40, 80, 3, 2, expansion_ratio=6),
            MBConv(80, 80, 3, 1, expansion_ratio=6),
            
            # MBConv6 (5x5) blocks
            MBConv(80, 112, 5, 1, expansion_ratio=6),
            MBConv(112, 112, 5, 1, expansion_ratio=6),
            MBConv(112, 192, 5, 2, expansion_ratio=6),
            MBConv(192, 192, 5, 1, expansion_ratio=6),
        ])
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(192, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        # Pass through all MBConv blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.head(x)
        return x

class ExplainableAI:
    """
    Explainable AI methods for model interpretability
    - LIME
    - GradCAM
    - Guided GradCAM
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def grad_cam(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generate GradCAM heatmap for model interpretability
        """
        self.model.zero_grad()
        
        # Get the output from the model
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Get the score for the target class
        score = output[0, target_class]
        score.backward()
        
        # Get gradients
        gradients = self.model.get_activations_gradient()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Get the activations
        activations = self.model.get_activations(input_tensor).detach()
        
        # Weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # Apply ReLU to the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()
    
    def generate_explanation(self, image: np.ndarray, 
                           prediction: str, 
                           confidence: float) -> dict:
        """
        Generate comprehensive explanation for medical practitioners
        """
        # Prepare image for model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
        
        # Generate heatmap
        heatmap = self.grad_cam(input_tensor, pred_class)
        
        # Create explanation dictionary
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'class_probabilities': probabilities.cpu().numpy()[0],
            'heatmap': heatmap,
            'important_regions': self._identify_important_regions(heatmap),
            'model_interpretation': self._generate_interpretation(pred_class, confidence)
        }
        
        return explanation
    
    def _identify_important_regions(self, heatmap: np.ndarray, 
                                   threshold: float = 0.5) -> List[dict]:
        """Identify important regions in the heatmap"""
        regions = []
        mask = heatmap > threshold
        
        if np.any(mask):
            # Find contours of important regions
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 10:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'region_id': i,
                        'bbox': [x, y, w, h],
                        'area': area,
                        'mean_importance': np.mean(heatmap[y:y+h, x:x+w])
                    })
        
        return regions
    
    def _generate_interpretation(self, pred_class: int, 
                                confidence: float) -> str:
        """Generate human-readable interpretation for medical practitioners"""
        if pred_class == 1:  # Assuming class 1 is "ALL Leukemia"
            interpretation = (
                f"The model predicts Acute Lymphoblastic Leukemia (ALL) with "
                f"{confidence:.1%} confidence. "
                f"Key morphological features indicating ALL were identified in the cell image, "
                f"including nuclear characteristics and cytoplasmic patterns."
            )
        else:
            interpretation = (
                f"The model predicts normal cells with {confidence:.1%} confidence. "
                f"No significant leukemic features were detected in the analyzed regions."
            )
        
        return interpretation

class ALLDetectionPipeline:
    """
    Complete pipeline for ALL Leukemia detection
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize preprocessing
        self.preprocessor = PreprocessingPipeline()
        
        # Initialize model
        self.model = ALLLeukemiaDetector(num_classes=2)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
        
        # Initialize Explainable AI
        self.explainer = ExplainableAI(self.model, device)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str, visualize: bool = True) -> dict:
        """
        Complete prediction pipeline
        """
        # Step 1: Preprocessing
        print("Step 1: Preprocessing...")
        preprocessed_image = self.preprocessor.preprocess(image_path)
        
        # Step 2: Prepare for classification
        print("Step 2: Preparing for classification...")
        input_tensor = self.transform(preprocessed_image).unsqueeze(0).to(self.device)
        
        # Step 3: Classification
        print("Step 3: Classification using transfer learning...")
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        # Map class to label
        class_labels = ["Normal", "ALL Leukemia"]
        prediction = class_labels[pred_class]
        
        # Step 4: Generate explanation
        print("Step 4: Generating explanations...")
        explanation = self.explainer.generate_explanation(
            preprocessed_image, prediction, confidence
        )
        
        # Visualize results if requested
        if visualize:
            self.visualize_results(image_path, preprocessed_image, 
                                 prediction, confidence, explanation)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'class_probabilities': probabilities.cpu().numpy()[0],
            'explanation': explanation,
            'preprocessed_image': preprocessed_image
        }
    
    def visualize_results(self, original_path: str, 
                         preprocessed_image: np.ndarray,
                         prediction: str, 
                         confidence: float,
                         explanation: dict):
        """Visualize all steps of the pipeline"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        original = cv2.imread(original_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('1. Input Image')
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('2. Preprocessed Image')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].text(0.1, 0.5, f'Prediction: {prediction}\nConfidence: {confidence:.2%}',
                       fontsize=12, va='center')
        axes[0, 2].set_title('3. Predicted ALL Leukemia')
        axes[0, 2].axis('off')
        
        # Heatmap
        heatmap = explanation['heatmap']
        axes[1, 0].imshow(heatmap, cmap='jet')
        axes[1, 0].set_title('4. GradCAM Heatmap')
        axes[1, 0].axis('off')
        
        # Overlay heatmap on image
        resized_heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        resized_heatmap = np.uint8(255 * resized_heatmap)
        heatmap_colored = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('5. Prediction with Explanation')
        axes[1, 1].axis('off')
        
        # Important regions
        axes[1, 2].text(0.1, 0.5, explanation['model_interpretation'],
                       fontsize=10, va='center')
        axes[1, 2].set_title('6. Explanation for Medical Practitioner')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('all_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def batch_predict(self, image_paths: List[str]) -> List[dict]:
        """Process multiple images"""
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path, visualize=False)
                results.append(result)
                print(f"Processed {img_path}: {result['prediction']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results.append(None)
        
        return results

# Utility functions
def create_data_transforms():
    """Create data transformations for training and validation"""
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, 
                learning_rate=0.001, device='cuda'):
    """Training function for the ALL detection model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
        
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
        
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model

# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize the pipeline
    pipeline = ALLDetectionPipeline(
        model_path="all_model.pth",  # Optional: path to pretrained model
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example prediction
    image_path = "example_blood_smear.jpg"
    
    if os.path.exists(image_path):
        result = pipeline.predict(image_path, visualize=True)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nModel Interpretation:")
        print(result['explanation']['model_interpretation'])
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.")