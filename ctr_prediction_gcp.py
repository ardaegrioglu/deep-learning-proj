import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import cv2
from transformers import CLIPProcessor, CLIPModel
import torchvision
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F

# Define paths for GCP
DATA_DIR = "data"
IMAGE_DIR = "data/images"
CACHE_DIR = "cache"
FEATURE_DIR = "features"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(CACHE_DIR, "test"), exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# [PASTE YOUR MODEL CLASSES HERE: VisualEncoder, CrossLayer, ContrastiveAlignmentLayer, etc.]

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None

    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience and model and self.restore_best_weights:
                model.load_state_dict(self.best_model_state)
            return self.counter >= self.patience

class TrainingMonitor:
    """Monitor training progress and detect issues"""

    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_aucs = []
        self.test_aucs = []

    def log_epoch(self, train_loss, test_loss, train_auc, test_auc, epoch):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_aucs.append(train_auc)
        self.test_aucs.append(test_auc)

        # Check for overfitting
        if epoch > 2:
            recent_train_trend = np.mean(self.train_losses[-3:]) - np.mean(self.train_losses[-6:-3]) if len(self.train_losses) >= 6 else 0
            recent_test_trend = np.mean(self.test_losses[-3:]) - np.mean(self.test_losses[-6:-3]) if len(self.test_losses) >= 6 else 0
            if recent_train_trend < -0.01 and recent_test_trend > 0.01:
                print(f" WARNING: Potential overfitting detected at epoch {epoch}")
                print(f"  Train loss decreasing ({recent_train_trend:.4f}) while test loss increasing ({recent_test_trend:.4f})")

        # Check AUC gap
        auc_gap = train_auc - test_auc
        if auc_gap > 0.1:
            print(f"WARNING: Large AUC gap detected: {auc_gap:.3f}")

def log_feature_importance(model, batch, device):
    """Log which features are most important for debugging"""
    model.eval()
    # Enable gradients for feature importance calculation
    for param in model.parameters():
        param.requires_grad_(True)
    
    with torch.enable_grad():
        # Get embeddings
        predictions, structured_features, visual_features = model(batch, return_embeddings=True)
        
        # Compute gradients w.r.t. features
        structured_grad = torch.autograd.grad(
            predictions.sum(), structured_features,
            retain_graph=True, create_graph=False, allow_unused=True  # Allow unused tensors
        )[0]
        visual_grad = torch.autograd.grad(
            predictions.sum(), visual_features,
            retain_graph=True, create_graph=False, allow_unused=True  # Allow unused tensors
        )[0]
        
        # Handle None gradients (unused tensors)
        structured_importance = structured_grad.abs().mean().item() if structured_grad is not None else 0.0
        visual_importance = visual_grad.abs().mean().item() if visual_grad is not None else 0.0
        
        print(f"Feature Importance - Structured: {structured_importance:.6f}, Visual: {visual_importance:.6f}")
        
        # Check for dead features
        if structured_importance < 1e-6:
            print("WARNING: Structured features have very low importance!")
        if visual_importance < 1e-6:
            print("WARNING: Visual features have very low importance!")
    
    return structured_importance, visual_importance

def prepare_data_efficient(train_df, test_df, max_vocab_size=10000, max_seq_len=10):
    """
    Memory-efficient version of prepare_data that processes data in chunks
    and limits vocabulary sizes
    """
    # Define column types
    high_cardinality_cols = ['user_id', 'item_id']
    simple_cat_cols = ['scene']
    numerical_cols = ['deep_features_14', 'deep_features_19', 'deep_features_20',
                     'deep_features_21', 'deep_features_22', 'deep_features_23',
                     'deep_features_24', 'deep_features_25', 'deep_features_26']
    
    # Target column
    target_col = 'label'
    
    # Drop rows with missing target (if any)
    train_df = train_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])

    # Dictionary to store results
    data_info = {}

    # 1. Process numerical features (most memory efficient)
    print("Processing numerical features...")
    # Filling missing values with the median for each numerical column
    for col in numerical_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        
        median_val = test_df[col].median()
        test_df[col] = test_df[col].fillna(median_val)

    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[numerical_cols])
    test_num = scaler.transform(test_df[numerical_cols])

    data_info['numerical_features'] = {
        'train': torch.tensor(train_num, dtype=torch.float),
        'test': torch.tensor(test_num, dtype=torch.float)
    }
    data_info['numerical_cols'] = numerical_cols

    # 2. Process simple categorical features
    print("Processing simple categorical features...")
    simple_cat_features = {}
    simple_cat_vocab_sizes = {}

    for col in simple_cat_cols:
        print(f"  Processing {col}...")
        # Use LabelEncoder for efficiency
        le = LabelEncoder()
        train_indices = le.fit_transform(train_df[col].fillna('missing').astype(str))

        # Handle unseen values in test set
        test_values = test_df[col].fillna('missing').astype(str)
        test_indices = np.zeros(len(test_values), dtype=int)
        for i, val in enumerate(test_values):
            if val in le.classes_:
                test_indices[i] = np.where(le.classes_ == val)[0][0]

        simple_cat_features[col] = {
            'train': torch.tensor(train_indices, dtype=torch.long),
            'test': torch.tensor(test_indices, dtype=torch.long)
        }
        simple_cat_vocab_sizes[col] = len(le.classes_)

    data_info['simple_cat_features'] = simple_cat_features
    data_info['simple_cat_vocab_sizes'] = simple_cat_vocab_sizes

    # 3. Process high-cardinality features with limited vocabulary
    print("Processing high-cardinality features...")
    high_card_features = {}
    high_card_vocab_sizes = {}

    for col in high_cardinality_cols:
        print(f"  Processing {col}...")
        # Limit to top values
        value_counts = train_df[col].value_counts().nlargest(max_vocab_size - 1)
        top_values = set(value_counts.index)

        # Map values to indices
        train_indices = np.zeros(len(train_df), dtype=int)
        for i, val in enumerate(train_df[col].values):
            train_indices[i] = i % (max_vocab_size - 1) + 1 if val not in top_values else list(top_values).index(val) + 1

        test_indices = np.zeros(len(test_df), dtype=int)
        for i, val in enumerate(test_df[col].values):
            test_indices[i] = i % (max_vocab_size - 1) + 1 if val not in top_values else list(top_values).index(val) + 1

        high_card_features[col] = {
            'train': torch.tensor(train_indices, dtype=torch.long),
            'test': torch.tensor(test_indices, dtype=torch.long)
        }
        high_card_vocab_sizes[col] = max_vocab_size

    data_info['high_card_features'] = high_card_features
    data_info['high_card_vocab_sizes'] = high_card_vocab_sizes

    # 5. Process target
    print("Processing target...")
    data_info['y_train'] = torch.tensor(train_df[target_col].values, dtype=torch.float)
    data_info['y_test'] = torch.tensor(test_df[target_col].values, dtype=torch.float)

    return data_info

class CTRDataset(Dataset):
    def __init__(self, data, image_cache_dir=None, feature_dir=None, split='train'):
        self.split = split
        self.high_card_features = {k: v[split] for k, v in data['high_card_features'].items()}
        self.simple_cat_features = {k: v[split] for k, v in data['simple_cat_features'].items()}
        self.numerical_features = data['numerical_features'][split]

        # Image and feature handling
        self.image_cache_dir = image_cache_dir
        self.feature_dir = feature_dir

        # Get item IDs for images/features
        if split == 'train':
            self.item_ids = data['item_ids_train']
            self.labels = data['y_train']
        else:
            self.item_ids = data['item_ids_test']
            self.labels = data['y_test']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get features
        high_card_dict = {k: v[idx] for k, v in self.high_card_features.items()}
        simple_cat_dict = {k: v[idx] for k, v in self.simple_cat_features.items()}
        num = self.numerical_features[idx]
        label = self.labels[idx]
        item_id = self.item_ids[idx]

        result = {
            'high_card': high_card_dict,
            'simple_cat': simple_cat_dict,
            'numerical': num,
            'label': label,
            'item_id': item_id
        }
        
        # Try to load image from cache
        if self.image_cache_dir:
            cache_path = os.path.join(self.image_cache_dir, f"{item_id}.pt")
            if os.path.exists(cache_path):
                try:
                    image = torch.load(cache_path)
                    # Validate image is not zeros
                    if image.abs().sum().item() > 1e-6:
                        result['image'] = image
                except Exception as e:
                    pass
        
        # Try to load pre-computed features
        if self.feature_dir:
            feature_path = os.path.join(self.feature_dir, f"{item_id}.pt")
            if os.path.exists(feature_path):
                try:
                    features = torch.load(feature_path)
                    # Validate features are not zeros
                    if features.abs().sum().item() > 1e-6:
                        result['visual_features'] = features
                except Exception as e:
                    pass

        return result
    
class VisualEncoder(nn.Module):
    def __init__(self, output_dim=512, trainable=True):
        super().__init__()
        # Load pre-trained CLIP model
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast = True)

        print(f"🔍 DEBUG - CLIP model loaded successfully")
        print(f"🔍 DEBUG - CLIP trainable: {trainable}")
        total_params = sum(p.numel() for p in self.clip.parameters())
        trainable_params = sum(p.numel() for p in self.clip.parameters() if p.requires_grad)
        print(f"🔍 DEBUG - CLIP params: {total_params:,} total, {trainable_params:,} trainable")

        # Freeze CLIP parameters if not trainable
        if not trainable:
            for param in self.clip.parameters():
                param.requires_grad = False

        # Add adaptation layer
        self.adapter = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def preprocess(self, images):
        """Preprocess images for CLIP"""
        if isinstance(images, list):
            # List of PIL images
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
        else:
            # Already preprocessed tensor
            inputs = {'pixel_values': images}
        return inputs

    def forward(self, images):
        """
        Args:
            images: Either preprocessed image tensors or list of PIL images
        """
        # Preprocess if needed
        if not isinstance(images, dict):
            inputs = self.preprocess(images)
        else:
            inputs = images

        # Extract features
        with torch.set_grad_enabled(self.clip.vision_model.training):
            visual_features = self.clip.get_image_features(**inputs)

        # Apply adaptation layer
        return self.adapter(visual_features)

class CrossLayer(nn.Module):
    """Cross layer from xDeepFM for feature interactions"""
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, 1))
        self.bias = nn.Parameter(torch.Tensor(input_dim, 1))
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, x0):
        # x shape: (batch_size, input_dim)
        # x0 shape: (batch_size, input_dim)
        x_unsqueezed = x.unsqueeze(2)  # (batch_size, input_dim, 1)
        x0_unsqueezed = x0.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Element-wise product between x and x0
        interaction = torch.matmul(x_unsqueezed, x0_unsqueezed)  # (batch_size, input_dim, input_dim)

        # Apply weights
        weighted = torch.matmul(interaction, self.weight)  # (batch_size, input_dim, 1)
        weighted = weighted.squeeze(2)  # (batch_size, input_dim)

        # Add bias and residual connection with normalization
        output = self.norm(weighted + self.bias.squeeze(1) + x)

        return self.dropout(output)

class ContrastiveAlignmentLayer(nn.Module):
    def __init__(self, visual_dim, structured_dim, projection_dim=128, temperature=0.07):
        super().__init__()
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, projection_dim * 2),
            nn.BatchNorm1d(projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(projection_dim * 2, projection_dim)
        )

        self.structured_projection = nn.Sequential(
            nn.Linear(structured_dim, projection_dim * 2),
            nn.BatchNorm1d(projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(projection_dim * 2, projection_dim)
        )

        self.temperature = temperature

    def forward(self, visual_features, structured_features):
        """
        Args:
            visual_features: Tensor of shape [batch_size, visual_dim]
            structured_features: Tensor of shape [batch_size, structured_dim]

        Returns:
            contrastive_loss: Scalar loss value
            visual_proj: Projected visual features
            struct_proj: Projected structured features
        """
        # Project features to common space
        visual_proj = self.visual_projection(visual_features)
        struct_proj = self.structured_projection(structured_features)

        # Normalize embeddings
        visual_proj = F.normalize(visual_proj, dim=1)
        struct_proj = F.normalize(struct_proj, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(visual_proj, struct_proj.transpose(0, 1)) / self.temperature

        # For contrastive loss calculation
        labels = torch.arange(len(visual_proj)).to(visual_proj.device)

        # Calculate losses in both directions
        loss_v2s = F.cross_entropy(similarity, labels)
        loss_s2v = F.cross_entropy(similarity.t(), labels)

        # Average the losses
        contrastive_loss = (loss_v2s + loss_s2v) / 2

        return contrastive_loss, visual_proj, struct_proj
    
class AdElementSegmenter:
    """Detects actual locations of ad elements using CLIP and sliding window"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = device
        print("Loading CLIP model for element detection...")
        
        # Load CLIP model for semantic understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(device)
        self.clip_model.eval()
        
        # Define element detection queries
        self.element_queries = {
            'product': [
                "a product being advertised",
                "merchandise for sale", 
                "an item or object being promoted",
                "consumer goods",
                "a product shot"
            ],
            'person': [
                "a person or human",
                "a model posing",
                "someone's face or body",
                "a human figure",
                "people in the image"
            ],
            'text': [
                "text or writing",
                "words or letters", 
                "advertising copy",
                "written content",
                "typography or font"
            ],
            'logo': [
                "a company logo",
                "brand mark or symbol",
                "corporate branding",
                "brand logo or emblem",
                "company identifier"
            ],
            'background': [
                "background scenery",
                "backdrop or setting",
                "environmental context",
                "scene background",
                "ambient setting"
            ]
        }
        
        print("CLIP-based element detector initialized")

    def detect_elements(self, image):
        """
        Detect actual locations of elements using CLIP sliding window approach

        Args:
            image: PIL Image or tensor

        Returns:
            Dictionary of element masks showing actual detected locations
        """
        print("🔍 Detecting actual element locations...")
        
        # Convert image to proper format
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]  # Remove batch dimension
            h, w = image.shape[-2:]
            # Convert to PIL for CLIP processing
            image_pil = transforms.ToPILImage()(image.cpu())
        else:
            image_pil = image
            w, h = image.size
        
        # Create sliding window patches for detailed analysis
        patch_size = 64  # Size of each analysis patch
        stride = 32      # Overlap between patches
        
        element_masks = {}
        
        for element_type, queries in self.element_queries.items():
            print(f"  🔎 Detecting {element_type}...")
            
            # Create heatmap for this element type
            heatmap = torch.zeros((h, w), device=self.device)
            
            # Sliding window analysis
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    # Extract patch
                    patch = image_pil.crop((x, y, x + patch_size, y + patch_size))
                    
                    # Score patch for this element type
                    score = self._score_patch_for_element(patch, queries)
                    
                    # Add score to heatmap (with Gaussian weighting for smooth blending)
                    self._add_score_to_heatmap(heatmap, x, y, patch_size, score)
            
            # Normalize and threshold the heatmap
            heatmap = self._normalize_and_threshold_heatmap(heatmap, element_type)
            element_masks[element_type] = heatmap
            
            # Print detection statistics
            coverage = (heatmap > 0.1).sum().item() / (h * w) * 100
            max_confidence = heatmap.max().item()
            print(f"{element_type}: {coverage:.1f}% coverage, max confidence: {max_confidence:.3f}")
        
        return element_masks

    def _score_patch_for_element(self, patch, queries):
        """Score a patch against element queries using CLIP"""
        try:
            # Prepare inputs for CLIP
            inputs = self.clip_processor(
                text=queries, 
                images=patch, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get CLIP scores
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Return maximum probability across all queries for this element
            return probs.max().item()
            
        except Exception as e:
            print(f"Error scoring patch: {e}")
            return 0.0

    def _add_score_to_heatmap(self, heatmap, x, y, patch_size, score):
        """Add score to heatmap with Gaussian weighting for smooth blending"""
        # Create Gaussian weight matrix for smooth blending
        center_x, center_y = patch_size // 2, patch_size // 2
        
        for dy in range(patch_size):
            for dx in range(patch_size):
                # Gaussian weight (higher in center, lower at edges)
                dist_from_center = ((dx - center_x) ** 2 + (dy - center_y) ** 2) ** 0.5
                weight = torch.exp(torch.tensor(-dist_from_center ** 2 / (2 * (patch_size / 4) ** 2)))
                
                # Add weighted score to heatmap
                heatmap_y = y + dy
                heatmap_x = x + dx
                
                if heatmap_y < heatmap.shape[0] and heatmap_x < heatmap.shape[1]:
                    heatmap[heatmap_y, heatmap_x] += score * weight

    def _normalize_and_threshold_heatmap(self, heatmap, element_type):
        """Normalize heatmap and apply element-specific thresholds"""
        # Normalize to 0-1 range
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply element-specific thresholds
        thresholds = {
            'product': 0.3,    # Products should be clearly visible
            'person': 0.25,    # People are usually prominent
            'text': 0.2,       # Text can be smaller but important
            'logo': 0.35,      # Logos should be distinctive
            'background': 0.15  # Background is more diffuse
        }
        
        threshold = thresholds.get(element_type, 0.25)
        
        # Apply threshold and smooth
        heatmap = torch.where(heatmap > threshold, heatmap, torch.zeros_like(heatmap))
        
        # Apply Gaussian smoothing for cleaner masks
        if heatmap.sum() > 0:
            heatmap = self._gaussian_smooth(heatmap)
        
        return heatmap

    def _gaussian_smooth(self, heatmap, kernel_size=5):
        """Apply Gaussian smoothing to heatmap"""
        try:
            from scipy import ndimage
            import numpy as np
            
            # Convert to numpy, smooth, convert back
            heatmap_np = heatmap.cpu().numpy()
            smoothed = ndimage.gaussian_filter(heatmap_np, sigma=kernel_size/3)
            return torch.tensor(smoothed, device=self.device, dtype=heatmap.dtype)
        except:
            # Fallback: simple averaging if scipy not available
            return heatmap
    
class CounterfactualTester:
    """Tests causal impact using actual detected element locations"""

    def __init__(self, model, segmenter, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.segmenter = segmenter
        self.device = device
        self.model.eval()
        print("Initialized CounterfactualTester with real element detection")

    def analyze_element_importance(self, batch, element_types=None):
        """
        Analyze causal importance using actual detected element locations
        """
        if element_types is None:
            element_types = ['product', 'person', 'text', 'logo', 'background']

        print(" Starting counterfactual analysis with real element detection...")

        # Get original prediction
        with torch.no_grad():
            raw_logits = self.model(batch)
            original_pred = torch.sigmoid(raw_logits).mean().item()
        
        print(f"Original prediction: {original_pred:.4f}")
        
        # Get original image
        if 'image' in batch:
            original_image = batch['image']
            if original_image.dim() == 4:
                original_image = original_image[0]
        else:
            print("No raw image found")
            return self._create_dummy_results(original_pred)
        
        # Detect actual element locations
        print("🎯 Detecting actual element locations...")
        element_masks = self.segmenter.detect_elements(original_image)

        # Test impact of each detected element
        element_impacts = {}

        for element in element_types:
            if element in element_masks and element_masks[element].sum() > 0:
                print(f"Testing {element} impact...")
                impact = self._test_element_impact_advanced(batch, element, element_masks[element])
                element_impacts[element] = impact
                print(f"{element}: {impact*100:+.1f}% impact")
            else:
                element_impacts[element] = 0.0
                print(f"{element}: Not detected")
        
        return {
            'original_pred': original_pred,
            'element_impacts': element_impacts,
            'element_masks': element_masks,
            'detection_quality': self._assess_detection_quality(element_masks)
        }

    def _test_element_impact_advanced(self, batch, element_type, mask):
        """Test impact using multiple sophisticated masking strategies"""
        try:
            impacts = []
            
            # Get original prediction
            with torch.no_grad():
                original_pred = self.model(batch).mean().item()
            
            # Strategy 1: Intelligent inpainting (replace with surrounding context)
            masked_batch_inpaint = self._apply_intelligent_inpainting(batch, mask)
            with torch.no_grad():
                pred_inpaint = self.model(masked_batch_inpaint).mean().item()
            impact_inpaint = (pred_inpaint - original_pred) / (original_pred + 1e-8)
            impacts.append(impact_inpaint)
            
            # Strategy 2: Content-aware blur (blur based on mask intensity)
            masked_batch_blur = self._apply_content_aware_blur(batch, mask)
            with torch.no_grad():
                pred_blur = self.model(masked_batch_blur).mean().item()
            impact_blur = (pred_blur - original_pred) / (original_pred + 1e-8)
            impacts.append(impact_blur)
            
            # Strategy 3: Semantic replacement (replace with semantically neutral content)
            masked_batch_semantic = self._apply_semantic_replacement(batch, mask, element_type)
            with torch.no_grad():
                pred_semantic = self.model(masked_batch_semantic).mean().item()
            impact_semantic = (pred_semantic - original_pred) / (original_pred + 1e-8)
            impacts.append(impact_semantic)
            
            # Return weighted average (inpainting gets highest weight as most realistic)
            weights = [0.5, 0.3, 0.2]
            weighted_impact = sum(w * i for w, i in zip(weights, impacts))
            
            return weighted_impact
            
        except Exception as e:
            print(f"Error testing {element_type}: {e}")
            return 0.0

    def _apply_intelligent_inpainting(self, batch, mask):
        """Replace masked regions with surrounding context"""
        masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if 'image' in batch:
            image = batch['image'].clone()
            if image.dim() == 4:
                image = image[0]
            
            # Create inpainted version
            inpainted = self._inpaint_region(image, mask)
            
            if batch['image'].dim() == 4:
                inpainted = inpainted.unsqueeze(0)
            
            masked_batch['image'] = inpainted
        
        return masked_batch

    def _inpaint_region(self, image, mask):
        """Simple but effective inpainting using surrounding pixels"""
        # Dilate mask to get surrounding context
        kernel_size = 5
        dilated_mask = self._dilate_mask(mask, kernel_size)
        context_mask = dilated_mask - mask
        
        # Get average color from context region
        if context_mask.sum() > 0:
            context_pixels = image * context_mask.unsqueeze(0)
            avg_color = context_pixels.sum(dim=(1, 2)) / context_mask.sum()
            avg_color = avg_color.view(3, 1, 1)
        else:
            # Fallback to image mean
            avg_color = image.mean(dim=(1, 2)).view(3, 1, 1)
        
        # Replace masked region with context color + noise for realism
        noise = torch.randn_like(image) * 0.1
        replacement = avg_color + noise
        
        mask_3d = mask.unsqueeze(0).expand_as(image)
        return image * (1 - mask_3d) + replacement * mask_3d

    def _apply_content_aware_blur(self, batch, mask):
        """Apply blur that varies with mask intensity"""
        masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if 'image' in batch:
            image = batch['image'].clone()
            if image.dim() == 4:
                image = image[0]
            
            # Apply variable blur based on mask intensity
            blurred = self._variable_blur(image, mask)
            
            if batch['image'].dim() == 4:
                blurred = blurred.unsqueeze(0)
            
            masked_batch['image'] = blurred
        
        return masked_batch

    def _variable_blur(self, image, mask):
        """Apply blur that varies with mask intensity"""
        from torchvision.transforms import GaussianBlur
        
        # Create multiple blur levels
        blur_light = GaussianBlur(kernel_size=5, sigma=1)(image)
        blur_medium = GaussianBlur(kernel_size=11, sigma=3)(image)
        blur_heavy = GaussianBlur(kernel_size=21, sigma=7)(image)
        
        # Blend based on mask intensity
        mask_3d = mask.unsqueeze(0).expand_as(image)
        
        # Light blur for low intensity, heavy blur for high intensity
        blended = (image * (1 - mask_3d) + 
                  blur_light * mask_3d * (mask_3d < 0.3).float() +
                  blur_medium * mask_3d * ((mask_3d >= 0.3) & (mask_3d < 0.7)).float() +
                  blur_heavy * mask_3d * (mask_3d >= 0.7).float())
        
        return blended

    def _apply_semantic_replacement(self, batch, mask, element_type):
        """Replace with semantically neutral content for this element type"""
        masked_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if 'image' in batch:
            image = batch['image'].clone()
            if image.dim() == 4:
                image = image[0]
            
            # Get semantic replacement based on element type
            replacement = self._get_semantic_replacement(image, element_type)
            
            mask_3d = mask.unsqueeze(0).expand_as(image)
            replaced = image * (1 - mask_3d) + replacement * mask_3d
            
            if batch['image'].dim() == 4:
                replaced = replaced.unsqueeze(0)
            
            masked_batch['image'] = replaced
        
        return masked_batch

    def _get_semantic_replacement(self, image, element_type):
        """Get appropriate replacement content for each element type"""
        # Define neutral replacements for each element type
        replacements = {
            'product': image.mean(dim=(1, 2)).view(3, 1, 1),  # Average color
            'person': torch.tensor([0.5, 0.5, 0.5], device=image.device).view(3, 1, 1),  # Gray
            'text': image.mean(dim=(1, 2)).view(3, 1, 1),  # Background color
            'logo': image.mean(dim=(1, 2)).view(3, 1, 1),  # Background color
            'background': image.median(dim=1)[0].median(dim=1)[0].view(3, 1, 1)  # Median color
        }
        
        base_replacement = replacements.get(element_type, image.mean(dim=(1, 2)).view(3, 1, 1))
        
        # Add subtle texture to avoid completely flat regions
        texture = torch.randn_like(image) * 0.05
        return base_replacement + texture

    def _dilate_mask(self, mask, kernel_size):
        """Dilate mask to get surrounding context"""
        # Simple dilation using max pooling
        import torch.nn.functional as F
        
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(mask_4d, kernel_size, stride=1, padding=kernel_size//2)
        return dilated.squeeze(0).squeeze(0)

    def _assess_detection_quality(self, element_masks):
        """Assess quality of element detection"""
        quality_scores = {}
        
        for element, mask in element_masks.items():
            if mask.sum() > 0:
                # Calculate quality metrics
                coverage = mask.sum().item() / mask.numel()
                max_intensity = mask.max().item()
                compactness = self._calculate_compactness(mask)
                
                quality = (max_intensity * 0.4 + compactness * 0.3 + min(coverage * 10, 1.0) * 0.3)
                quality_scores[element] = quality
            else:
                quality_scores[element] = 0.0
        
        return quality_scores

    def _calculate_compactness(self, mask):
        """Calculate how compact/concentrated the mask is"""
        if mask.sum() == 0:
            return 0.0
        
        # Find center of mass
        y_coords, x_coords = torch.where(mask > 0.1)
        if len(y_coords) == 0:
            return 0.0
        
        center_y = y_coords.float().mean()
        center_x = x_coords.float().mean()
        
        # Calculate average distance from center
        distances = ((y_coords.float() - center_y) ** 2 + (x_coords.float() - center_x) ** 2) ** 0.5
        avg_distance = distances.mean()
        
        # Normalize by image size
        max_distance = (mask.shape[0] ** 2 + mask.shape[1] ** 2) ** 0.5
        compactness = 1.0 - (avg_distance / max_distance)
        
        return compactness.item()

    def _create_dummy_results(self, original_pred):
        """Create dummy results when image is not available"""
        return {
            'original_pred': original_pred,
            'element_impacts': {element: 0.0 for element in ['product', 'person', 'text', 'logo', 'background']},
            'element_masks': {},
            'detection_quality': {}
        }

    def visualize_element_importance(self, image, results, save_path=None):
        """
        Create advanced heatmap visualization showing actual detected element locations
        """
        print(f"🎨 Creating advanced heatmap visualization...")
        
        # Convert image for visualization
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image[0]
            image_np = image.permute(1, 2, 0).cpu().numpy()
            
            # Denormalize ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = np.array(image) / 255.0

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout: 3 rows x 6 columns
        gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.3)
        
        # Row 1: Original + Detection Overview
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_detection = fig.add_subplot(gs[0, 1:3])
        ax_quality = fig.add_subplot(gs[0, 3:5])
        ax_summary = fig.add_subplot(gs[0, 5])
        
        # Row 2 & 3: Individual element heatmaps
        element_axes = []
        for i in range(10):  # 2 rows x 5 columns for elements
            row = 1 + i // 5
            col = i % 5
            if row < 3:  # Only use 2 rows
                ax = fig.add_subplot(gs[row, col])
                element_axes.append(ax)

        # Plot original image
        ax_orig.imshow(image_np)
        ax_orig.set_title(f"Original Image\nPred: {results['original_pred']:.4f}", 
                         fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Plot detection overview (all elements combined)
        ax_detection.imshow(image_np)
        
        # Combine all element masks with different colors
        combined_overlay = np.zeros((*image_np.shape[:2], 4))  # RGBA
        element_colors_rgba = {
            'product': [1, 0, 0, 0.6],    # Red
            'person': [0, 0, 1, 0.6],     # Blue  
            'text': [0, 1, 0, 0.6],       # Green
            'logo': [1, 0, 1, 0.6],       # Magenta
            'background': [1, 0.5, 0, 0.4] # Orange
        }
        
        for element, color in element_colors_rgba.items():
            if element in results['element_masks']:
                mask = results['element_masks'][element].cpu().numpy()
                if mask.sum() > 0:
                    # Add colored overlay
                    for c in range(3):
                        combined_overlay[:, :, c] += mask * color[c]
                    combined_overlay[:, :, 3] = np.maximum(combined_overlay[:, :, 3], mask * color[3])
        
        # Clip values and display
        combined_overlay = np.clip(combined_overlay, 0, 1)
        ax_detection.imshow(combined_overlay)
        ax_detection.set_title("All Detected Elements\n(Red=Product, Blue=Person, Green=Text, Magenta=Logo, Orange=Background)", 
                              fontsize=10, fontweight='bold')
        ax_detection.axis('off')
        
        # Plot detection quality scores
        if 'detection_quality' in results:
            quality_scores = results['detection_quality']
            elements = list(quality_scores.keys())
            scores = list(quality_scores.values())
            
            bars = ax_quality.bar(elements, scores, color=['red', 'blue', 'green', 'magenta', 'orange'])
            ax_quality.set_title("Detection Quality Scores", fontsize=12, fontweight='bold')
            ax_quality.set_ylabel("Quality Score")
            ax_quality.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax_quality.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.setp(ax_quality.get_xticklabels(), rotation=45, ha='right')
        
        # Plot impact summary
        impacts = results['element_impacts']
        elements = list(impacts.keys())
        impact_values = [impacts[e] * 100 for e in elements]  # Convert to percentage
        
        colors = ['green' if v > 0 else 'red' if v < 0 else 'gray' for v in impact_values]
        bars = ax_summary.barh(elements, impact_values, color=colors)
        ax_summary.set_title("Impact Summary (%)", fontsize=12, fontweight='bold')
        ax_summary.set_xlabel("Impact (%)")
        ax_summary.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, impact_values):
            width = bar.get_width()
            ax_summary.text(width + (0.1 if width >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                           f'{value:+.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        
        # Plot individual element heatmaps
        element_list = ['product', 'person', 'text', 'logo', 'background']
        colormaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        
        for i, (element, cmap) in enumerate(zip(element_list, colormaps)):
            if i < len(element_axes):
                ax = element_axes[i]
                
                # Show original image as background
                ax.imshow(image_np, alpha=0.7)
                
                # Overlay element mask
                if element in results['element_masks']:
                    mask = results['element_masks'][element].cpu().numpy()
                    if mask.sum() > 0:
            # Create heatmap overlay
                        im = ax.imshow(mask, alpha=0.8, cmap=cmap, vmin=0, vmax=1)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label('Detection\nConfidence', fontsize=8)
                        cbar.ax.tick_params(labelsize=6)
                
                # Set title with impact and quality info
                impact = results['element_impacts'].get(element, 0) * 100
                quality = results.get('detection_quality', {}).get(element, 0)
                
                title_color = 'green' if impact > 1 else 'red' if impact < -1 else 'gray'
                ax.set_title(f"{element.capitalize()}\nImpact: {impact:+.1f}% | Quality: {quality:.2f}", 
                           color=title_color, fontsize=10, fontweight='bold')
            ax.axis('off')

        # Hide unused axes
        for i in range(len(element_list), len(element_axes)):
            element_axes[i].axis('off')

        # Add overall title
        fig.suptitle(f"Counterfactual Analysis: Real Element Detection\nOriginal Prediction: {results['original_pred']:.4f}", 
                    fontsize=16, fontweight='bold')
        
        # Save the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white')
            print(f"💾 Saved comprehensive heatmap to {save_path}")
        
        # Display
        plt.show()

        return fig
    
class CTRPredictionModel(nn.Module):
    def __init__(self, data_info, embed_dim=16, visual_dim=512, cross_layers=3,
                 deep_layers=[256, 128, 64], dropout=0.4, use_images=True, contrastive_weight=0.1):
        super().__init__()

        # Store feature info
        self.high_card_vocab_sizes = data_info['high_card_vocab_sizes']
        self.simple_cat_vocab_sizes = data_info['simple_cat_vocab_sizes']
        self.num_numerical = len(data_info['numerical_cols'])
        self.use_images = use_images
        self.contrastive_weight = contrastive_weight
        self.visual_dim = visual_dim

        # High-cardinality feature embeddings
        self.high_card_embeddings = nn.ModuleDict({
            col: nn.Sequential(
                nn.Embedding(vocab_size, embed_dim, padding_idx=0),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(0.1)
            ) for col, vocab_size in self.high_card_vocab_sizes.items()
        })
        self.high_card_dim = len(self.high_card_vocab_sizes) * embed_dim

        # Simple categorical feature embeddings
        self.simple_cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for col, vocab_size in self.simple_cat_vocab_sizes.items()
        })
        self.simple_cat_dim = len(self.simple_cat_vocab_sizes) * embed_dim

        # Numerical features processing
        self.numerical_bn = nn.BatchNorm1d(self.num_numerical)
        self.numerical_dim = self.num_numerical

        # Calculate structured features dimension
        structured_dim = self.high_card_dim + self.simple_cat_dim + self.numerical_dim

        # Structured features processing
        self.structured_processor = nn.Sequential(
            nn.Linear(structured_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Visual encoder and projection
        if self.use_images:
            self.visual_encoder = VisualEncoder(output_dim=visual_dim, trainable = True)
            # Visual projection to match other feature dimensions
            self.visual_projection_dim = 128  # Increased from 64
            self.visual_proj = nn.Sequential(
                nn.Linear(visual_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, self.visual_projection_dim),
                nn.BatchNorm1d(self.visual_projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)  # Reduced dropout for final layer
            )
        else:
            self.visual_projection_dim = 0

        # Calculate total feature dimension
        total_dim = 128 + self.visual_projection_dim  # 128 from structured processor + visual projection

        # Cross Network
        self.cross_layers_module = nn.ModuleList([
            CrossLayer(total_dim, dropout = dropout*0.5) for _ in range(cross_layers)
        ])

        # Deep Network
        deep_input_dim = total_dim
        self.deep_layers_module = nn.ModuleList()
        for layer_dim in deep_layers:
            self.deep_layers_module.append(nn.Sequential(
                nn.Linear(deep_input_dim, layer_dim),
                nn.BatchNorm1d(layer_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
            deep_input_dim = layer_dim

        # Final prediction layer
        final_input_dim = total_dim + deep_layers[-1]  # Cross output + Deep output
        self.final_layer = nn.Sequential(
            nn.Linear(final_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(32, 1)  # Single output for binary classification
        )

        # Contrastive learning components
        if self.use_images and contrastive_weight > 0:
            self.use_contrastive = True
            self.contrastive_head = ContrastiveAlignmentLayer(
                visual_dim=self.visual_projection_dim,
                structured_dim=structured_dim,
                projection_dim=128
            )
        else:
            self.use_contrastive = False

        self._initialize_weights()

    def _initialize_weights(self):
        """Proper weight initialization to prevent saturation"""
        print("Initializing model weights...")
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                # Normal initialization for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.BatchNorm1d):
                # Standard batch norm initialization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for final layer to prevent saturation
        if hasattr(self, 'final_layer'):
                print("   🎯 Special initialization for final layer...")
                
                # Find the last linear layer in the final_layer sequential
                final_linear = None
                for name, module in self.final_layer.named_modules():
                    if isinstance(module, nn.Linear):
                        final_linear = module
                        final_layer_name = name
                
                if final_linear is not None:
                    # Use smaller gain to prevent saturation
                    nn.init.xavier_uniform_(final_linear.weight, gain=0.1)
                    nn.init.constant_(final_linear.bias, 0.0)
                    print(f"   🎯 Final layer ({final_layer_name}) initialized with gain=0.1")
                else:
                    print("   ⚠️ Warning: Could not find final linear layer")
            
        print("✅ Weight initialization complete")

    def forward(self, batch, return_embeddings=False):
        """
        Forward pass for the model
        
        Args:
            batch: Dictionary containing all inputs
            return_embeddings: Whether to return embeddings for contrastive learning
            
        Returns:
            Prediction or (prediction, structured_features, visual_features)
        """
        print(f"DEBUG - Batch keys: {list(batch.keys())}")
    
        # Check if images are in the batch
        if 'image' in batch:
            images = batch['image']
            print(f"DEBUG - Images: shape={images.shape}, dtype={images.dtype}")
            print(f"DEBUG - Images stats: mean={images.mean().item():.6f}, std={images.std().item():.6f}")
            print(f" DEBUG - Images range: min={images.min().item():.6f}, max={images.max().item():.6f}")
            
            # Check if images are all zeros
            if images.abs().sum().item() < 1e-6:
                print(" CRITICAL ERROR: All images are zeros! Visual encoder will output zeros.")
                print("   This explains why visual vectors are 0.000000")
            else:
                print("Images contain real data")
                
            # Check image preprocessing
            if images.min().item() < -3 or images.max().item() > 3:
                print("WARNING: Images may not be properly normalized for CLIP")
        else:
            print("CRITICAL ERROR: No 'image' key in batch!")
            print("   Available keys:", list(batch.keys()))
            print("   Visual encoder cannot work without images")
        
        if 'visual_features' in batch:
            visual_features = batch['visual_features']
            print(f"🔍 DEBUG - Pre-computed visual features: shape={visual_features.shape}, mean={visual_features.mean().item():.6f}")
            if visual_features.abs().sum().item() < 1e-6:
                print("ERROR: Pre-computed visual features are also zeros!")
                
        # Process high cardinality features
        high_card_embeddings = []
        for col, embedding_layer in self.high_card_embeddings.items():
            high_card_embeddings.append(embedding_layer(batch['high_card'][col]))
        high_card_vectors = torch.cat(high_card_embeddings, dim=1)
        
        # Process simple categorical features
        simple_cat_embeddings = []
        for col, embedding_layer in self.simple_cat_embeddings.items():
            simple_cat_embeddings.append(embedding_layer(batch['simple_cat'][col]))
        simple_cat_vectors = torch.cat(simple_cat_embeddings, dim=1)
        
        # Process numerical features
        numerical_vectors = self.numerical_bn(batch['numerical'])

        print(f" DEBUG - High card vectors: shape={high_card_vectors.shape}, mean={high_card_vectors.mean().item():.6f}")
        print(f"DEBUG - Simple cat vectors: shape={simple_cat_vectors.shape}, mean={simple_cat_vectors.mean().item():.6f}")
        print(f"DEBUG - Numerical vectors: shape={numerical_vectors.shape}, mean={numerical_vectors.mean().item():.6f}")

        # Combine structured features
        structured_features = torch.cat([
            high_card_vectors,
            simple_cat_vectors,
            numerical_vectors
        ], dim=1)
        
        # Process structured features
        processed_structured = self.structured_processor(structured_features)

        print(f"DEBUG - Processed structured: shape={processed_structured.shape}, mean={processed_structured.mean().item():.6f}")
        
        # Process visual features
        visual_vectors = None
        if self.use_images:
            # First try to use images directly - this is more reliable
            if 'image' in batch and batch['image'] is not None:
                # Check if images are valid
                if batch['image'].abs().sum().item() > 1e-6:
                    # Extract features from image
                    raw_visual_features = self.visual_encoder(batch['image'])
                    visual_vectors = raw_visual_features
                
            # Only fall back to pre-computed features if image processing failed
            if visual_vectors is None and 'visual_features' in batch:
                # Check if features are valid
                if batch['visual_features'].abs().sum().item() > 1e-6:
                    visual_vectors = batch['visual_features']
            
            # If we still don't have valid visual vectors, use zeros
            if visual_vectors is None:
                visual_vectors = torch.zeros(batch['numerical'].size(0), self.visual_dim, device=batch['numerical'].device)
            
                # Apply visual projection
                visual_vectors = self.visual_proj(visual_vectors)
            else:
                visual_vectors = torch.zeros(batch['numerical'].size(0), self.visual_projection_dim, device=batch['numerical'].device)
        
        print(f"🔍 DEBUG - Visual vectors: shape={visual_vectors.shape}, mean={visual_vectors.mean().item():.6f}")
        
        # Check if visual features are being used
        if visual_vectors.std().item() < 1e-6:
            print("⚠️ WARNING: Visual vectors have very low variance - may not be learning!")


        # Combine processed features
        combined_features = torch.cat([processed_structured, visual_vectors], dim=1)

        print(f"🔍 DEBUG - Combined features: shape={combined_features.shape}, mean={combined_features.mean().item():.6f}")

        # Ensure structured features require gradients
        structured_features.requires_grad_(True)

        # if structured_contribution > visual_contribution * 10:
        #     print("⚠️ WARNING: Structured features dominating - visual features may be underutilized!")
        # elif visual_contribution > structured_contribution * 10:
        #     print("⚠️ WARNING: Visual features dominating - structured features may be underutilized!")
        
        # Cross network
        cross_output = combined_features
        for cross_layer in self.cross_layers_module:
            cross_output = cross_layer(cross_output, combined_features)

        print(f"🔍 DEBUG - Cross output: shape={cross_output.shape}, mean={cross_output.mean().item():.6f}")
        
        # Deep network
        deep_output = combined_features
        for layer in self.deep_layers_module:
            deep_output = layer(deep_output)

        print(f"🔍 DEBUG - Deep output: shape={deep_output.shape}, mean={deep_output.mean().item():.6f}")
        
        # Combine outputs and make final prediction
        final_input = torch.cat([cross_output, deep_output], dim=1)
        # Get raw logits (single computation, no duplication)
        raw_logits = self.final_layer(final_input).squeeze(-1)
        
        # Return embeddings if requested (for contrastive learning)
        raw_logits = self.final_layer(final_input).squeeze(-1)  # Raw logits

        if return_embeddings:
            return raw_logits, structured_features, visual_vectors

        return raw_logits
    
    def compute_contrastive_loss(self, structured_features, visual_vectors):
        """
        Compute contrastive loss between structured and visual features
        
        Args:
            structured_features: Structured feature embeddings
            visual_vectors: Visual feature embeddings
            
        Returns:
            Contrastive loss value
        """
        if not self.use_contrastive or not hasattr(self, "contrastive_head"):
            return torch.tensor(0.0, device=structured_features.device)
        
        contrastive_loss, _, _ = self.contrastive_head(visual_vectors, structured_features)
        return contrastive_loss

# Pre-process and cache images
def cache_processed_images(data_info, image_dir, output_dir, transform):
    """
    Process and cache images to disk to speed up training
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which split we're processing based on the output directory
    if 'train' in output_dir:
        item_ids = data_info['item_ids_train']
    else:
        item_ids = data_info['item_ids_test']
    
    print(f"Processing and caching {len(item_ids)} images to {output_dir}...")
    
    for item_id in tqdm(item_ids):
        # Skip if already cached
        cache_path = os.path.join(output_dir, f"{item_id}.pt")
        if os.path.exists(cache_path):
            continue
            
        # Try to find the image with different extensions
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = os.path.join(image_dir, f"{item_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            # Create a blank image if not found
            blank_tensor = torch.zeros(3, 224, 224)
            torch.save(blank_tensor, cache_path)
            continue
        
        try:
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            tensor = transform(image)
            
            # Validate tensor is not zeros
            if tensor.abs().sum().item() > 1e-6:
                torch.save(tensor, cache_path)
            else:
                print(f"Warning: Image {item_id} produced zero tensor")
        except Exception as e:
            print(f"Error processing image {item_id}: {e}")


# Extract and save CLIP features
def extract_and_save_clip_features(model, data_info, image_dir, feature_dir, transform, batch_size=16):
    """
    Extract CLIP features from images and save them to disk
    """

    os.makedirs(feature_dir, exist_ok=True)
    
    # Combine train and test item IDs to process all images once
    all_item_ids = np.concatenate([data_info['item_ids_train'], data_info['item_ids_test']])
    unique_item_ids = np.unique(all_item_ids)
    
    print(f"Extracting CLIP features for {len(unique_item_ids)} unique images...")
    
    # Dictionary to store all features
    features_dict = {}
    
    # Create a simple dataset for images only
    class ImageDataset(Dataset):
        def __init__(self, item_ids, image_dir, transform):
            self.item_ids = item_ids
            self.image_dir = image_dir
            self.transform = transform
            
        def __len__(self):
            return len(self.item_ids)
            
        def __getitem__(self, idx):
            item_id = self.item_ids[idx]
            
            # Try different extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                image_path = os.path.join(self.image_dir, f"{item_id}{ext}")
                if os.path.exists(image_path):
                    try:
                        image = Image.open(image_path).convert('RGB')
                        return self.transform(image), item_id
                    except:
                        break
            
            # Return blank image if not found or error
            return torch.zeros((3, 224, 224)), item_id
    
    # Create dataset and dataloader for all unique images
    dataset = ImageDataset(unique_item_ids, image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    # Extract features
    model.eval()
    with torch.no_grad():
        for batch_images, batch_ids in tqdm(dataloader, desc="Extracting features"):
            # Add this validation check
            if batch_images.abs().sum().item() < 1e-6:
                print(f"Skipping batch - all images are zeros")
                continue

            batch_images = batch_images.to(next(model.parameters()).device)  # Use model's device
            features = model.visual_encoder(batch_images)
            
            # Store features in dictionary
            for i, item_id in enumerate(batch_ids):
                # Only save non-zero features
                if features[i].abs().sum().item() > 1e-6:
                    feature = features[i].cpu()
                    torch.save(feature, os.path.join(output_dir, f"{item_id}.pt"))
    
    # Save all features to a single file
    torch.save(features_dict, os.path.join(feature_dir, "features.pt"))
    print(f"Saved features for {len(features_dict)} images to {feature_dir}/features.pt")

# Training and evaluation functions
def train_model(model, train_loader, optimizer, criterion, device, epoch, contrastive_weight=0.1):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_contrastive_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                {k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v 
                for k, v in batch.items()}
        
        # Forward pass
        pred, structured_features, visual_features = model(batch, return_embeddings=True)
        
        # Calculate CTR prediction loss
        ctr_loss = criterion(pred, batch['label'].float())  # pred is raw logits, criterion applies sigmoid internally
        
        # Calculate contrastive loss if enabled
        contrastive_loss = model.compute_contrastive_loss(structured_features, visual_features)
        
        # Combine losses
        loss = ctr_loss + contrastive_weight * contrastive_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += ctr_loss.item()
        total_contrastive_loss += contrastive_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader)
    
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Contrastive Loss = {avg_contrastive_loss:.4f}")
    return avg_loss, avg_contrastive_loss

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                    {k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            raw_logits = model(batch)
            
            # Calculate loss
            loss = criterion(raw_logits, batch['label'].float())
            
            # Update metrics
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(raw_logits).cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate AUC and AP
    auc = roc_auc_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_preds)
    
    avg_loss = total_loss / len(test_loader)
    
    print(f"Test Loss = {avg_loss:.4f}, AUC = {auc:.4f}, AP = {ap:.4f}")
    return avg_loss, auc, ap

def validate_data_and_model(model, data_loader, device):
    """Validate that data and model are working correctly"""
    print(" Validating data and model...")
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= 3:  # Check first 3 batches
                break
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                    {k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v 
                    for k, v in batch.items()}
            
            print(f"\n📊 Batch {i} validation:")
            
            # Check target values
            if 'target' in batch:
                targets = batch['target']
                print(f"   Targets: min={targets.min().item():.4f}, max={targets.max().item():.4f}, mean={targets.mean().item():.4f}")
            
            # Check model prediction
            pred = model(batch)
            print(f"   Predictions: min={pred.min().item():.4f}, max={pred.max().item():.4f}, mean={pred.mean().item():.4f}")
            
            # Check if predictions are all the same
            if pred.std().item() < 1e-6:
                print("   ⚠️ WARNING: All predictions are identical!")
            
            # Check for NaN or inf
            if torch.isnan(pred).any():
                print("   ❌ ERROR: NaN values in predictions!")
            if torch.isinf(pred).any():
                print("   ❌ ERROR: Inf values in predictions!")
    print("✅ Data validation complete")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, args.data_file), nrows=args.max_rows)
    
    # Split into train and test
    df['log_time'] = pd.to_datetime(df['log_time'])
    train_df = df[(df['log_time'] >= '2023-07-10') & (df['log_time'] <= '2023-08-05')]
    test_df = df[(df['log_time'] >= '2023-08-06') & (df['log_time'] <= '2023-08-09')]
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Prepare data
    data_info = prepare_data_efficient(
        train_df,
        test_df,
        max_vocab_size=args.max_vocab_size,
        max_seq_len=args.max_seq_len
    )
    
    # Add item_ids for image loading
    data_info['item_ids_train'] = train_df['item_id'].values
    data_info['item_ids_test'] = test_df['item_id'].values
    
    # Define image transformations
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # Cache processed images
    if not os.path.exists(os.path.join(CACHE_DIR, "train")) or not os.path.exists(os.path.join(CACHE_DIR, "test")):
        print("Caching processed images...")
        for split in ['train', 'test']:
            cache_processed_images(
                data_info, 
                IMAGE_DIR, 
                os.path.join(CACHE_DIR, split), 
                image_transforms[split]
            )
    else:
        print("Using existing image cache")
    
    # Create model
    model = CTRPredictionModel(
        data_info=data_info,
        embed_dim=args.embed_dim,
        visual_dim=args.visual_dim,
        cross_layers=args.cross_layers,
        deep_layers=args.deep_layers,
        dropout=args.dropout,
        use_images=args.use_images,
        contrastive_weight=args.contrastive_weight
    ).to(device)
    
    # Extract and save CLIP features
    if not os.path.exists(os.path.join(FEATURE_DIR, "features.pt")):
        print("Extracting CLIP features...")
        extract_and_save_clip_features(
            model,
            data_info,
            IMAGE_DIR,
            FEATURE_DIR,
            image_transforms['test'],
            batch_size=args.batch_size
        )
    else:
        print("Using existing CLIP features")
    
    # Create datasets
    train_dataset = CTRDataset(
        data_info, 
        image_cache_dir=os.path.join(CACHE_DIR, "train"),
        feature_dir=FEATURE_DIR, 
        split='train'
    )
    
    test_dataset = CTRDataset(
        data_info, 
        image_cache_dir=os.path.join(CACHE_DIR, "test"),
        feature_dir=FEATURE_DIR, 
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )

    print("\n🔍 Debugging dataset image loading...")
    sample_batch_found = False
    for i, batch in enumerate(train_loader):
        print(f"🔍 Sample batch {i}:")
        print(f"   Batch keys: {list(batch.keys())}")
        
        if 'image' in batch:
            images = batch['image']
            print(f"   Images: shape={images.shape}, mean={images.mean().item():.6f}")
            if images.abs().sum().item() < 1e-6:
                print("   ❌ ERROR: Dataset is loading zero images!")
                print("   Check your image preprocessing and loading pipeline")
            else:
                print("   ✅ Dataset is loading real images")
            sample_batch_found = True
        else:
            print("   ❌ ERROR: No 'image' key in batch from dataset!")
            print("   Check your CTRDataset.__getitem__ method")
        
        if 'visual_features' in batch:
            visual_features = batch['visual_features']
            print(f"   Pre-computed visual features: shape={visual_features.shape}, mean={visual_features.mean().item():.6f}")
        
        break  # Only check first batch

    if not sample_batch_found:
        print("❌ CRITICAL: Could not find any batches with images!")

    print("🔍 Dataset debugging complete\n")
    
    # Define loss function and optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(1.0, device=device),  # Adjust if data is imbalanced
    reduction='mean'
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6
    )

    validate_data_and_model(model, train_loader, device)
    
    # Training loop
    train_losses = []
    test_losses = []
    aucs = []
    train_aucs = []
    contrastive_losses = []

    early_stopping = EarlyStopping(patience=3, min_delta=0.001, restore_best_weights=True)
    monitor = TrainingMonitor()

    print("Starting training with overfitting prevention...")
    print(f"Early stopping patience: {early_stopping.patience} epochs")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, contrastive_loss = train_model(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device, 
            epoch, 
            contrastive_weight=args.contrastive_weight
        )
        
        # Evaluate
        test_loss, auc, ap = evaluate_model(model, test_loader, criterion, device)
        train_eval_loss, train_auc, train_ap = evaluate_model(model, train_loader, criterion, device)

        # Log feature importance every few epochs
        if epoch % 2 == 0:  # Every 2 epochs
            print(f"\n Feature Importance Analysis - Epoch {epoch}:")
            
            # Get a sample batch from the test loader
            sample_batch = next(iter(test_loader))
            sample_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else 
                {k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v
                for k, v in sample_batch.items()
            }

            # Log feature importance
            log_feature_importance(model, sample_batch, device)
        
        # Update metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_aucs.append(train_auc)
        aucs.append(auc)
        contrastive_losses.append(contrastive_loss)

        # Log training progress
        monitor.log_epoch(train_loss, test_loss, train_auc, auc, epoch)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Early stopping check
        if early_stopping(test_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best test loss: {early_stopping.best_loss:.4f}")
            break

        # Print epoch summary
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} → Test Loss: {test_loss:.4f}")
        print(f"  Train AUC: {train_auc:.4f} → Test AUC: {auc:.4f}")
        print(f"  AUC Gap: {train_auc - auc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'ctr_model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'auc': auc,
        }, checkpoint_path)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_aucs, label='Train AUC')
    plt.plot(aucs, label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('AUC Curves')
    
    plt.subplot(1, 3, 3)
    plt.plot(contrastive_losses, label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Contrastive Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'))
    
    # Advanced Counterfactual Analysis with Real Element Detection
    print("\n" + "="*60)
    print("ADVANCED COUNTERFACTUAL ANALYSIS")
    print("Real Element Detection + Sophisticated Masking")
    print("="*60)
    
    # Initialize advanced segmenter and tester
    print("Initializing advanced analysis components...")
    segmenter = AdElementSegmenter(device=device)
    tester = CounterfactualTester(model, segmenter, device=device)
    
    # Test on examples
    print("Running advanced counterfactual analysis...")
    
    analysis_results = []
    
    for i, batch in enumerate(test_loader):
        if i >= 3:  # Analyze 3 examples
            break
        
        print(f"\n{'='*40}")
        print(f"ANALYZING EXAMPLE {i}")
        print(f"{'='*40}")
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                {k2: v2.to(device) for k2, v2 in v.items()} if isinstance(v, dict) else v 
                for k, v in batch.items()}
        
        try:
            # Run comprehensive analysis
            results = tester.analyze_element_importance(batch)
        
            # Create advanced visualization
            save_path = os.path.join(RESULTS_DIR, f'advanced_counterfactual_analysis_example_{i}.png')
            fig = tester.visualize_element_importance(
            batch['image'], 
            results, 
                save_path=save_path
            )
            
            # Print detailed analysis
            print(f"\n DETAILED ANALYSIS RESULTS:")
            print(f"Original Prediction: {results['original_pred']:.4f}")
            
            print(f"\n  Element Detection Quality:")
            for element, quality in results.get('detection_quality', {}).items():
                status = "🟢" if quality > 0.5 else "🟡" if quality > 0.2 else "🔴"
                print(f"      {status} {element.capitalize()}: {quality:.3f}")
            
            print(f"\n Element Impact Analysis:")
            sorted_impacts = sorted(results['element_impacts'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True)
            
            for element, impact in sorted_impacts:
                impact_pct = impact * 100
                if abs(impact_pct) > 0.5:  # Only show significant impacts
                    status = "🔴" if impact_pct < -1 else "🟢" if impact_pct > 1 else "🟡"
                    significance = "HIGH" if abs(impact_pct) > 3 else "MEDIUM" if abs(impact_pct) > 1 else "LOW"
                    print(f"      {status} {element.capitalize()}: {impact_pct:+.1f}% ({significance})")
            
            # Store results for summary
            analysis_results.append({
                'example': i,
                'prediction': results['original_pred'],
                'impacts': results['element_impacts'],
                'quality': results.get('detection_quality', {})
            })
            
            # Close figure to save memory
            plt.close(fig)
            
        except Exception as e:
            print(f" Error analyzing example {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if analysis_results:
        # Calculate average impacts across all examples
        all_impacts = {}
        for result in analysis_results:
            for element, impact in result['impacts'].items():
                if element not in all_impacts:
                    all_impacts[element] = []
                all_impacts[element].append(impact * 100)
        
        print("📊 Average Element Impacts Across All Examples:")
        for element, impacts in all_impacts.items():
            avg_impact = sum(impacts) / len(impacts)
            std_impact = (sum((x - avg_impact) ** 2 for x in impacts) / len(impacts)) ** 0.5
            significance = "HIGH" if abs(avg_impact) > 2 else "MEDIUM" if abs(avg_impact) > 0.5 else "LOW"
            status = "🔴" if avg_impact < -0.5 else "🟢" if avg_impact > 0.5 else "⚪"
            print(f"   {status} {element.capitalize()}: {avg_impact:+.1f}% ± {std_impact:.1f}% ({significance})")
    
    print(f"\n Advanced counterfactual analysis complete!")
    print(f" Detailed visualizations saved to: {RESULTS_DIR}")
    print(f" Each visualization shows:")
    print(f"   • Actual detected element locations (not predefined regions)")
    print(f"   • Detection quality scores")
    print(f"   • Sophisticated masking strategies")
    print(f"   • Comprehensive impact analysis")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Model checkpoints saved to {CHECKPOINT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTR Prediction with Multimodal Model")
    
    # Data parameters
    parser.add_argument("--data_file", type=str, default="antm2c_10m_part0", help="Data file name")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to load")
    parser.add_argument("--max_vocab_size", type=int, default=5000, help="Maximum vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=5, help="Maximum sequence length")
    
    # Model parameters
    parser.add_argument("--embed_dim", type=int, default=16, help="Embedding dimension")
    parser.add_argument("--visual_dim", type=int, default=512, help="Visual feature dimension")
    parser.add_argument("--cross_layers", type=int, default=3, help="Number of cross layers")
    parser.add_argument("--deep_layers", type=str, default="256,128,64", help="Deep layers dimensions (comma-separated)")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--use_images", action="store_true", default=True, help="Whether to use images")
    parser.add_argument("--contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=15, help="Number of epochs")
    
    args = parser.parse_args()
    args.deep_layers = [int(x) for x in args.deep_layers.split(",")]
    
    main(args)