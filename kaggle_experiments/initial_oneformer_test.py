#!/usr/bin/env python3
"""
Physics-Enhanced OneFormer with Logit Space Operations
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
KAGGLE_ADE20K_PATH = '/kaggle/input/ade20k-scene-parsing/ADEChallengeData2016'
OUTPUT_DIR = Path('./physics_enhanced_oneformer')
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
np.random.seed(42)

# ADE20K Classes
ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
    'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book',
    'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
    'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
    'streetlight', 'booth', 'television', 'airplane', 'dirt track',
    'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman',
    'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
    'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool',
    'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike',
    'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
    'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase',
    'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator',
    'glass', 'clock', 'flag'
]


class ADE20KDataset:
    def __init__(self, root_path, split='validation', image_size=512, max_samples=None):
        self.root = Path(root_path)
        self.split = split
        self.image_size = image_size
        
        self.img_dir = self.root / 'images' / split
        self.ann_dir = self.root / 'annotations' / split
        
        self.images = sorted(list(self.img_dir.glob('*.jpg')))
        self.masks = sorted(list(self.ann_dir.glob('*.png')))
        
        if max_samples:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        mask_path = self.masks[idx]
        mask = Image.open(mask_path)
        mask_resized = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask_array = np.array(mask_resized).astype(np.int64)
        
        mask_converted = mask_array.copy()
        mask_converted[mask_array == 0] = 255
        valid_mask = (mask_array >= 1) & (mask_array <= 150)
        mask_converted[valid_mask] = mask_array[valid_mask] - 1
        invalid_mask = mask_array > 150
        mask_converted[invalid_mask] = 255
        mask_converted = np.clip(mask_converted, 0, 255)
        
        return {
            'pil_image': img_resized,
            'mask': torch.from_numpy(mask_converted),
            'filename': img_path.stem
        }


def collate_fn(batch):
    return {
        'pil_image': [item['pil_image'] for item in batch],
        'mask': torch.stack([item['mask'] for item in batch]),
        'filename': [item['filename'] for item in batch]
    }


class OneFormerBaseline(nn.Module):
    def __init__(self, model_name="shi-labs/oneformer_ade20k_swin_large"):
        super().__init__()
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.oneformer = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.oneformer.eval()
        
        for param in self.oneformer.parameters():
            param.requires_grad = False
    
    def forward(self, batch_or_images):
        if isinstance(batch_or_images, dict):
            pil_images = batch_or_images['pil_image']
        else:
            pil_images = batch_or_images
            
        with torch.no_grad():
            inputs = self.processor(
                images=pil_images,
                task_inputs=["semantic"] * len(pil_images),
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
            
            outputs = self.oneformer(**inputs)
            
            predictions = self.processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(512, 512)] * len(pil_images)
            )
            
            return {
                'predictions': torch.stack(predictions),
                'raw_outputs': outputs
            }


class PhysicsEnhancer(nn.Module):
    def __init__(self, num_classes=150):
        super().__init__()
        self.num_classes = num_classes
        
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        self.class_enhancer = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # Initialize final layer to zero
        final_conv = self.class_enhancer[-1]
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)
        
        self.enhancement_scale = nn.Parameter(torch.tensor(0.01))
    
    def forward(self, rgb_images):
        features = self.spatial_features(rgb_images)
        attention = self.spatial_attention(features)
        attended_features = features * attention
        enhancement_map = self.class_enhancer(attended_features)
        
        enhancement_map = F.interpolate(
            enhancement_map, 
            size=rgb_images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        attention_upsampled = F.interpolate(
            attention,
            size=rgb_images.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        
        return torch.tanh(enhancement_map) * attention_upsampled * self.enhancement_scale


class PhysicsEnhancedOneFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.oneformer_baseline = OneFormerBaseline()
        self.physics_enhancer = PhysicsEnhancer()
        self.alpha = nn.Parameter(torch.tensor(-8.0))
        
    def forward(self, batch):
        pil_images = batch['pil_image']
        
        oneformer_outputs = self.oneformer_baseline(pil_images)
        oneformer_preds = oneformer_outputs['predictions']
        outputs = oneformer_outputs['raw_outputs']
        
        # Convert PIL to tensors
        rgb_tensors = []
        for img in pil_images:
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)
            rgb_tensors.append(img_tensor)
        rgb_batch = torch.stack(rgb_tensors)
        
        enhancement_maps = self.physics_enhancer(rgb_batch)
        
        # Extract logits from OneFormer
        with torch.no_grad():
            mask_logits = outputs.class_queries_logits
            masks = outputs.masks_queries_logits
            
            B, Q, C = mask_logits.shape
            H, W = 512, 512
            
            if masks.shape[-2:] != (H, W):
                masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
            
            mask_probs = masks.sigmoid()
            class_probs = mask_logits[..., :-1].softmax(dim=-1)
            
            base_logits = torch.einsum('bqhw,bqc->bchw', mask_probs, class_probs)
            base_logits = torch.log(base_logits.clamp(min=1e-8))
        
        # Apply physics enhancement
        alpha = torch.sigmoid(self.alpha)
        enhanced_logits = base_logits + alpha * enhancement_maps
        final_preds = enhanced_logits.argmax(dim=1)
        
        return {
            'predictions': final_preds,
            'oneformer_predictions': oneformer_preds,
            'logits': enhanced_logits
        }


class Evaluator:
    def __init__(self, num_classes=150):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.sample_count = 0
        
    def update(self, predictions, targets):
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        for pred, target in zip(predictions, targets):
            valid_mask = target != 255
            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]
            
            for t, p in zip(target_valid.flatten(), pred_valid.flatten()):
                if t < self.num_classes and p < self.num_classes:
                    self.confusion_matrix[t, p] += 1
            
            self.sample_count += 1
    
    def get_metrics(self):
        ious = []
        for i in range(self.num_classes):
            intersection = self.confusion_matrix[i, i]
            union = (self.confusion_matrix[i, :].sum() + 
                    self.confusion_matrix[:, i].sum() - 
                    self.confusion_matrix[i, i])
            
            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(np.nan)
        
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0
        pixel_acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        
        return {
            'mean_iou': mean_iou,
            'pixel_accuracy': pixel_acc,
            'per_class_iou': ious,
            'valid_classes': len(valid_ious)
        }


def evaluate_model(model, dataloader, max_batches=None):
    model.eval()
    evaluator = Evaluator()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            masks = batch['mask'].to(device)
            outputs = model(batch)
            predictions = outputs['predictions']
            
            evaluator.update(predictions, masks)
    
    return evaluator.get_metrics()


def train_model(model, train_loader, val_loader, num_epochs=10, patience=3):
    for param in model.oneformer_baseline.parameters():
        param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    baseline_metrics = evaluate_model(model.oneformer_baseline, val_loader, max_batches=50)
    best_miou = baseline_metrics['mean_iou']
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_miou': [],
        'alpha': [],
        'enhancement_scale': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        model.oneformer_baseline.eval()
        
        epoch_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            masks = batch['mask'].to(device)
            masks = torch.clamp(masks, min=0, max=255)
            masks[masks >= 150] = 255
            
            outputs = model(batch)
            loss = criterion(outputs['logits'], masks)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        history['alpha'].append(torch.sigmoid(model.alpha).item())
        history['enhancement_scale'].append(model.physics_enhancer.enhancement_scale.item())
        
        val_metrics = evaluate_model(model, val_loader, max_batches=50)
        val_miou = val_metrics['mean_iou']
        history['val_miou'].append(val_miou)
        
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Val mIoU={val_miou:.4f}")
        
        scheduler.step(val_miou)
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Final evaluation
    if (OUTPUT_DIR / 'best_model.pth').exists():
        model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth'))
    
    final_metrics = evaluate_model(model, val_loader, max_batches=100)
    
    results = {
        'baseline_metrics': baseline_metrics,
        'final_metrics': final_metrics,
        'improvement': {
            'miou': (final_metrics['mean_iou'] - baseline_metrics['mean_iou']) * 100,
            'pixel_acc': (final_metrics['pixel_accuracy'] - baseline_metrics['pixel_accuracy']) * 100
        },
        'training_history': history
    }
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, results


def main():
    # Create datasets
    train_dataset = ADE20KDataset(
        KAGGLE_ADE20K_PATH,
        split='training',
        image_size=512,
        max_samples=500
    )
    
    val_dataset = ADE20KDataset(
        KAGGLE_ADE20K_PATH,
        split='validation',
        image_size=512,
        max_samples=200
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    model = PhysicsEnhancedOneFormer().to(device)
    
    # Train model
    trained_model, results = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=10,
        patience=4
    )
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Baseline mIoU: {results['baseline_metrics']['mean_iou']*100:.2f}%")
    print(f"Enhanced mIoU: {results['final_metrics']['mean_iou']*100:.2f}%")
    print(f"Improvement: {results['improvement']['miou']:+.2f}%")
    
    return trained_model, results


if __name__ == "__main__":
    model, results = main()