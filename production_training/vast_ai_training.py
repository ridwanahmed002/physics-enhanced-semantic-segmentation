#!/usr/bin/env python3
"""
Physics-Enhanced OneFormer: Production Version
Comprehensive ablation study for IR-guided semantic segmentation
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
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')


class Config:
    def __init__(self):
        self.data_root = "./ADEChallengeData2016"
        self.output_root = "./thesis_results"
        self.checkpoint_dir = f"{self.output_root}/checkpoints"
        self.results_dir = f"{self.output_root}/results"
        self.figures_dir = f"{self.output_root}/figures"
        
        # Model configuration
        self.image_size = 512
        self.num_classes = 150
        
        # Training configuration
        self.batch_size = 16
        self.learning_rate = 1e-6
        self.weight_decay = 0.01
        self.num_epochs = 30
        self.warmup_epochs = 2
        self.grad_clip = 1.0
        self.num_workers = 8
        self.use_amp = True
        self.max_samples = None
        
        # Physics configuration
        self.init_alpha = -3.0
        self.temperature_range = (273, 323)
        
        # Create directories
        for path in [self.checkpoint_dir, self.results_dir, self.figures_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


class ThermalAwarePhysicsEnhancer(nn.Module):
    def __init__(self, num_classes=150, temperature_range=(273, 323)):
        super().__init__()
        self.num_classes = num_classes
        self.temp_min, self.temp_max = temperature_range
        
        self.thermal_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        
        self.enhancement_scale = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, rgb_images, thermal_images=None):
        if thermal_images is None:
            thermal_images = rgb_images.mean(dim=1, keepdim=True)
        
        thermal_feat = self.thermal_encoder(thermal_images)
        rgb_feat = self.rgb_encoder(rgb_images)
        
        attended = thermal_feat + rgb_feat
        combined = torch.cat([thermal_feat, attended], dim=1)
        enhancement = self.decoder(combined)
        
        return torch.tanh(enhancement) * self.enhancement_scale


class SimplePhysicsEnhancer(nn.Module):
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
        
        nn.init.zeros_(self.class_enhancer[-1].weight)
        nn.init.zeros_(self.class_enhancer[-1].bias)
        
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


class PhysicsOneFormer(nn.Module):
    def __init__(self, config, ablation_mode='full'):
        super().__init__()
        self.config = config
        self.ablation_mode = ablation_mode
        
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.oneformer = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.oneformer.eval()
        
        for param in self.oneformer.parameters():
            param.requires_grad = False
        
        if ablation_mode == 'baseline':
            self.physics_enhancer = None
            self.alpha = None
        elif ablation_mode == 'physics_fixed':
            self.physics_enhancer = ThermalAwarePhysicsEnhancer()
            self.alpha = nn.Parameter(torch.tensor(-2.197), requires_grad=False)
        elif ablation_mode == 'physics_no_thermal':
            self.physics_enhancer = SimplePhysicsEnhancer()
            self.alpha = nn.Parameter(torch.tensor(config.init_alpha))
        else:
            self.physics_enhancer = ThermalAwarePhysicsEnhancer()
            self.alpha = nn.Parameter(torch.tensor(config.init_alpha))  # sigmoid(-3) ≈ 0.047
            
    def extract_oneformer_logits(self, pil_images):
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            inputs = self.processor(
                images=pil_images,
                task_inputs=["semantic"] * len(pil_images),
                return_tensors="pt"
            )
            
            device = next(self.oneformer.parameters()).device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            outputs = self.oneformer(**inputs)
            
            predictions = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(512, 512)] * len(pil_images)
            )
            
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
            
            del mask_probs, class_probs, masks, mask_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return base_logits, torch.stack(predictions)
    
    def forward(self, batch):
        pil_images = batch['pil_image']
        
        base_logits, oneformer_preds = self.extract_oneformer_logits(pil_images)
        
        if self.physics_enhancer is None:
            return {
                'predictions': oneformer_preds,
                'logits': base_logits,
                'oneformer_predictions': oneformer_preds,
                'enhancement_applied': False
            }
        
        device = base_logits.device
        
        rgb_tensors = []
        thermal_tensors = []
        for img in pil_images:
            img_array = np.array(img).astype(np.float32) / 255.0
            rgb_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)
            rgb_tensors.append(rgb_tensor)
            thermal_tensor = rgb_tensor.mean(dim=0, keepdim=True)
            thermal_tensors.append(thermal_tensor)
            
        rgb_batch = torch.stack(rgb_tensors)
        thermal_batch = torch.stack(thermal_tensors)
        
        if self.ablation_mode == 'physics_no_thermal':
            enhancement = self.physics_enhancer(rgb_batch)
        else:
            enhancement = self.physics_enhancer(rgb_batch, thermal_batch)
        
        alpha = torch.sigmoid(self.alpha) if self.alpha.requires_grad else torch.sigmoid(self.alpha).detach()
        if enhancement.shape[-2:] != base_logits.shape[-2:]:
            enhancement = F.interpolate(enhancement, size=base_logits.shape[-2:], mode='bilinear', align_corners=False)
        
        # Ensure shapes match before enhancement
        assert enhancement.shape == base_logits.shape, f"Shape mismatch: {enhancement.shape} vs {base_logits.shape}"
        
        enhancement = torch.clamp(enhancement, min=-0.1, max=0.1)
        enhanced_logits = base_logits + alpha * enhancement
        enhanced_logits = torch.clamp(enhanced_logits, min=-100, max=100)
        
        enhanced_preds = enhanced_logits.argmax(dim=1)
        
        return {
            'predictions': enhanced_preds,
            'logits': enhanced_logits,
            'oneformer_predictions': oneformer_preds,
            'enhancement_applied': True,
            'alpha': alpha.item(),
            'enhancement_scale': self.physics_enhancer.enhancement_scale.item()
        }


class ADE20KDataset:
    def __init__(self, root_path, split='training', image_size=512, max_samples=None):
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
        
        return {
            'pil_image': img_resized,
            'mask': torch.from_numpy(mask_converted).long(),
            'filename': img_path.stem,
        }


def collate_fn(batch):
    return {
        'pil_image': [item['pil_image'] for item in batch],
        'mask': torch.stack([item['mask'] for item in batch]),
        'filename': [item['filename'] for item in batch],
    }


class Evaluator:
    def __init__(self, num_classes=150):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.per_image_ious = []
        
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
            
            image_ious = []
            for cls in range(self.num_classes):
                intersection = ((pred_valid == cls) & (target_valid == cls)).sum()
                union = ((pred_valid == cls) | (target_valid == cls)).sum()
                if union > 0:
                    image_ious.append(intersection / union)
            if image_ious:
                self.per_image_ious.append(np.mean(image_ious))
    
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
        
        class_accs = []
        for i in range(self.num_classes):
            class_total = self.confusion_matrix[i, :].sum()
            if class_total > 0:
                class_accs.append(self.confusion_matrix[i, i] / class_total)
        mean_acc = np.mean(class_accs) if class_accs else 0
        
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        fw_iou = np.sum([freq[i] * ious[i] for i in range(self.num_classes) if not np.isnan(ious[i])])
        
        return {
            'mean_iou': mean_iou,
            'pixel_accuracy': pixel_acc,
            'mean_accuracy': mean_acc,
            'frequency_weighted_iou': fw_iou,
            'per_class_iou': ious,
            'valid_classes': len(valid_ious),
            'per_image_mean_iou': np.mean(self.per_image_ious) if self.per_image_ious else 0
        }


def train_with_ablation(config, ablation_mode='full', experiment_name=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining: {ablation_mode}")
    print(f"Experiment: {experiment_name or ablation_mode}")
    print(f"Device: {device}")
    
    from torch.utils.data import DataLoader
    
    train_dataset = ADE20KDataset(
        config.data_root,
        split='training',
        image_size=config.image_size,
        max_samples=config.max_samples
    )
    
    val_dataset = ADE20KDataset(
        config.data_root,
        split='validation',
        image_size=config.image_size,
        max_samples=config.max_samples // 4 if config.max_samples else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    model = PhysicsOneFormer(config, ablation_mode).to(device)
    
    if ablation_mode == 'baseline':
        print("Evaluating OneFormer baseline...")
        evaluator = Evaluator()
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating baseline"):
                masks = batch['mask'].to(device)
                outputs = model(batch)
                evaluator.update(outputs['predictions'], masks)
        
        metrics = evaluator.get_metrics()
        
        results = {
            'ablation_mode': ablation_mode,
            'metrics': metrics,
            'trainable_params': 0,
            'training_time': 0
        }
        
        save_path = Path(config.results_dir) / f"{experiment_name or ablation_mode}_results.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Baseline mIoU: {metrics['mean_iou']*100:.2f}%")
        return results
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = len(train_loader) * config.warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.0)
    
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    best_miou = 0
    history = {'train_loss': [], 'val_miou': [], 'learning_rate': []}
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        model.train()
        model.oneformer.eval()
        
        epoch_loss = 0
        valid_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            masks = batch['mask'].to(device)
            
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                outputs = model(batch)
                loss = criterion(outputs['logits'], masks)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss {loss.item()}, skipping batch {batch_idx}")
                optimizer.zero_grad()
                continue
                
            optimizer.zero_grad()
            
            if config.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.grad_clip)
                optimizer.step()
            
            scheduler.step()
            epoch_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
        
        avg_loss = epoch_loss / max(valid_batches, 1)
        history['train_loss'].append(avg_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        evaluator = Evaluator()
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                masks = batch['mask'].to(device)
                outputs = model(batch)
                evaluator.update(outputs['predictions'], masks)
        
        metrics = evaluator.get_metrics()
        val_miou = metrics['mean_iou']
        history['val_miou'].append(val_miou)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, mIoU={val_miou:.4f} ({val_miou*100:.2f}%)")
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_miou': best_miou
            }, Path(config.checkpoint_dir) / f"{experiment_name or ablation_mode}_best.pth")
            print(f"  New best model saved: {best_miou*100:.2f}%")
        
        # Clear GPU memory after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    training_time = time.time() - start_time
    
    final_evaluator = Evaluator()
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final evaluation"):
            masks = batch['mask'].to(device)
            outputs = model(batch)
            final_evaluator.update(outputs['predictions'], masks)
    
    final_metrics = final_evaluator.get_metrics()
    
    results = {
        'ablation_mode': ablation_mode,
        'experiment_name': experiment_name or ablation_mode,
        'best_miou': best_miou,
        'final_metrics': final_metrics,
        'history': history,
        'trainable_params': trainable_params,
        'training_time': training_time,
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    }
    
    save_path = Path(config.results_dir) / f"{experiment_name or ablation_mode}_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTraining complete!")
    print(f"Best mIoU: {best_miou*100:.2f}%")
    print(f"Training time: {training_time/60:.1f} minutes")
    
    return results


def run_complete_ablation_study(config):
    ablation_experiments = [
        ('baseline', 'OneFormer Baseline (No Physics)'),
        ('physics_fixed', 'Physics Enhancement (Fixed α=0.1)'),
        ('physics_no_thermal', 'Physics Enhancement (No Thermal)'),
        ('full', 'Full Model (Thermal-Aware Physics)')
    ]
    
    all_results = {}
    
    for ablation_mode, description in ablation_experiments:
        print(f"\nABLATION: {description}")
        
        try:
            results = train_with_ablation(config, ablation_mode)
            all_results[ablation_mode] = results
            
            summary_path = Path(config.results_dir) / 'ablation_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Ablation {ablation_mode} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        generate_results_visualization(all_results, config)
    
    return all_results


def generate_results_visualization(results, config):
    try:
        figures_dir = Path(config.figures_dir)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = []
        mious = []
        
        for mode, result in results.items():
            methods.append(mode.replace('_', ' ').title())
            if 'best_miou' in result:
                mious.append(result['best_miou'] * 100)
            else:
                mious.append(result['metrics']['mean_iou'] * 100)
        
        bars = ax.bar(methods, mious, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Mean IoU (%)')
        ax.set_title('Ablation Study Results: Physics Enhancement Impact')
        ax.set_ylim(0, max(mious) * 1.1)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figures_dir / 'ablation_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for mode, result in results.items():
            if 'history' in result and result['history']['train_loss']:
                epochs = range(1, len(result['history']['train_loss']) + 1)
                ax1.plot(epochs, result['history']['train_loss'], 
                        label=mode.replace('_', ' ').title(), linewidth=2)
                ax2.plot(epochs, [x*100 for x in result['history']['val_miou']], 
                        label=mode.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation mIoU (%)')
        ax2.set_title('Validation Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Physics-Enhanced OneFormer Experiments')
    parser.add_argument('--mode', choices=['full_ablation', 'single'], default='single')
    parser.add_argument('--ablation', choices=['baseline', 'physics_fixed', 'physics_no_thermal', 'full'], 
                       default='full')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--data_root', type=str, default='./ADEChallengeData2016')
    parser.add_argument('--output_root', type=str, default='./thesis_results')
    
    args = parser.parse_args()
    
    config = Config()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.data_root = args.data_root
    config.output_root = args.output_root
    
    print("PHYSICS-ENHANCED ONEFORMER: EXPERIMENTS")
    print(f"Mode: {args.mode}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    if args.mode == 'single':
        print(f"\nRunning single ablation: {args.ablation}")
        results = train_with_ablation(config, args.ablation)
        print(f"\nExperiment complete: {args.ablation}")
        if 'best_miou' in results:
            print(f"Best mIoU: {results['best_miou']*100:.2f}%")
        return results
    
    elif args.mode == 'full_ablation':
        print(f"\nRunning complete ablation study...")
        results = run_complete_ablation_study(config)
        
        print("\nABLATION STUDY COMPLETE")
        
        for mode, result in results.items():
            if 'best_miou' in result:
                miou = result['best_miou']
            else:
                miou = result['metrics']['mean_iou']
            print(f"{mode:20}: {miou*100:6.2f}% mIoU")
        
        if 'baseline' in results:
            baseline_miou = results['baseline']['metrics']['mean_iou']
            print(f"\nImprovements over baseline:")
            for mode, result in results.items():
                if mode != 'baseline':
                    if 'best_miou' in result:
                        miou = result['best_miou']
                    else:
                        miou = result['metrics']['mean_iou']
                    improvement = (miou - baseline_miou) * 100
                    print(f"{mode:20}: {improvement:+6.2f}%")
        
        return results


if __name__ == "__main__":
    results = main()
    print("\nExperiments completed!")
    print(f"Results saved to: ./thesis_results/")