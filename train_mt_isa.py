"""
Training Script for MT-ISA Model
Complete training pipeline with validation and checkpointing
"""

import os
import json
import re
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import T5Tokenizer
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple

from mt_isa_model import MTISAModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_usable_auxiliary_text(
    text: Optional[str],
    sentence: str,
    require_sentence_match: bool = False
) -> bool:
    """Accept only English extracted text, not generated verdicts."""
    if not text or re.search(r'[^\x00-\x7F]', text):
        return False

    normalized = text.lower().strip(' .\"\'')
    if normalized in {
        'positive', 'negative', 'neutral', 'insufficient data available',
        'insufficient information', 'unable to determine'
    }:
        return False
    if len(text.split()) > 5:
        return False
    return not require_sentence_match or normalized in sentence.lower()


class AspectSentimentDataset(Dataset):
    """Dataset for aspect sentiment analysis with auxiliary data"""
    
    def __init__(
        self,
        instances: List[Dict],
        auxiliary_data: Optional[List[Dict]] = None,
        tokenizer=None,
        max_length: int = 256
    ):
        """
        Args:
            instances: List of instances with keys:
                      [id, sentence, target, gold_polarity, is_implicit, opinion_term]
            auxiliary_data: List of auxiliary data dicts with keys:
                           [instance_id, aspect, aspect_confidence, opinion, opinion_confidence]
            tokenizer: T5 tokenizer
            max_length: Max sequence length for tokenization
        """
        self.instances = instances
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Index auxiliary data by instance ID
        self.auxiliary_by_id = {}
        if auxiliary_data:
            for aux in auxiliary_data:
                self.auxiliary_by_id[aux['instance_id']] = aux
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx) -> Dict:
        """Get dataset item"""
        instance = self.instances[idx]
        instance_id = instance['id']
        
        # Basic info
        sentence = instance['sentence']
        target = instance['target']  # aspect term (could be 'NULL' for implicit)
        gold_polarity = instance['gold_polarity']
        
        # Get auxiliary data if available
        aux_data = self.auxiliary_by_id.get(instance_id, None)

        raw_aspect = (
            aux_data.get('aspect')
            if aux_data and aux_data.get('aspect') is not None
            else target
        )
        aspect_text = raw_aspect if is_usable_auxiliary_text(raw_aspect, sentence) else target
        aspect_conf = (
            aux_data.get('aspect_confidence', 0.5)
            if aux_data and aux_data.get('aspect_confidence') is not None
            else 0.5
        )

        raw_opinion = (
            aux_data.get('opinion')
            if aux_data and aux_data.get('opinion') is not None
            else (instance.get('opinion_term') or '')
        )
        opinion_text = (
            raw_opinion
            if is_usable_auxiliary_text(raw_opinion, sentence)
            else (instance.get('opinion_term') or '')
        )
        opinion_conf = (
            aux_data.get('opinion_confidence', 0.5)
            if aux_data and aux_data.get('opinion_confidence') is not None
            else 0.5
        )

        # Prepare inputs
        item = {
            'instance_id': instance_id,
            'sentence': sentence,
            'target': target,
            'gold_polarity': gold_polarity,
            'aspect_text': aspect_text,
            'aspect_confidence': torch.tensor(aspect_conf, dtype=torch.float32),
            'opinion_text': opinion_text,
            'opinion_confidence': torch.tensor(opinion_conf, dtype=torch.float32)
        }
        
        # Tokenize inputs for each task
        if self.tokenizer:
            # ASPECT TASK: "aspect: sentence [SEP] target"
            aspect_input = f"aspect: {sentence} [SEP] {target}"
            aspect_tokens = self.tokenizer(
                aspect_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize aspect label
            aspect_label = self.tokenizer(
                aspect_text,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            aspect_labels = aspect_label['input_ids'].squeeze(0)
            aspect_labels[aspect_labels == self.tokenizer.pad_token_id] = -100
            
            item['aspect_input_ids'] = aspect_tokens['input_ids'].squeeze(0)
            item['aspect_attention_mask'] = aspect_tokens['attention_mask'].squeeze(0)
            item['aspect_labels'] = aspect_labels
            
            # OPINION TASK: "opinion: sentence [SEP] target [SEP] aspect"
            opinion_input = f"opinion: {sentence} [SEP] {target} [SEP] {aspect_text}"
            opinion_tokens = self.tokenizer(
                opinion_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize opinion label
            opinion_label = self.tokenizer(
                opinion_text if opinion_text else 'none',
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            opinion_labels = opinion_label['input_ids'].squeeze(0)
            opinion_labels[opinion_labels == self.tokenizer.pad_token_id] = -100
            
            item['opinion_input_ids'] = opinion_tokens['input_ids'].squeeze(0)
            item['opinion_attention_mask'] = opinion_tokens['attention_mask'].squeeze(0)
            item['opinion_labels'] = opinion_labels
            
            # POLARITY TASK: generate the polarity label as text
            polarity_input = f"sentiment polarity: {sentence} [SEP] {target}"
            polarity_tokens = self.tokenizer(
                polarity_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            polarity_label = self.tokenizer(
                gold_polarity.lower(),
                max_length=8,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(0)
            polarity_label[polarity_label == self.tokenizer.pad_token_id] = -100

            item['polarity_input_ids'] = polarity_tokens['input_ids'].squeeze(0)
            item['polarity_attention_mask'] = polarity_tokens['attention_mask'].squeeze(0)
            item['polarity_labels'] = polarity_label
        
        return item


class Trainer:
    """Trainer class for MT-ISA model"""
    
    def __init__(
        self,
        model: MTISAModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        output_dir: str = 'models/outputs/',
        num_epochs: int = 20,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = 10
    ):
        """
        Args:
            model: MT-ISA model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: GPU/CPU device
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            early_stopping_patience: Patience for early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'task_weights': []}
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                aspect_input_ids=batch['aspect_input_ids'],
                aspect_attention_mask=batch['aspect_attention_mask'],
                aspect_labels=batch['aspect_labels'],
                aspect_confidence=batch['aspect_confidence'],

                opinion_input_ids=batch['opinion_input_ids'],
                opinion_attention_mask=batch['opinion_attention_mask'],
                opinion_labels=batch['opinion_labels'],
                opinion_confidence=batch['opinion_confidence'],

                polarity_input_ids=batch['polarity_input_ids'],
                polarity_attention_mask=batch['polarity_attention_mask'],
                polarity_labels=batch['polarity_labels'],

                return_losses=True
            )
            
            loss = outputs['combined_loss']
            
            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            # Optimization step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}'})
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                outputs = self.model(
                    aspect_input_ids=batch['aspect_input_ids'],
                    aspect_attention_mask=batch['aspect_attention_mask'],
                    aspect_labels=batch['aspect_labels'],
                    aspect_confidence=batch['aspect_confidence'],

                    opinion_input_ids=batch['opinion_input_ids'],
                    opinion_attention_mask=batch['opinion_attention_mask'],
                    opinion_labels=batch['opinion_labels'],
                    opinion_confidence=batch['opinion_confidence'],

                    polarity_input_ids=batch['polarity_input_ids'],
                    polarity_attention_mask=batch['polarity_attention_mask'],
                    polarity_labels=batch['polarity_labels'],

                    return_losses=True
                )
                
                loss = outputs['combined_loss']
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        task_weights = self.model.t_awl.get_task_weights()
        
        return avg_loss, task_weights
    
    def train(self):
        """Train for num_epochs"""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output dir: {self.output_dir}")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss, task_weights = self.validate()
            logger.info(f"Val loss: {val_loss:.4f}")
            logger.info(f"Task weights: {task_weights}")
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['task_weights'].append(task_weights)
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f'checkpoint-epoch-{epoch+1}.pt'
            self.save_checkpoint(checkpoint_path)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = self.output_dir / 'checkpoint-best.pt'
                self.save_checkpoint(best_path)
                logger.info(f"✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping triggered!")
                    break
        
        # Save final history
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"\nTraining complete! History saved to {history_path}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'epoch': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)


def main():
    parser = argparse.ArgumentParser(description='Train MT-ISA model')
    parser.add_argument('--train-data', type=str, default='data/processed/train_implicit.json',
                        help='Training data JSON')
    parser.add_argument('--aux-data', type=str, default='data/auxiliary/train_implicit_aux.json',
                        help='Auxiliary data JSON')
    parser.add_argument('--val-data', type=str, default='data/processed/test_implicit.json',
                        help='Validation data JSON')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--d-awl-strategy', type=str, default='input',
                        choices=['input', 'output', 'input_output'],
                        help='D-AWL strategy')
    parser.add_argument('--t-awl-version', type=str, default='alf2',
                        choices=['alf1', 'alf2'],
                        help='T-AWL version')
    parser.add_argument('--model-name', type=str, default='google/flan-t5-base',
                        help='Pretrained model name')
    parser.add_argument('--output-dir', type=str, default='models/outputs/',
                        help='Output directory')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info("Loading data...")
    with open(args.train_data, 'r') as f:
        train_instances = json.load(f)
    with open(args.val_data, 'r') as f:
        val_instances = json.load(f)
    
    # Load auxiliary data
    aux_data = None
    if Path(args.aux_data).exists():
        with open(args.aux_data, 'r') as f:
            aux_data = json.load(f)
        logger.info(f"Loaded {len(aux_data)} auxiliary data items")
    
    # Create datasets
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    
    train_dataset = AspectSentimentDataset(
        train_instances, aux_data, tokenizer
    )
    val_dataset = AspectSentimentDataset(
        val_instances, None, tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = MTISAModel(
        model_name=args.model_name,
        d_awl_strategy=args.d_awl_strategy,
        t_awl_version=args.t_awl_version
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Scheduler
    total_steps = len(train_loader) * args.num_epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=total_steps)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
