"""
MT-ISA Model Implementation
Includes Flan-T5 backbone with D-AWL, T-AWL, and automatic loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
import math
from typing import Optional, Tuple, Dict
import logging


logger = logging.getLogger(__name__)


class DataLevelAWL(nn.Module):
    """
    Data-Level Automatic Weight Learning
    Three strategies: Input (I), Output (O), Input-Output (I-O)
    """
    
    def __init__(self, strategy: str = 'input'):
        """
        Args:
            strategy: 'input', 'output', or 'input_output'
        """
        super().__init__()
        assert strategy in ['input', 'output', 'input_output'], \
            f"Unknown D-AWL strategy: {strategy}"
        self.strategy = strategy
    
    def apply_input_strategy(
        self,
        embeddings: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Input Strategy (I): Scale embeddings by confidence
        e_i = c_i * Emb(x_i)
        
        Args:
            embeddings: [batch_size, seq_len, embed_dim]
            confidence_scores: [batch_size]
            
        Returns:
            Scaled embeddings [batch_size, seq_len, embed_dim]
        """
        # Reshape confidence for broadcasting: [batch_size] -> [batch_size, 1, 1]
        confidence_scores = confidence_scores.view(-1, 1, 1)
        scaled_embeddings = embeddings * confidence_scores
        
        return scaled_embeddings
    
    def apply_output_strategy(
        self,
        loss: torch.Tensor,
        confidence_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Output Strategy (O): Scale loss by confidence
        L = c_i * log p(y)
        
        Args:
            loss: Scalar loss
            confidence_scores: [batch_size]
            
        Returns:
            Weighted loss
        """
        # Average confidence across batch
        mean_confidence = confidence_scores.mean()
        weighted_loss = loss * mean_confidence
        
        return weighted_loss
    
    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        confidence_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply D-AWL strategy
        
        For Input strategy: requires embeddings and confidence_scores
        For Output strategy: requires loss and confidence_scores
        """
        if self.strategy == 'input':
            assert embeddings is not None and confidence_scores is not None
            return self.apply_input_strategy(embeddings, confidence_scores)
        
        elif self.strategy == 'output':
            assert loss is not None and confidence_scores is not None
            return self.apply_output_strategy(loss, confidence_scores)
        
        elif self.strategy == 'input_output':
            # Both input and output scaling
            if embeddings is not None and confidence_scores is not None:
                embeddings = self.apply_input_strategy(embeddings, confidence_scores)
            if loss is not None and confidence_scores is not None:
                loss = self.apply_output_strategy(loss, confidence_scores)
            
            return embeddings if embeddings is not None else loss


class TaskLevelAWL(nn.Module):
    """
    Task-Level Automatic Weight Learning using Homoscedastic Uncertainty
    Implements automatic loss functions ALF1 and ALF2
    """
    
    def __init__(self, num_tasks: int = 3, alf_version: str = 'alf2'):
        """
        Args:
            num_tasks: Number of tasks (aspect, opinion, polarity)
            alf_version: 'alf1' or 'alf2'
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.alf_version = alf_version
        
        # Learnable task uncertainty parameters σ_k^2
        # Initialize to 1.0 (equal weight)
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(
        self,
        losses: Tuple[torch.Tensor, ...],
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute combined loss with task-level weights
        
        Args:
            losses: Tuple of task losses (L_aspect, L_opinion, L_polarity)
            reduction: 'mean' or 'sum'
            
        Returns:
            Combined loss
        """
        assert len(losses) == self.num_tasks, \
            f"Expected {self.num_tasks} losses, got {len(losses)}"
        
        # Convert log_sigma_sq to sigma_sq (always positive)
        sigma_sq = torch.exp(self.log_sigma_sq)
        
        # Compute weighted losses: (1/σ_k^2) * L_k
        weighted_losses = []
        for k, loss_k in enumerate(losses):
            weight_k = 1.0 / (sigma_sq[k] + 1e-8)  # Add epsilon for stability
            weighted_loss = weight_k * loss_k
            weighted_losses.append(weighted_loss)
        
        # Sum weighted losses
        total_weighted_loss = sum(weighted_losses)
        
        # Add regularization term
        if self.alf_version == 'alf1':
            # ALF1: log(σ²) - can be unstable
            reg_term = torch.sum(self.log_sigma_sq)
        
        elif self.alf_version == 'alf2':
            # ALF2: ln(σ² + 1) - more stable
            reg_term = torch.sum(torch.log(sigma_sq + 1))
        
        else:
            raise ValueError(f"Unknown ALF version: {self.alf_version}")
        
        # Total loss
        total_loss = total_weighted_loss + reg_term
        
        if reduction == 'mean':
            # Normalize by number of tasks
            total_loss = total_loss / self.num_tasks
        
        return total_loss
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights for inspection"""
        sigma_sq = torch.exp(self.log_sigma_sq).detach()
        weights = {
            'aspect': float(1.0 / (sigma_sq[0] + 1e-8)),
            'opinion': float(1.0 / (sigma_sq[1] + 1e-8)),
            'polarity': float(1.0 / (sigma_sq[2] + 1e-8))
        }
        return weights


class MTISAModel(nn.Module):
    """
    Multi-Task Learning for Implicit Sentiment Analysis with Automatic Weight Learning
    
    Architecture:
    - Shared encoder (Flan-T5 encoder)
    - Task-specific decoders:
        1. Aspect inference
        2. Opinion inference
        3. Polarity classification
    - D-AWL for data-level uncertainty
    - T-AWL for task-level uncertainty
    """
    
    def __init__(
        self,
        model_name: str = 'google/flan-t5-small',
        d_awl_strategy: str = 'input',
        t_awl_version: str = 'alf2',
        num_polarity_classes: int = 3
    ):
        """
        Args:
            model_name: Pretrained model name
            d_awl_strategy: 'input', 'output', or 'input_output'
            t_awl_version: 'alf1' or 'alf2'
            num_polarity_classes: Number of polarity classes
        """
        super().__init__()
        
        self.model_name = model_name
        self.d_awl_strategy = d_awl_strategy
        self.t_awl_version = t_awl_version
        self.num_polarity_classes = num_polarity_classes
        
        # Load pretrained T5 model
        self.backbone = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Get model dimensions
        self.hidden_dim = self.backbone.config.d_model
        
        # D-AWL module
        self.d_awl = DataLevelAWL(strategy=d_awl_strategy)
        
        # T-AWL module
        self.t_awl = TaskLevelAWL(num_tasks=3, alf_version=t_awl_version)
        
        # Task-specific classification head for polarity
        # (for fine-grained control if needed)
        self.polarity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_polarity_classes)
        )
        
        logger.info(f"Initialized MT-ISA model:")
        logger.info(f"  Backbone: {model_name}")
        logger.info(f"  D-AWL strategy: {d_awl_strategy}")
        logger.info(f"  T-AWL version: {t_awl_version}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
    
    def forward(
        self,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        opinion_input_ids: torch.Tensor,
        opinion_attention_mask: torch.Tensor,
        polarity_input_ids: torch.Tensor,
        polarity_attention_mask: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        aspect_confidence: Optional[torch.Tensor] = None,
        opinion_labels: Optional[torch.Tensor] = None,
        opinion_confidence: Optional[torch.Tensor] = None,
        polarity_labels: Optional[torch.Tensor] = None,
        polarity_label_id: Optional[torch.Tensor] = None,

        return_losses: bool = True
    ) -> Dict:
        """
        Forward pass for all three tasks
        
        Returns:
            Dictionary with losses and predictions for each task
        """
        outputs = {}
        
        # ==================== ASPECT TASK ====================
        if aspect_input_ids is not None:
            aspect_output = self.backbone(
                input_ids=aspect_input_ids,
                attention_mask=aspect_attention_mask,
                labels=aspect_labels,
                decoder_attention_mask=aspect_labels != -100 if aspect_labels is not None else None
            )
            
            aspect_loss = aspect_output.loss
            
            # Apply D-AWL if confidence scores provided
            if aspect_confidence is not None and aspect_loss is not None:
                if self.d_awl_strategy == 'output':
                    aspect_loss = self.d_awl(
                        loss=aspect_loss,
                        confidence_scores=aspect_confidence
                    )
            
            outputs['aspect_loss'] = aspect_loss
            outputs['aspect_logits'] = aspect_output.logits
        
        # ==================== OPINION TASK ====================
        if opinion_input_ids is not None:
            opinion_output = self.backbone(
                input_ids=opinion_input_ids,
                attention_mask=opinion_attention_mask,
                labels=opinion_labels,
                decoder_attention_mask=opinion_labels != -100 if opinion_labels is not None else None
            )
            
            opinion_loss = opinion_output.loss
            
            # Apply D-AWL if confidence scores provided
            if opinion_confidence is not None and opinion_loss is not None:
                if self.d_awl_strategy == 'output':
                    opinion_loss = self.d_awl(
                        loss=opinion_loss,
                        confidence_scores=opinion_confidence
                    )
            
            outputs['opinion_loss'] = opinion_loss
            outputs['opinion_logits'] = opinion_output.logits
        
        # ==================== POLARITY TASK ====================
        if polarity_input_ids is not None:
            encoder_output = self.backbone.encoder(
                input_ids=polarity_input_ids,
                attention_mask=polarity_attention_mask,
            )
            cls_hidden = encoder_output.last_hidden_state[:, 0, :]
            polarity_logits = self.polarity_head(cls_hidden)

            outputs['polarity_logits'] = polarity_logits

            if polarity_label_id is not None:
                polarity_loss = nn.functional.cross_entropy(
                    polarity_logits,
                    polarity_label_id.long()
                )
                outputs['polarity_loss'] = polarity_loss
            elif polarity_labels is not None:
                polarity_loss = self.backbone(
                    input_ids=polarity_input_ids,
                    attention_mask=polarity_attention_mask,
                    labels=polarity_labels,
                    decoder_attention_mask=polarity_labels != -100 if polarity_labels is not None else None
                ).loss
                outputs['polarity_loss'] = polarity_loss
        
        # ==================== TASK-LEVEL AWL ====================
        if return_losses and all(k in outputs for k in ['aspect_loss', 'opinion_loss', 'polarity_loss']):
            aspect_loss = outputs['aspect_loss']
            opinion_loss = outputs['opinion_loss']
            polarity_loss = outputs['polarity_loss']
            
            # Compute combined loss with T-AWL
            combined_loss = self.t_awl(
                losses=(aspect_loss, opinion_loss, polarity_loss),
                reduction='mean'
            )
            
            outputs['combined_loss'] = combined_loss
            
            # Store task weights for monitoring
            outputs['task_weights'] = self.t_awl.get_task_weights()
        
        return outputs
    
    def predict_polarity(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict polarity class for input
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        Returns:
            Predicted class logits or probabilities
        """
        # Get encoder output
        encoder_output = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_hidden = encoder_output.last_hidden_state
        
        # Use [CLS]-like token (first token)
        cls_hidden = encoder_hidden[:, 0, :]
        
        # Pass through classification head
        logits = self.polarity_head(cls_hidden)
        
        return logits
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'd_awl_strategy': self.d_awl_strategy,
            't_awl_version': self.t_awl_version,
            'task_weights': self.t_awl.get_task_weights()
        }


def test_model():
    """Test script for MT-ISA model"""
    print("Testing MT-ISA model...")
    
    # Initialize model
    model = MTISAModel(
        model_name='google/flan-t5-small',
        d_awl_strategy='input',
        t_awl_version='alf2'
    )
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Dummy tokenized inputs
    input_ids = torch.randint(0, 32100, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    labels = torch.randint(0, 32100, (batch_size, seq_len)).to(device)
    polarity_label_id = torch.randint(0, 3, (batch_size,)).to(device)
    confidence = torch.rand(batch_size).to(device) * 0.4 + 0.6  # [0.6, 1.0]

    # Forward pass
    print("\nForward pass...")
    with torch.no_grad():
        outputs = model(
            aspect_input_ids=input_ids,
            aspect_attention_mask=attention_mask,
            aspect_labels=labels,
            aspect_confidence=confidence,

            opinion_input_ids=input_ids,
            opinion_attention_mask=attention_mask,
            opinion_labels=labels,
            opinion_confidence=confidence,

            polarity_input_ids=input_ids,
            polarity_attention_mask=attention_mask,
            polarity_label_id=polarity_label_id
        )
    
    print(f"Combined loss: {outputs['combined_loss'].item():.4f}")
    print(f"Task weights: {outputs['task_weights']}")
    print("\n✓ Model test successful!")


if __name__ == '__main__':
    test_model()
