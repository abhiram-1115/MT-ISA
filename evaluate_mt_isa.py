"""
Evaluation Script for MT-ISA Model
Computes metrics and generates predictions
"""

import json
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
from typing import Dict, List, Optional, Tuple

from mt_isa_model import MTISAModel
from train_mt_isa import AspectSentimentDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolarityPredictor:
    """Make polarity predictions using trained model"""
    
    def __init__(self, model_path: str, device: torch.device):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: Device to use
        """
        self.device = device
        
        # Load checkpoint
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load model (use the default small backbone used for training)
        self.model = MTISAModel(
            model_name='google/flan-t5-small',  # Adjust if different
            d_awl_strategy='input',
            t_awl_version='alf2'
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
        
        # Polarity to ID mapping
        self.polarity_to_id = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }
        self.id_to_polarity = {v: k for k, v in self.polarity_to_id.items()}
    
    def predict(self, sentence: str, target: str) -> Tuple[str, float]:
        """
        Predict polarity for sentence + target
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            
        Returns:
            (predicted_polarity, confidence)
        """
        # Format input
        input_text = f"sentiment polarity: {sentence} [SEP] {target}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding='max_length'
        ).to(self.device)
        
        with torch.no_grad():
            # Get encoder output
            encoder_output = self.model.backbone.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Get CLS hidden state
            cls_hidden = encoder_output.last_hidden_state[:, 0, :]
            
            # Classify
            logits = self.model.polarity_head(cls_hidden)
            
            # Get prediction
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_id].item()
        
        predicted_polarity = self.id_to_polarity[pred_id]
        
        return predicted_polarity, confidence


class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self, model_path: str, device: torch.device):
        self.predictor = PolarityPredictor(model_path, device)
        self.device = device
    
    def evaluate_dataset(
        self,
        test_instances: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate on test dataset
        
        Args:
            test_instances: Test instances with gold labels
            output_path: Optional path to save predictions
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Evaluating on {len(test_instances)} instances")
        
        predictions = []
        gold_labels = []
        predicted_labels = []
        correct = 0
        
        for instance in test_instances:
            sentence = instance['sentence']
            target = instance['target']
            gold_polarity = instance['gold_polarity'].lower()
            
            # Predict
            pred_polarity, confidence = self.predictor.predict(sentence, target)
            
            # Store
            gold_labels.append(gold_polarity)
            predicted_labels.append(pred_polarity.lower())
            
            predictions.append({
                'instance_id': instance['id'],
                'sentence': sentence,
                'target': target,
                'gold_polarity': gold_polarity,
                'predicted_polarity': pred_polarity.lower(),
                'confidence': confidence,
                'correct': pred_polarity.lower() == gold_polarity
            })
            
            if pred_polarity.lower() == gold_polarity:
                correct += 1
        
        # Compute metrics
        accuracy = accuracy_score(gold_labels, predicted_labels)
        f1 = f1_score(gold_labels, predicted_labels, average='macro', zero_division=0)
        precision = precision_score(gold_labels, predicted_labels, average='macro', zero_division=0)
        recall = recall_score(gold_labels, predicted_labels, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'correct': correct,
            'total': len(test_instances)
        }
        
        # Save predictions
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'predictions': predictions
                }, f, indent=2)
            logger.info(f"Predictions saved to {output_path}")
        
        return metrics, predictions
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
        print(f"F1 Score (macro): {metrics['f1_score']:.4f}")
        print(f"Precision (macro): {metrics['precision']:.4f}")
        print(f"Recall (macro): {metrics['recall']:.4f}")
        print("=" * 60)
    
    def print_errors(self, predictions: List[Dict], num_errors: int = 10):
        """Print example errors"""
        errors = [p for p in predictions if not p['correct']]
        
        if not errors:
            print("\n✓ No errors!")
            return
        
        print(f"\n{'='*60}")
        print(f"EXAMPLE ERRORS ({min(num_errors, len(errors))} of {len(errors)})")
        print(f"{'='*60}")
        
        for idx, error in enumerate(errors[:num_errors]):
            print(f"\n{idx+1}. Instance: {error['instance_id']}")
            print(f"   Sentence: {error['sentence']}")
            print(f"   Target: {error['target']}")
            print(f"   Gold: {error['gold_polarity']}")
            print(f"   Predicted: {error['predicted_polarity']} "
                  f"(confidence: {error['confidence']:.2f})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MT-ISA model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/processed/test_implicit.json',
                        help='Test data JSON')
    parser.add_argument('--output', type=str, default='models/predictions.json',
                        help='Output predictions JSON')
    parser.add_argument('--show-errors', type=int, default=10,
                        help='Number of errors to show')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_instances = json.load(f)
    
    # Initialize evaluator
    evaluator = Evaluator(args.model_path, device)
    
    # Evaluate
    metrics, predictions = evaluator.evaluate_dataset(
        test_instances,
        output_path=args.output
    )
    
    # Print results
    evaluator.print_metrics(metrics)
    evaluator.print_errors(predictions, num_errors=args.show_errors)


if __name__ == '__main__':
    main()
