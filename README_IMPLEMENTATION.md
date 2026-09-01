# MT-ISA Implementation Guide
## Multi-Task Learning with LLMs for Implicit Sentiment Analysis

Complete implementation of the MT-ISA framework from the paper with Ollama for local LLMs.

---

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (in one terminal)
ollama serve

# 3. Pull a model (in another terminal)
ollama pull mistral

# 4. Process data
python data_loader.py --train-path data/semeval2014/Restaurants_Train.xml \
                      --test-path data/semeval2014/Restaurants_Test.xml

# 5. Generate auxiliary data (takes time - LLM inference)
python auxiliary_generator.py --input data/processed/train_implicit.json \
                             --model mistral \
                             --max-instances 50  # Start small!

# 6. Train model
python train_mt_isa.py --train-data data/processed/train_implicit.json \
                       --aux-data data/auxiliary/train_implicit_aux.json \
                       --batch-size 16 \
                       --num-epochs 10

# 7. Evaluate
python evaluate_mt_isa.py --model-path models/outputs/checkpoint-best.pt \
                          --test-data data/processed/test_implicit.json
```

---

## Detailed Step-by-Step Guide

### Step 1: Environment Setup

#### 1.1 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

#### 1.3 Verify GPU (Optional)
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Step 2: Ollama Setup

#### 2.1 Install Ollama
Download from: https://ollama.ai/

#### 2.2 Start Ollama Server
```bash
# In a dedicated terminal, start Ollama
ollama serve

# You should see: listening on 127.0.0.1:11434
```

#### 2.3 Download a Model
In another terminal:
```bash
# Option 1: Fast, smaller model (recommended for testing)
ollama pull mistral

# Option 2: Better quality
ollama pull neural-chat

# Option 3: Larger, slower
ollama pull llama2

# Verify
ollama list  # Should show your downloaded model
```

#### 2.4 Test Ollama Connection
```bash
python ollama_client.py  # Should print successful connection message
```

### Step 3: Data Preparation

#### 3.1 Download SemEval-2014 Dataset

**Option A: Manual Download**
1. Go to: https://www.aclweb.org/anthology/S14-2004/
2. Download SemEval-2014 ABSA dataset
3. Extract to `data/semeval2014/`

Expected files:
```
data/semeval2014/
├── Restaurants_Train.xml
├── Restaurants_Test.xml
├── Laptops_Train.xml
└── Laptops_Test.xml
```

**Option B: Use Script**
```bash
# Create data directory
mkdir -p data/semeval2014

# Note: You still need to manually download from the link above
# Then extract the files to data/semeval2014/
```

#### 3.2 Process Dataset
```bash
# Process Restaurants dataset
python data_loader.py \
    --train-path data/semeval2014/Restaurants_Train.xml \
    --test-path data/semeval2014/Restaurants_Test.xml \
    --output-dir data/processed/ \
    --split-type implicit

# Or for all instances
python data_loader.py \
    --train-path data/semeval2014/Restaurants_Train.xml \
    --test-path data/semeval2014/Restaurants_Test.xml \
    --output-dir data/processed/ \
    --split-type all
```

This creates:
- `data/processed/train_implicit.json` - Training instances (implicit sentiments only)
- `data/processed/test_implicit.json` - Test instances
- `data/processed/train_full.json` - All instances (for reference)

Output example:
```
============================================================
TRAINING SET STATISTICS
============================================================
Total instances: 727
  - Implicit: 727
  - Explicit: 0
Polarity distribution: {'positive': 405, 'negative': 217, 'neutral': 105}
Aspect categories: {'RESTAURANT#GENERAL': 93, 'FOOD#QUALITY': 187, ...}
```

---

### Step 4: Generate Auxiliary Data (Phase 1)

**⚠️ WARNING: This step is SLOW (LLM inference)**
- ~30-60 seconds per instance with Ollama
- 100 instances ≈ 1-2 hours
- **Start with `--max-instances 10` for testing!**

#### 4.1 Quick Test (10 instances)
```bash
python auxiliary_generator.py \
    --input data/processed/train_implicit.json \
    --output data/auxiliary/train_implicit_aux.json \
    --model mistral \
    --max-epochs 10 \
    --max-instances 10 \
    --verbose
```

Expected output:
```
============================================================
AUXILIARY DATA GENERATION STATISTICS
============================================================
Total instances: 10
Converged: 9 (90.0%)
Refinement iterations:
  Mean: 1.89
  Min: 1
  Max: 10
Aspect confidence:
  Mean: 0.78
  Min: 0.52
  Max: 0.92
Opinion confidence:
  Mean: 0.71
  Min: 0.50
  Max: 0.88
============================================================
```

#### 4.2 Full Dataset (when ready)
```bash
# Remove --max-instances to process all
python auxiliary_generator.py \
    --input data/processed/train_implicit.json \
    --output data/auxiliary/train_implicit_aux.json \
    --model mistral \
    --max-epochs 10
```

**To speed up:**
- Use smaller model: `ollama pull mistral` (fastest)
- Reduce max refinement epochs: `--max-epochs 5`
- Run in parallel (requires code modification)

#### 4.3 Inspect Generated Data
```bash
# View first few examples
python -c "
import json
with open('data/auxiliary/train_implicit_aux.json') as f:
    data = json.load(f)
    for i, item in enumerate(data[:3]):
        print(f'Instance {i}:')
        print(f'  Aspect: {item[\"aspect\"]} (conf: {item[\"aspect_confidence\"]:.2f})')
        print(f'  Opinion: {item[\"opinion\"]} (conf: {item[\"opinion_confidence\"]:.2f})')
        print(f'  Converged: {item[\"converged\"]} (iterations: {item[\"refinement_iterations\"]})')
        print()
"
```

---

### Step 5: Train Model

#### 5.1 Test Run (Small batch)
```bash
python train_mt_isa.py \
    --train-data data/processed/train_implicit.json \
    --aux-data data/auxiliary/train_implicit_aux.json \
    --val-data data/processed/test_implicit.json \
    --batch-size 8 \
    --num-epochs 2 \
    --learning-rate 1e-5 \
    --d-awl-strategy input \
    --t-awl-version alf2 \
    --model-name google/flan-t5-base \
    --output-dir models/outputs/ \
    --early-stopping-patience 5
```

Expected output:
```
Device: cuda:0
Train samples: 727
Val samples: 318
Initializing model...
Model 'google/flan-t5-base' available
✓ Model 'google/flan-t5-base' available

============================================================
Epoch 1/2
============================================================
Training: 100%|████████████| 92/92 [15:23<00:00, 10.04s/batch]
Train loss: 3.2145
Validating: 100%|████████| 40/40 [03:12<00:00, 4.80s/batch]
Val loss: 2.8934
Task weights: {'aspect': 1.23, 'opinion': 0.95, 'polarity': 0.87}
✓ Best model saved (val_loss=2.8934)

============================================================
Epoch 2/2
============================================================
...
```

#### 5.2 Configuration Options

**Model Sizes** (smaller = faster, larger = better):
```bash
# Small (250M params, fast)
--model-name google/flan-t5-small

# Base (250M params, default)
--model-name google/flan-t5-base

# Large (780M params)
--model-name google/flan-t5-large

# XL (3B params)
--model-name google/flan-t5-xl

# XXL (13B params, requires high GPU memory)
--model-name google/flan-t5-xxl
```

**D-AWL Strategies**:
```bash
# Input scaling (good for small models)
--d-awl-strategy input

# Output loss weighting (good for large models)
--d-awl-strategy output

# Both (risky, can under-perform)
--d-awl-strategy input_output
```

**T-AWL Versions**:
```bash
# Stable (recommended)
--t-awl-version alf2

# Unstable (for comparison)
--t-awl-version alf1
```

**Training Parameters**:
```bash
# Batch size (reduce if GPU OOM)
--batch-size 16  # 32 for large GPU, 8 for small GPU

# Learning rate
--learning-rate 1e-5  # Standard for fine-tuning

# Epochs
--num-epochs 20  # 10-20 typical

# Gradient accumulation (if batch size too small)
--gradient-accumulation-steps 2  # Doubles effective batch size

# Early stopping
--early-stopping-patience 10  # Stop if no improvement for 10 epochs
```

#### 5.3 Full Training
```bash
python train_mt_isa.py \
    --train-data data/processed/train_implicit.json \
    --aux-data data/auxiliary/train_implicit_aux.json \
    --val-data data/processed/test_implicit.json \
    --batch-size 16 \
    --num-epochs 20 \
    --learning-rate 1e-5 \
    --d-awl-strategy input \
    --t-awl-version alf2 \
    --model-name google/flan-t5-base \
    --output-dir models/outputs/
```

#### 5.4 Monitor Training

**Check training progress**:
```bash
# View loss curves
python -c "
import json
with open('models/outputs/history.json') as f:
    history = json.load(f)
    print('Epoch\tTrain Loss\tVal Loss')
    for i, (tl, vl) in enumerate(zip(history['train_loss'], history['val_loss'])):
        print(f'{i+1}\t{tl:.4f}\t\t{vl:.4f}')
"
```

**Saved checkpoints**:
```bash
ls -lh models/outputs/
# checkpoint-best.pt       - Best model (lowest val loss)
# checkpoint-epoch-1.pt    - Epoch 1 checkpoint
# checkpoint-epoch-2.pt    - Epoch 2 checkpoint
# history.json             - Training curves
```

---

### Step 6: Evaluation

#### 6.1 Evaluate on Test Set
```bash
python evaluate_mt_isa.py \
    --model-path models/outputs/checkpoint-best.pt \
    --test-data data/processed/test_implicit.json \
    --output models/predictions.json \
    --show-errors 10
```

Expected output:
```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 0.8245 (262/318)
F1 Score (macro): 0.7934
Precision (macro): 0.8102
Recall (macro): 0.7856
============================================================

============================================================
EXAMPLE ERRORS (5 of 56)
============================================================

1. Instance: rest_train_0001
   Sentence: I went there for bite size food with my date.
   Target: food
   Gold: positive
   Predicted: neutral (confidence: 0.42)

2. Instance: rest_train_0005
   ...
```

#### 6.2 Interpret Predictions
```bash
# View predictions JSON
python -c "
import json
with open('models/predictions.json') as f:
    data = json.load(f)
    metrics = data['metrics']
    print(f\"Accuracy: {metrics['accuracy']:.4f}\")
    print(f\"F1 Score: {metrics['f1_score']:.4f}\")
    
    # View first prediction
    pred = data['predictions'][0]
    print(f\"\\nExample prediction:\")
    print(f\"  Sentence: {pred['sentence']}\")
    print(f\"  Target: {pred['target']}\")
    print(f\"  Gold: {pred['gold_polarity']}\")
    print(f\"  Predicted: {pred['predicted_polarity']} (conf: {pred['confidence']:.2f})\")
"
```

---

## Troubleshooting

### 1. Ollama Connection Error
```
Error: Cannot connect to Ollama at http://localhost:11434
```

**Solution:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Verify model is downloaded
ollama list
```

### 2. Model Not Found
```
Error: Model 'mistral' not found
```

**Solution:**
```bash
ollama pull mistral  # Download the model
ollama list          # Verify
```

### 3. Out of Memory (GPU)
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size: `--batch-size 8` (from 16)
- Use smaller model: `--model-name google/flan-t5-small`
- Enable gradient accumulation: add `--gradient-accumulation-steps 2`
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""` before running

### 4. Slow LLM Inference
```
Auxiliary generation takes too long
```

**Solutions:**
- Use faster model: `ollama pull mistral`
- Reduce refinement epochs: `--max-epochs 5`
- Process fewer instances: `--max-instances 50`
- Increase max tokens timeout in `ollama_client.py`

### 5. CUDA Memory During Training
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```bash
# Option 1: Reduce batch size
--batch-size 8

# Option 2: Accumulate gradients
--gradient-accumulation-steps 2

# Option 3: Use CPU
export CUDA_VISIBLE_DEVICES=""
python train_mt_isa.py ...
```

### 6. Missing Data Files
```
FileNotFoundError: data/processed/train_implicit.json
```

**Solution:**
```bash
# Run data loader first
python data_loader.py --train-path data/semeval2014/Restaurants_Train.xml \
                      --test-path data/semeval2014/Restaurants_Test.xml
```

---

## Mathematical Details

### Loss Functions

**Primary Task (Polarity)**:
```
L_p = -log p(gold_polarity | input)
```

**Auxiliary Tasks (Aspect & Opinion)**:
```
L_a = -Σ_t log p(aspect_token_t | input)
L_o = -Σ_t log p(opinion_token_t | input)
```

**Data-Level AWL (D-AWL)**:
```
Input Strategy:    ẽ_i = c_i * e_i
Output Strategy:   L = c_i * log p(y|x)
```

**Task-Level AWL (T-AWL)**:
```
L = (1/σ₁²)L_a + (1/σ₂²)L_o + (1/σ₃²)L_p + Σ ln(σ_k² + 1)
```

Where:
- σ_k² = learnable task uncertainty parameters
- c_i = confidence score from LLM generation [0.5-1.0]
- e_i = token embeddings

---

## Reproduction Results

Expected accuracies on SemEval-2014 Restaurant (Implicit):

| Model | D-AWL | T-AWL | Accuracy |
|-------|-------|-------|----------|
| Flan-T5-base | Input | ALF2 | ~82% |
| Flan-T5-large | Output | ALF2 | ~85% |
| Flan-T5-xl | Input | ALF2 | ~87% |
| Flan-T5-xxl | Output | ALF2 | ~89% |

**Note:** Results depend on auxiliary data quality and random initialization.

---

## Code Structure

```
├── data_loader.py             # Parse SemEval-2014 XML
├── ollama_client.py           # Ollama LLM client
├── auxiliary_generator.py     # Phase 1: Self-refine + confidence
├── mt_isa_model.py           # Core model + D-AWL + T-AWL
├── train_mt_isa.py           # Phase 6: Training loop
├── evaluate_mt_isa.py        # Evaluation + metrics
├── requirements.txt           # Dependencies
└── README_IMPLEMENTATION.md   # This file
```

---

## Next Steps

1. **Optimize Ollama:**
   - Try different models (mistral, neural-chat, llama2)
   - Adjust temperature and max_tokens

2. **Experiment with Models:**
   - Compare D-AWL strategies (input vs output)
   - Try different Flan-T5 sizes

3. **Improve Results:**
   - Collect more instances (auxiliary generation is slow)
   - Fine-tune prompts in `auxiliary_generator.py`
   - Adjust confidence thresholds

4. **Production Deployment:**
   - Save model and tokenizer
   - Create inference API
   - Benchmark latency

---

## Reference

Original Paper:
> Lai, W., Xie, H., Xu, G., & Li, Q. (2025).
> "Multi-Task Learning With LLMs for Implicit Sentiment Analysis:
> Data-Level and Task-Level Automatic Weight Learning"
> IEEE Transactions on Knowledge and Data Engineering, Vol. 38, No. 1.

---

## Support

If you encounter issues:

1. Check **Troubleshooting** section above
2. Verify all dependencies: `pip list | grep -E "torch|transformers|ollama"`
3. Test individual components:
   ```bash
   python ollama_client.py      # Test Ollama
   python mt_isa_model.py       # Test model
   python data_loader.py        # Test data loading
   ```
4. Review logs in `models/outputs/`

---

**Happy training! 🚀**
