# Optimization Notes for Prompt Tuning

## Current Results (100 epochs, 10 prompts, 10 defect samples)

```
Prompt-Tuned: AUROC = 0.7672, AP = 0.8723
Zero-Shot:    AUROC = 0.7635, AP = 0.8705
Improvement:  +0.48% AUROC, +0.20% AP
```

## Issues Identified

### 1. **Baseline AUROC Discrepancy**
- **Expected** (from SIMILARITY_ANALYSIS_RESULTS.md): 0.8674
- **Actual**: 0.7635
- **Difference**: -0.1039 (-12%)

**Possible causes**:
- Different k-NN parameter (need to check original)
- Prompt modulation affecting both models
- Different similarity computation method

### 2. **Limited Improvement**
The prompt tuning only provides +0.48% improvement, which is minimal.

## Optimization Strategies

### Strategy 1: Reduce Prompt Influence
**Problem**: The 0.1 scaling factor for prompt modulation might be too aggressive

**Solution**: Try smaller scaling factors (0.01, 0.05)

```python
# Current
patch_embed = patch_embed + 0.1 * prompt_effect

# Try
patch_embed = patch_embed + 0.01 * prompt_effect  # More conservative
```

### Strategy 2: Different Prompt Mechanism
**Problem**: Current approach modulates embeddings post-hoc, which may disrupt pretrained features

**Solutions**:
1. **Learnable scaling per feature dimension**
2. **Gating mechanism** - let model learn when to apply prompts
3. **Separate prompt pathway** - add prompts without modifying base embeddings

### Strategy 3: Save Best Checkpoint
**Problem**: Final separability (0.2383) is lower than peak (0.3601 at epoch 40)

**Solution**: Implement early stopping or save best checkpoint

```python
if current_separability > best_separability:
    best_separability = current_separability
    save_checkpoint(model, 'best_prompts.pt')
```

### Strategy 4: More Training Data
**Problem**: Only using 10 defect samples

**Solution**: Try 20-50 defect samples for more robust learning

### Strategy 5: Different Loss Function
**Problem**: Separability loss may not translate to AUROC improvement

**Solution**: Try contrastive loss or direct AUROC optimization

### Strategy 6: Larger Prompts
**Problem**: 10 prompts may not provide enough capacity

**Solution**: Try 20-50 prompts

### Strategy 7: Check Original Baseline
**Most Important**: Run your original zero-shot script to confirm baseline

```bash
./venv/bin/python scripts/simple_anomaly_detection.py \
    --category screw \
    --batch-size 32
```

## Recommended Optimization Order

1. **First**: Verify baseline with original script
2. **Second**: Reduce prompt influence (0.01 scaling)
3. **Third**: Save best checkpoint (epoch 40)
4. **Fourth**: Train with more defect samples (20)
5. **Fifth**: Try larger model (vitb16) or more prompts (20)

## Expected Improvements

With optimizations:
- Target AUROC: 0.85-0.90 (assuming baseline ~0.867)
- Expected gain: +2-5% over corrected baseline
- Best case: Match or exceed your few-shot results

## Quick Test Command

Test with reduced prompt influence:

```bash
# Edit prompt_model.py line 118:
# Change: patch_embed = patch_embed + 0.1 * prompt_effect
# To:     patch_embed = patch_embed + 0.01 * prompt_effect

./venv/bin/python prompt_based_feature_adaption/train_prompts.py \
    --category screw \
    --num-prompts 20 \
    --num-defect-samples 20 \
    --num-epochs 150
```
