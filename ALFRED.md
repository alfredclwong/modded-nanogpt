Goal:

- Build up from the original nanoGPT to the SOTA modded-nanoGPT
- Collect intermediate profiling and ablation results

Steps:

- RoPE, QK-Norm, RELU^2
- Muon optimiser
- Quantisation
- Skip connections
- Flash Attention 3, long-short sliding window attention, YaRN
- Distributed training, data loading
- Batch eos alignment, max doc length
- Fine-grained gradient accumulation
- Sparse attention gate
- Custom triton kernels
- Misc: softcap logits, initialisation, untie head, extra attention embeddings
- ??: back out first contributions from 8 layers, Polar Express, smear module for 1 token lookback

Plan:

0. Base
   - Use vram_factor (VF) to decrease mb_size/seq_len and increase grad_accum_steps, maintaining same tokens per step
   - VF>16 will reduce seq_len from 2048, undesirable but not too detrimental
1. RoPE, QK-Norm, RELU^2 (wasn't actually using RELU^2 until #3)
   - Simple improvements
2. bfloat16 linear+embed
   - Compute RoPE cache (and loss/gradients?) in float32
3. Multi-GPU
   - Distributed data generator
   - Broadcast initial weights
   - Distributed Adam (f16), param shape groups, gradient sync (important)
4. BOS alignment
5. Muon

Runs
- stoic-resonance-3, mmm4, VF=32 T=1024, #0, 111s/step
- bright-darkness-5, mmm4, VF=32 T=1024, #1, 135s/step
- winter-voice-6, colab t4, VF=4, #1, 77s/step, ~14.1GB max mem, 77s/step
- wise-blaze-7, colab t4, VF=4, #2 (bf16 loss), ~13.3GB max mem, 107s/step
- sage-planet-12, 2x rtx3060, VF=4, #3, not compiled, 7.6GB max mem, 8s/step
- stellar-universe-13, 2x rtx3060, VF=4, #3, compiled, 6.6 (5.45 cuda) GB max mem, 7s/step
- valiant-firebrand, 2x rtx3060, VF=2, #3 (1e-3 lr), compiled, crashed for unknown reason (see error.txt)
- 
