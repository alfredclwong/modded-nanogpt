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
4. Hyper-Connections
   - DistAdam chunk padding
   - f32 dynamic hyper-connections
   - lr_mul and/or scaled norm
5. Attention upgrade
   - variable length
   - yarn
   - long-short sliding window, warmup
6. BOS alignment
7. Muon

Runs
- stoic-resonance-3, mmm4, VF=32 T=1024, #0, 111s/step
- bright-darkness-5, mmm4, VF=32 T=1024, #1, 135s/step
- winter-voice-6, colab t4, VF=4, #1, 77s/step, ~14.1GB max mem, 77s/step
- wise-blaze-7, colab t4, VF=4, #2 (bf16 loss), ~13.3GB max mem, 107s/step
- sage-planet-12, 2x rtx3060, VF=4, #3, not compiled, 7.6GB max mem, 8s/step
- (1) stellar-universe-13, 2x rtx3060, VF=4, #3, compiled, 6.6 (5.5 cuda) GB max mem, 7s/step
- valiant-firebrand, 2x rtx3060, VF=2, #3 (1e-3 lr), compiled, crashed for unknown reason (see error.txt)
- visionary-microwave-15, 2x rtx3070, VF=8, #4 (no-mlp init bug, no-mean bug), 7.1 (5.8 cuda) GB max mem, 18s/step
- mild-wildflower-18, 2x rtx3070, VF=8, #4, 5.9 GB, 17s/step
- fallen-donkey-19, 2x rtx5070, VF=2, #4 (hc=False), 10.3 GB, 2.7s/step
- silver-snow-20, 2x rtx5060, VF=4, #4 (rate=4), 10.4 GB, 9s/step
- clear-shape-21, 2x rtx5070, VF=4, #4 (rate=4, dynamic=False, empty init fix), 8.5 GB, 4.0s/step
- lilac-morning-22, 2x rtx5070, VF=4, #4 (rate=4), 10.5 GB, 9.4s/step
- sleek-salad-23, 2x rtx5070, VF=4, #4, (rate=-2, frac empty init fix), 8.6 GB, 7.7s/step
- playful-monkey-25, 2x rtx5070, VF=2, #4 (rate=1), 8.4 GB, 7.4s/step
- dutiful-vortex-26, 2x rtx5070, VF=4, #4 (rate=-2, hc.lr_mul=8), ,
- zesty-voice-27, 2x rtx5070, VF=4, #4 (rate=4, hc.lr_mul=8), 10.4 (8.75) GB, 9.1s/step
- (2a) astral-wildflower-28, 2x rtx5070, VF=4, #4 (rate=4, hc.lr_mul=100), 10.4 GB, 9.0s
- fluent-capybara-29, 2x rtx5070, VF=4, #4 (rate=4, rms w/ scale), 10.8 GB, 9.4s
- (3) polished-tree-33, 2x rtx4070, VF=4, #4 (rate=-2, shc.lr_mul=1000, dhc.lr_mul=100, rms w/ scale), 8.8 GB, 10.7s
- driven-waterfall-34, 2x rtx4070, VF=2, #4 (zero-init, 1 rope), 10.4 GB, 3.0s
- (1) efficient-dragon-36, 2x rtx4070, VF=2, #4, 10.2 GB, 3.0s
- (3) ethereal-elevator-38, 2x rtx4070, VF=4, #4, 8.4 GB, 9.2s
- (2b) youthful-cherry-39, 2x rtx4070, VF=4, #4, 9.1 GB, 10s
- should go back to 100/10 lr_muls

Highlights
1. Baseline multi-GPU run
2. DHCx4 (a) DHCx2 (b)
3. DFCx2
