python data/cached_fineweb10B.py 9
torchrun --standalone --nproc_per_node=2 src/modded_nanogpt/train.py
