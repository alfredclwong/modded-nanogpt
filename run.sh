torchrun --standalone --nproc_per_node=8 train_gpt2.py --gpt.flex_kernel_consumer True \
    --train.batch_size 16 --train.sequence_length --train.sequence_length 32768
