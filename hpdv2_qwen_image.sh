export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=2

GPU_NUM=2
MODEL_PATH="data/qwenimage"
OUTPUT_DIR_PREFIX="data/qwenimage/rl_embeddings"

prompt_dirs=(
    "/home/dingruiyi/HPDv2/benchmark/drawbench.txt"
)

for prompt_dir in "${prompt_dirs[@]}"; do
    echo "Processing $(basename "${prompt_dir}" .txt)..."
    OUTPUT_DIR="${OUTPUT_DIR_PREFIX}_$(basename "${prompt_dir}" .txt)"
    torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
        fastvideo/data_preprocess/preprocess_qwenimage_embedding.py \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --prompt_dir $prompt_dir
done
