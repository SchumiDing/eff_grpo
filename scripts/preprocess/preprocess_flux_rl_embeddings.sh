GPU_NUM=4 # 2,4,8
MODEL_PATH="data/flux"
OUTPUT_DIR="data/rl_embeddings"

export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/mnt/shared-storage-user/mineru4s/dingruiyi/share/cuda-12.8/lib64:$LD_LIBRARY_PATH
export RANK=0
export WORLD_SIZE=4



prompt_dirs=(
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/anime.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/concept-art.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/drawbench.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/paintings.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/photo.txt"
    # "/mnt/shared-storage-user/mineru4s/dingruiyi/DanceGRPO/assets/hpdv2.txt"
)

OUTPUT_DIR_PREFIX="data/flux/rl_embeddings"

for prompt_dir in "${prompt_dirs[@]}"; do
    echo "Processing $(basename "${prompt_dir}" .txt)..."
    OUTPUT_DIR="${OUTPUT_DIR_PREFIX}_$(basename "${prompt_dir}" .txt)"
    torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir $prompt_dir
done
