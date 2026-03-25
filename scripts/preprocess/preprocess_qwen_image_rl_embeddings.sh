
GPU_NUM=4
MODEL_PATH="data/qwenimage"
OUTPUT_DIR_PREFIX="data/qwenimage/rl_embeddings"

prompt_dirs=(
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/anime.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/concept-art.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/drawbench.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/paintings.txt"
    "/mnt/shared-storage-user/mineru4s/dingruiyi/hpdv2_test/benchmark/photo.txt"
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
