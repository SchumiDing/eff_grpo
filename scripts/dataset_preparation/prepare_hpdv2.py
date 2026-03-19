import json
from datasets import load_dataset
import os

def prepare_hpdv2_prompts(output_path="assets/prompts.txt", limit=1000):
    print("Loading HPDv2 dataset...")
    # Loading the train split of HPDv2
    dataset = load_dataset("ymhao/HPDv2", split='train', trust_remote_code=True)
    
    print(f"Extracting first {limit} unique prompts...")
    prompts = set()
    for item in dataset:
        prompts.add(item['prompt'])
        if len(prompts) >= limit:
            break
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in sorted(list(prompts)):
            f.write(prompt + "\n")
            
    print(f"Successfully saved {len(prompts)} prompts to {output_path}")

if __name__ == "__main__":
    prepare_hpdv2_prompts()
