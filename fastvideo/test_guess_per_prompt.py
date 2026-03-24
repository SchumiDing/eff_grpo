"""
测试脚本:验证guess逻辑是否按照每个prompt组来选择
"""
import torch

def test_guess_logic():
    """模拟guess选择逻辑"""
    # 模拟参数
    num_prompts = 3
    num_generations = 4
    B = num_prompts * num_generations  # 12
    num_guess = 2  # 每个prompt组内guess 2个
    num_steps = 5
    
    device = 'cpu'
    
    # 初始化
    last_step_guessed_mask = torch.zeros(B, device=device, dtype=torch.bool)
    cumulative_guess_count = torch.zeros(B, device=device, dtype=torch.long)
    
    print(f"总样本数: {B} (prompts={num_prompts}, generations={num_generations})")
    print(f"每个prompt组内guess数量: {num_guess}")
    print(f"总步数: {num_steps}\n")
    
    for step in range(num_steps):
        print(f"{'='*60}")
        print(f"Step {step + 1}/{num_steps}")
        print(f"{'='*60}")
        
        # 最后一步不猜
        if step == num_steps - 1:
            guessed_mask = torch.zeros(B, device=device, dtype=torch.bool)
            print("最后一步,不进行guess")
        else:
            guessed_mask = torch.zeros(B, device=device, dtype=torch.bool)
            
            # 对每个prompt组分别处理
            for prompt_idx in range(num_prompts):
                start_idx = prompt_idx * num_generations
                end_idx = start_idx + num_generations
                
                print(f"\nPrompt {prompt_idx} (rollouts {start_idx}-{end_idx-1}):")
                
                # 找到当前prompt组内上一步没被猜过的样本
                group_can_guess_mask = ~last_step_guessed_mask[start_idx:end_idx]
                group_can_guess_indices = torch.where(group_can_guess_mask)[0]
                
                print(f"  可以guess的rollouts (组内索引): {group_can_guess_indices.tolist()}")
                
                # 从中选择guess次数最少的 num_guess 个来猜
                num_guess_in_group = min(num_guess, len(group_can_guess_indices))
                
                if num_guess_in_group > 0:
                    # 获取候选样本的累计guess次数
                    group_can_guess_counts = cumulative_guess_count[start_idx:end_idx][group_can_guess_indices]
                    print(f"  这些rollouts的累计guess次数: {group_can_guess_counts.tolist()}")
                    
                    # 按guess次数排序,选择次数最少的num_guess个
                    sorted_indices = torch.argsort(group_can_guess_counts)[:num_guess_in_group]
                    group_guess_indices = group_can_guess_indices[sorted_indices]
                    
                    print(f"  选中guess的rollouts (组内索引): {group_guess_indices.tolist()}")
                    
                    # 转换为全局索引
                    global_guess_indices = start_idx + group_guess_indices
                    print(f"  选中guess的rollouts (全局索引): {global_guess_indices.tolist()}")
                    
                    # 更新累计guess次数和mask
                    cumulative_guess_count[global_guess_indices] += 1
                    guessed_mask[global_guess_indices] = True
                else:
                    print(f"  本组无可guess的rollouts")
        
        # 更新last_step_guessed_mask
        last_step_guessed_mask = guessed_mask.clone()
        
        print(f"\n当前步guess的rollouts (全局): {torch.where(guessed_mask)[0].tolist()}")
        print(f"当前步infer的rollouts (全局): {torch.where(~guessed_mask)[0].tolist()}")
        print(f"累计guess次数: {cumulative_guess_count.tolist()}\n")
    
    print(f"{'='*60}")
    print("最终统计")
    print(f"{'='*60}")
    print(f"每个rollout的总guess次数: {cumulative_guess_count.tolist()}")
    
    # 按prompt组统计
    for prompt_idx in range(num_prompts):
        start_idx = prompt_idx * num_generations
        end_idx = start_idx + num_generations
        group_counts = cumulative_guess_count[start_idx:end_idx]
        print(f"Prompt {prompt_idx} (rollouts {start_idx}-{end_idx-1}): {group_counts.tolist()}, 平均={group_counts.float().mean():.2f}")
    
    print(f"\n总体平均guess次数: {cumulative_guess_count.float().mean():.2f}")
    print(f"预期每个prompt组内guess次数: {num_guess} × {num_steps - 1} = {num_guess * (num_steps - 1)} (总)")
    print(f"预期每个rollout平均guess次数: {num_guess * (num_steps - 1) / num_generations:.2f}")

if __name__ == "__main__":
    test_guess_logic()
