'''
贪心搜索在每个时间步t都选取当前概率分布中概率最大的词,直到y达到预设最大长度时停止生成
'''
import torch

def greedy_decoding(input_ids, max_tokens=300):

 with torch.inference_mode():
    
    for _ in range(max_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        if next_token == tokenizer.eos_token_id:
            break
        input_ids = torch.cat([input_ids, rearrange(next_token, 'c -> 1 c')], dim=-1)
 generated_text = tokenizer.decode(input_ids[0])
 return generated_text