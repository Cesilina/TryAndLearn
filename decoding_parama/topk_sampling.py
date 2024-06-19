'''
Top-K 采样（在每个时间步选择条件概率排名前 K 的词语，然后在这 K 个词语中进行随机采样。
这种方法既能保持一定的生成质量，又能增加文本的多样性，并且可以通过限制候选词语的数量来控制生成文本的多样性。
这个过程使得生成的文本在保持一定的生成质量的同时，也具有一定的多样性，因为在候选词语中仍然存在一定的竞争性。
'''
def top_k_sampling(input_ids, max_tokens=100, top_k=50, temperature=1.0):

    for _ in range(max_tokens):
        with torch.inference_mode():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            top_k_probs = F.softmax(top_k_logits / temperature, dim=-1)
            next_token_index = torch.multinomial(top_k_probs, num_samples=1) # 多分布采样
            next_token = top_k_indices.gather(-1, next_token_index)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    generated_text = tokenizer.decode(input_ids[0])
    return generated_text