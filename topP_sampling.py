'''
Nucleus Sampling（核采样），也被称为Top-p Sampling旨在在保持生成文本质量的同时增加多样性。
这种方法可以视作是Top-K Sampling的一种变体，它在每个时间步根据模型输出的概率分布选择概率累积超过给定阈值p的词语集合，然后在这个词语集合中进行随机采样。
这种方法会动态调整候选词语的数量，以保持一定的文本多样性。
在Nucleus Sampling中，模型在每个时间步生成词语时，首先按照概率从高到低对词汇表中的所有词语进行排序，然后模型计算累积概率，并找到累积概率超过给定阈值p的最小词语子集，
这个子集就是所谓的“核”（nucleus）。模型在这个核中进行随机采样，根据词语的概率分布来选择最终输出的词语。这样做可以保证所选词语的总概率超过了阈值p，同时也保持了一定的多样性。
参数p是Nucleus Sampling中的重要参数，它决定了所选词语的概率总和。p的值会被设置在(0,1]之间，表示词语总概率的一个下界。
Nucleus Sampling 能够保持一定的生成质量，因为它在一定程度上考虑了概率分布。通过选择概率总和超过给定阈值p的词语子集进行随机采样，Nucleus Sampling 能够增加生成文本的多样性。
'''

def top_p_sampling(input_ids, max_tokens=100, top_p=0.95):
 with torch.inference_mode():
    for _ in range(max_tokens):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        sorted_probabilities = F.softmax(sorted_logits, dim=-1) 
        cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False 
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits.scatter_(-1, indices_to_remove[None, :], float('-inf'))
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
 generated_text = tokenizer.decode(input_ids[0])
 return generated_text