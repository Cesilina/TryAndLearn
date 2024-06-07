'''
温度参数采样常用于基于概率的生成模型。通过引入一个温度的参数来调整模型输出的概率分布，从而控制生成文本的多样性
在温度参数采样中,模型在每个时间步生成词语时,会计算出词语的条件概率分布,然后模型将这个条件概率分布中的每个词语的概率值除以温度参数,对结果进行归一化处理,
获得新的归一化概率分布。较高的温度值会使概率分布更平滑,从而增加生成文本的多样性。低概率的词语也有较高的可能性被选择；
而较低的温度值则会使概率分布更集中，更倾向于选择高概率的词语，因此生成的文本更加确定性。最后模型根据这个新的归一化概率分布进行随机采样，选择生成的词语。
'''
import torch
import torch.nn.functional as F

def temperature_sampling(logits, temperature=1.0):
    logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)
    sampled_token = torch.multinomial(probabilities, 1)
return sampled_token.item()