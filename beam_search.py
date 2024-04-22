import torch
import torch.nn.functional as F

def beam_search(model, init_input, beam_width, max_length):
    """
    :param model: 神经网络模型
    :param init_input: 初始输入，通常是包含开始符的编码
    :param beam_width: beam search的宽度
    :param max_length: 最大序列长度
    :return: 最终的输出序列和对应的分数
    """
    # 初始化输出序列列表和分数列表
    output_sequences = []
    output_scores = []

    # 初始化beam search的候选序列
    sequences = [(init_input, 0)]  # (sequence, score)

    # 遍历所有可能的输出长度
    for step in range(1, max_length + 1):
        # 对于每个候选序列，生成下一个可能的词
        new_sequences = []
        for sequence, score in sequences:
            output, prob = model(sequence)
            # 获取最可能的下一个词
            topv, topi = torch.topk(prob, k=beam_width)
            for new_seq, new_prob in zip(output, topi):
                # 计算新的序列和分数
                new_seq = torch.cat((sequence, new_seq.unsqueeze(0)), 0)
                new_score = score + F.log_softmax(new_prob, dim=1).squeeze()
                new_sequences.append((new_seq, new_score))

        # 选择beam width个最佳候选项
        sequences = sorted(new_sequences, key=lambda x: -x[1])[:beam_width]

        # 如果找到了结束符，就结束搜索
        if step > 1 and 1 in [seq[0][-1].item() for seq, _ in sequences]:
            break

    # 选择最佳序列
    best_sequence, best_score = max(sequences, key=lambda x: x[1])
    output_sequences.append(best_sequence[0].unsqueeze(0))
    output_scores.append(best_score.item())

    return output_sequences[0], output_scores[0]

# 假设的模型和输入
class FakeModel:
    def __call__(self, x):
        # 假设模型输出一个概率分布，其中包含一个结束符的概率
        return torch.randn(1, 10), torch.tensor([0.01])

# 假设的输入
init_input = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # 包含开始符的编码

# 执行beam search
final_sequence, final_score = beam_search(FakeModel(), init_input, beam_width=3, max_length=10)

print("Generated sequence:", final_sequence)
print("Score:", final_score)