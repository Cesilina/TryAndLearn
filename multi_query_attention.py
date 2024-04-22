'''
1.在计算多头注意力的时候，query仍然进行分头，和多头注意力机制相同，而key和value只有一个头
2.在多查询注意力（MQA）中，
query的维度为[batch_size, num_heads, seq_len, head_dim]，
key和value的维度为[batch_size, 1, seq_len, head_dim]。
这样就无法直接进行矩阵的乘法，为了完成这一乘法，可以采用torch的广播乘法
'''

import torch
from torch import nn


class MutiQueryAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MutiQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        ## 初始化Q、K、V投影矩阵
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim) ###此时k和v只有一组
        self.v_linear = nn.Linear(hidden_size, self.head_dim) ###
        
        ## 输出线性层
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)
        
        query = self.split_head(query)
        key = self.split_head(key, 1)
        value = self.split_head(value, 1)
        
        ## 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        
        ## 对注意力分数进行归一化
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, value)
        
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads) # [batch_size, seq_len, hidden_size]
        
        output = self.o_linear(output)
        
        return output
        
        
        
        
    def split_head(self, x, head_num=None):
        
        batch_size = x.size()[0]
        
        if head_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1,2)
        

if __name__ == '__main__':
    batch_size = 10
    num_heads = 8
    hidden_size = 64
    seq_len = 16

    mha = MutiQueryAttention(hidden_size, num_heads)

    hidden_state = torch.randn((batch_size, seq_len, hidden_size))
    output = mha(hidden_state)
    print(output.shape)