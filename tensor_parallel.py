import time
import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo")

if __name__ == "__main__":
    rank = dist.get_rank()
    w_values = [
        [0.2, 0.9, 0.7, 0.7],
        [0.5, 0.2, 0.6, 0.0],
        [0.1, 0.2, 0.8, 0.8],
        [0.3, 0.5, 0.8, 1.0],
        [0.7, 0.6, 0.8, 0.2],
        [0.6, 0.1, 0.2, 0.2],
        [0.7, 0.7, 0.2, 0.7],
        [0.8, 0.4, 0.5, 0.6],
        [0.8, 1.0, 0.1, 0.3],
        [0.7, 0.9, 0.9, 0.9],
    ]
    w = torch.tensor(w_values, requires_grad=True)
    if rank == 0:
        x = torch.rand(4, 6)  # 只有 Rank 0 初始化 X，然后广播给其他的 。
    else:
        x = torch.zeros(4, 6)  # 创建一个 0 矩阵用于接收 Rank 0 的广播结果。

    # Rank 0 将 X 广播给其他的 Rank
    dist.broadcast_object_list(x, src=0)
    w_tensor = torch.tensor(w_values, requires_grad=True)
    target_y = torch.matmul(w_tensor, x)
    s = torch.sum(target_y)  # 前向矩阵乘法
    s.backward()  # 反向梯度
    print(target_y, w_tensor.grad)

    