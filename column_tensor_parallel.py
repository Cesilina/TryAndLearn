import torch

def check_tensor_value_equal(x, y):
    x = torch.round(x, decimals=4)
    y = torch.round(y, decimals=4)
    diff = torch.sum(x - y)
    return diff == 0


def test_column_split(w, x, target_y, target_w_grad):
    rank = dist.get_rank()

    if rank == 0:
        sub_x = torch.split(x, [3, 3], dim=1)[0] # 获取 X[:,0:3]
    elif rank == 1:
        sub_x = torch.split(x, [3, 3], dim=1)[1] # 获取 X[:,3:6]

    sub_y = torch.matmul(w, sub_x)  # 子矩阵乘法
    tensor_list = [torch.zeros((10, 3)) for _ in range(2)]
    dist.all_gather(tensor_list, sub_y)  # allgather 获取给个 rank 的乘法结果
    output = torch.concat(tensor_list, dim=1) # 将各 rank 上的结果拼接到一起。
    matmul_equal = check_tensor_value_equal(output, target_y)  # 检查是否和原矩阵乘法结果一致。

    s = torch.sum(sub_y)
    s.backward()  # 求 X 的梯度
    w_grad = w.grad
    dist.all_reduce(w_grad)  # 通过 allreduce 对各个 rank 上的梯度求和。
    grad_equal = check_tensor_value_equal(w_grad, target_w_grad) # 检查反向梯度和原矩阵乘法结果一致。
    if rank == 1:
        time.sleep(0.1)  # 防止后面的 print 打印重叠

    print(f"Rank {rank}: Matmul equal {matmul_equal}")
    print(f"Rank {rank}: Grad equal {grad_equal}")



def test_row_split(w, x, target_y, target_w_grad):
    rank = dist.get_rank()
    if rank == 0:
        sub_w = torch.split(w, [2, 2], dim=1)[0]  # 获取 W[:,0:2]
        sub_w.retain_grad()  # 保留 W 子矩阵的梯度。
        sub_x = torch.split(x, [2, 2], dim=0)[0]   # 获取 X[0:2,:]
    elif rank == 1:
        sub_w = torch.split(w, [2, 2], dim=1)[1]  # 获取 W[:,2:4]
        sub_w.retain_grad()  # 保留 W 子矩阵的梯度。
        sub_x = torch.split(x, [2, 2], dim=0)[1]   # 获取 X[2:4,:]

    sub_y = torch.matmul(sub_w, sub_x)
    dist.all_reduce(sub_y)  # 通过 allreduce 对各个rank 的前向乘法结果汇总求和。
    matmul_equal = check_tensor_value_equal(sub_y, target_y) # 检查前向结果是否和原矩阵乘法结果一致。

    # backward
    s = torch.sum(sub_y)
    s.backward(retain_graph=True)
    tensor_list = [torch.zeros((10, 2)) for _ in range(2)]
    dist.all_gather(tensor_list, sub_w.grad)  # 通过 allgather 获取各个 rank 的梯度。
    w_grad = torch.concat(tensor_list, dim=1)
    grad_equal = check_tensor_value_equal(w_grad, target_w_grad) # 检查反向梯度和原矩阵乘法结果一致。

    if rank == 1:
        time.sleep(0.1)  # 防止后面的 print 打印重叠

    print(f"Rank {rank}: Matmul equal {matmul_equal}")
    print(f"Rank {rank}: Grad equal {grad_equal}")