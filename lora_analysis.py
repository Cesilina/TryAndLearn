'''lora显存占用分析'''
import torch
print(torch.cuda.device_count())

from torch import nn
from peft import LoraConfig, get_peft_model, get_peft_model
import time

x_train = torch.randn((100,10))
y_train = torch.randn((100,1))

net = nn.Sequential(
    nn.Linear(10, 20),
    nn.Sigmoid(),
    nn.Linear(20, 30),
    nn.Sigmoid(),
    nn.Linear(30, 1)
)


print(net)

config = LoraConfig(target_modules=["0"], r=2)
model = get_peft_model(net, config)                       # 加了lora的网络
criterion = torch.nn.MSELoss(reduction='mean')            # 定义损失函数，采用均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.3)  # 定义优化器，采用Adam


# base 前向计算时间
time1 = time.time()
for i in range(100000):
    y_pre = net(x_train)            # 前向传播
print("base 前向计算时间: ", time.time() - time1)

# lora 前向计算时间
time1 = time.time()
for i in range(100000):  
    y_pre = model(x_train)            # 前向传播
print("lora 前向计算时间", time.time() - time1)

# base 反向传播时间
time1 = time.time()
for i in range(100000):  
    y_pre = net(x_train)            # 前向传播
    loss = criterion(y_pre, y_train)  # 计算损失
    optimizer.zero_grad()             # 梯度清零
    loss.backward()                   # 反向传播
    optimizer.step()                  # 使用优化器更新梯度
print("base 反向计算时间", time.time() - time1)

# lora 反向传播时间
time1 = time.time()
for i in range(100000):
    y_pre = model(x_train)            # 前向传播
    loss = criterion(y_pre, y_train)  # 计算损失
    optimizer.zero_grad()             # 梯度清零
    loss.backward()                   # 反向传播
    optimizer.step()                  # 使用优化器更新梯度
print("lora 反向计算时间", time.time() - time1)