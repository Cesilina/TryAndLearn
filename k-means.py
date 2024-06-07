import numpy as np

def kmeans(data, k, thresh=1, max_iterations=100):

  # 随机初始化k个中心点
  centers = data[np.random.choice(data.shape[0], k, replace=False)]

  for _ in range(max_iterations):
    # 计算每个样本到各个中心点的距离
    distances = np.linalg.norm(data[:, None] - centers, axis=2) # 形状为[num_samples, num_centers]

    # 根据距离最近的中心点将样本分配到对应的簇
    labels = np.argmin(distances, axis=1)
    print(labels)

    # 更新中心点为每个簇的平均值
    new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])

    # 判断中心点是否收敛，多种收敛条件可选
    # 条件1：中心点不再改变
    if np.all(centers == new_centers):
      break
    # 条件2：中心点的阈值小于某个阈值
    # center_change = np.linalg.norm(new_centers - centers)
    # if center_change < thresh:
    #     break
    centers = new_centers

  return labels, centers

# 生成一些随机数据作为示例输入
data = np.random.rand(100, 2)  # 100个样本，每个样本有两个特征

# 手动实现K均值算法
k = 3  # 聚类数为3
labels, centers = kmeans(data, k)

# 打印簇标签和聚类中心点
print("簇标签:", labels)
print("聚类中心点:", centers)