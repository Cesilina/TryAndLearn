'''
在每次更新时用1个样本，可以看到多了随机两个字，随机也就是说我们用样本中的一个例子来近似我所有的样本，来调整θ，因而随机梯度下降是会带来一定的问题，
因为计算得到的并不是准确的一个梯度。对于最优化问题，凸问题，虽然不是每次迭代得到的损失函数都向着全局最优方向，
 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。

numpy实现随机梯度下降法
'''


# 数据加载
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, Y = fetch_california_housing(return_X_y=True)
print(X.shape, Y.shape)  # (20640, 8), (20640, )

# 数据预处理
ones = np.ones(shape=(X.shape[0], 1))
X = np.hstack([X, ones])
validate_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validate_size, shuffle=True)

# batch 函数
def get_batch(batchsize: int, X: np.ndarray, Y: np.ndarray):
    assert 0 == X.shape[0]%batchsize, f'{X.shape[0]}%{batchsize} != 0'
    batchnum = X.shape[0]//batchsize
    X_new = X.reshape((batchnum, batchsize, X.shape[1]))
    Y_new = Y.reshape((batchnum, batchsize, ))

    for i in range(batchnum):
        yield X_new[i, :, :], Y_new[i, :]

# 损失函数
def mse(X: np.ndarray, Y: np.ndarray, W: np.ndarray):
    return 0.5 * np.mean(np.square(X@W-Y))

def diff_mse(X: np.ndarray, Y: np.ndarray, W: np.ndarray):
    return X.T@(X@W-Y) / X.shape[0]

# 模型训练
lr = 0.001          # 学习率
num_epochs = 1000   # 训练周期
batch_size = 64     # |每个batch包含的样本数
validate_every = 4  # 多少个周期进行一次检验
def train(num_epochs: int, batch_size: int, validate_every: int, W0: np.ndarray, X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray):
    loop = tqdm(range(num_epochs))
    loss_train = []
    loss_validate = []
    W = W0
    # 遍历epoch
    for epoch in loop:
        loss_train_epoch = 0
        # 遍历batch
        for x_batch, y_batch in get_batch(64, X_train, Y_train):
            loss_batch = mse(X=x_batch, Y=y_batch, W=W)
            loss_train_epoch += loss_batch*x_batch.shape[0]/X_train.shape[0]
            grad = diff_mse(X=x_batch, Y=y_batch, W=W)
            W = W - lr*grad

        loss_train.append(loss_train_epoch)
        loop.set_description(f'Epoch: {epoch}, loss: {loss_train_epoch}')

        if 0 == epoch%validate_every:
            loss_validate_epoch = mse(X=X_test, Y=Y_test, W=W)
            loss_validate.append(loss_validate_epoch)
            print('============Validate=============')
            print(f'Epoch: {epoch}, train loss: {loss_train_epoch}, val loss: {loss_validate_epoch}')
            print('================================')
    plot_loss(np.array(loss_train), np.array(loss_validate), validate_every)

# 程序运行
W0 = np.random.random(size=(X.shape[1], ))  # 初始权重
train(num_epochs=num_epochs, batch_size=batch_size, validate_every=validate_every, W0=W0, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)