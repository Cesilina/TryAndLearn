'''
EMA 【指数滑动平均】
前t个时刻的数据的加权平均， 时间越近，权重越大，而且是指数式的
梯度下降过程中,用EMA融合历史的梯度，无论收敛速度和稳定性都有显著性收益
'''
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone() # 更新影子权重

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name] # 评估阶段，此处用影子权重代替进行评估

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
ema = EMA(model, 0.999)
ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
def train():
    optimizer.step() # 更新了模型的权重了
    ema.update() #更新影子权重

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
# 此处的验证应该是训练过程中的eval，restore之后，再进行下一个step的训练
def evaluate():
    ema.apply_shadow()
    # evaluate
    ema.restore()