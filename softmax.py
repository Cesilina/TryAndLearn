
class SoftMaxLayer(BaseLayer):    
    def __init__(self, name):        
        super(SoftMaxLayer,self).__init__(name) 

        def forward(self,input):        
            vec_max = np.max( input,axis=1 )[np.newaxis,:].T        
            input -= vec_max        
            exp = np.exp(input)        
            output = exp / (np.sum(exp,axis=1)[np.newaxis,:].T)        
            return output


class SMCrossEntropyLossLayer(BaseLayer):    
    def __init__(self, name):        
        super(SMCrossEntropyLossLayer,self).__init__(name)    

        def forward(self,pred,real):        
            self.softmax_p = SoftMaxLayer("softmax").forward(pred)        
            self.real = real        
            loss = 0        
            for i in range(self.real.shape[0]):            
                loss += -np.log( self.softmax_p[i,real[i]] )        
                loss /= self.real.shape[0]        
            return loss    
            
        def backward(self):        
            for i in range(self.real.shape[0]):           
                self.softmax_p[i,self.real[i]] -= 1        
                self.softmax_p = self.softmax_p / self.real.shape[0]        
            return self.softmax_p


'''基于numpy实现softmax的代码'''
import numpy as np

def softmax(x):
    """
    计算输入x的Softmax。
    参数x可以是x的向量或矩阵。
    返回与x形状相同的Softmax向量或矩阵。
    """
    # 防止溢出，通过减去x的最大值来进行数值稳定性处理
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
