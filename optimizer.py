import numpy as np


class AdamW:
    def __int__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps= 1e-8, wd=0.01)->None:
        self.parameters = parameters

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        
        self.t = 0
        
        self.m = []
        self.v = []
        
        for param in self.parameters:
            self.m.append(np.zeros_like(param["value"]))
            self.v.append(np.zeros_like(param["value"]))
            
    def step(self)-> None:
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            value = param["value"]
            grad = param["grad"]
            
            self.m[i] = (self.beta1 * self.m[i] + (1 - self.beta1) * grad)
            
            self.v[i] = (self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2))
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            adam_update = m_hat / (np.sqrt(v_hat) + self.eps)
            
            if param["wd"]:
                value -= self.lr * self.wd * value
                
            value -= self.lr * adam_update
            
    def zero_grad(self)-> None:
        for param in self.parameters:
            param["grad"].fill(0)