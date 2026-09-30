import numpy as np

class LayerNorm:
    def __init__(self, eps: float, C: int)-> None:
        self.eps = eps
        self.C = C
        self.gamma = np.ones(C)
        self.beta = np.zeros(C)
        
        self.gamma_grads = None
        self.beta_grads = None
        
        self.inputs = None
        self.mean = None
        self.var = None
        self.norm_inputs = None
        self.s = None
        self.a = None
        
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        self.inputs = inputs
        
        mean = np.mean(inputs, axis=-1, keepdims=True)
        self.mean = mean
        var = np.var(inputs, axis=-1,keepdims=True)
        self.var = var
        
        s = np.sqrt(var + self.eps)
        self.s = s
        a = (inputs - mean)
        self.a = a
        norm_inputs = a / s
        self.norm_inputs = norm_inputs
        
        out = self.gamma * norm_inputs + self.beta
        return out
        
        
        
    def backward(self, prev_grads: np.ndarray)-> np.ndarray:
        
        self.beta_grads = np.sum(prev_grads, axis=(0,1))
        self.gamma_grads = np.sum(prev_grads * self.norm_inputs, axis=(0,1))
        
        d_norm_inputs = prev_grads * self.gamma
        
        d_a = d_norm_inputs / self.s
        
        d_s = np.sum(d_norm_inputs * (- self.a / self.s ** 2), axis=-1, keepdims=True)
        
        d_var = d_s * (1 / (2 * self.s)) 
        
        d_a_var = d_var * (2 * self.a / self.C)
        
        d_a_total = d_a + d_a_var
        
        d_mean = -np.sum(d_a_total, axis=-1, keepdims=True)
        
        d_input_mean = d_mean / self.C
        
        d_input = d_a_total + d_input_mean
        
        return d_input
        
        
        
        
        
        
        
        