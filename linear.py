import numpy as np

class LinearLayer:
    def __init__(self, C_in, C_out, init_std=None)-> None:
        std = np.sqrt(2 / C_in) if init_std is None else init_std
        self.W = np.random.randn(C_in, C_out) * std
        self.b = np.zeros(C_out)
        
        self.W_grads = np.zeros_like(self.W)
        self.b_grads = np.zeros_like(self.b)
        
        self.inputs = None
        
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        self.inputs = inputs
        return inputs @ self.W + self.b
    
    def backward(self, prev_grads: np.ndarray)-> np.ndarray:
        
        self.b_grads[:] = np.sum(prev_grads, axis=(0,1))
        
        self.W_grads[:] = np.sum(self.inputs.transpose(0,2,1) @ prev_grads, axis=0)
        d_input = prev_grads @ self.W.T
        
        return d_input
    
    def parameters(self)-> list[dict]:
        return [
        {
            "value": self.W,
            "grad": self.W_grads,
            "wd": True
        },
        
        {
            "value": self.b,
            "grad": self.b_grads,
            "wd": False
        }
    ]
