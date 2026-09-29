import numpy as np

def softmax(x: np.ndarray)-> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        self.mask = inputs > 0
        return np.maximum(0, inputs)
    
    def backward(self, prev_grads: np.ndarray)-> np.ndarray:
        return prev_grads * self.mask
