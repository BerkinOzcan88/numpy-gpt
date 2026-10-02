import numpy as np
from activation import ReLU
from linear import LinearLayer

class FeedForward:
    def __init__(self, C: int)-> None:
        self.in_layer = LinearLayer(C, 4 * C)
        self.RelU = ReLU()
        self.out_layer = LinearLayer(4 * C, C)
        
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        return self.out_layer.forward(self.RelU.forward(self.in_layer.forward(inputs)))
    
    def backward(self, prev_grads: np.ndarray)-> np.ndarray:
        return self.in_layer.backward(self.RelU.backward(self.out_layer.backward(prev_grads)))
    
    def parameters(self) -> list[dict]:
        params = []

        params += self.in_layer.parameters()
        params += self.out_layer.parameters()

        return params