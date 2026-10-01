import numpy as np
from activation import softmax

class CrossEntropyLoss:
    def __init__(self):
        self.targets = None
        self.probs = None
    
    def forward(self, logits: np.ndarray, targets: np.ndarray)-> np.ndarray:
        self.targets = targets
        probs = softmax(logits)
        self.probs = probs
        
        B, T = targets.shape
        
        correct_probs = probs[np.arange(B)[:,None], np.arange(T)[None,:], targets]
        
        loss = - np.mean(np.log(correct_probs + 1e-12))
        
        return loss
        
    def backward(self)-> np.ndarray:   
        B, T = self.targets.shape
        grads = self.probs.copy()
        
        grads[np.arange(B)[:,None], np.arange(T)[None,:], self.targets] -=1
        
        return grads