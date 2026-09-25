import numpy as np
from activation import softmax

class CausalSelfAttentionLayer:
    def __init__(self, embedding_dim: int)-> None:
        self.W_q = np.random.randn(embedding_dim,embedding_dim) * 0.02
        self.W_k = np.random.randn(embedding_dim,embedding_dim) * 0.02
        self.W_v = np.random.randn(embedding_dim,embedding_dim) * 0.02
        self.W_o = np.random.randn(embedding_dim,embedding_dim) * 0.02
        
        
        self.W_q_grads = np.zeros_like(self.W_q)
        self.W_k_grads = np.zeros_like(self.W_k)
        self.W_v_grads = np.zeros_like(self.W_v)
        self.W_o_grads = np.zeros_like(self.W_o)
    
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        B, T, C = inputs.shape
        
        Q = inputs @ self.W_q
        K = inputs @ self.W_k
        V = inputs @ self.W_v
        
        scores = (Q @ K.transpose(0,2,1)) / np.sqrt(Q.shape[-1])
        
        mask = np.tril(np.ones(T,T))
        
        scores = np.where(mask, scores, -np.inf)  
        
        attention_weights = softmax(scores)
        
        attention_output = attention_weights @ V
        
        return attention_output @ self.W_o
        
          
        