import numpy as np
from activation import softmax

class CausalSelfAttentionLayer:
    def __init__(self, C: int)-> None:
        self.W_q: np.ndarray  = np.random.randn(C,C) * 0.02
        self.W_k: np.ndarray  = np.random.randn(C,C) * 0.02
        self.W_v: np.ndarray  = np.random.randn(C,C) * 0.02
        self.W_o: np.ndarray  = np.random.randn(C,C) * 0.02
        
        
        self.W_q_grads: np.ndarray  = np.zeros_like(self.W_q)
        self.W_k_grads: np.ndarray  = np.zeros_like(self.W_k)
        self.W_v_grads: np.ndarray  = np.zeros_like(self.W_v)
        self.W_o_grads: np.ndarray  = np.zeros_like(self.W_o)
        
        self.inputs: np.ndarray = None
        self.Q: np.ndarray  = None
        self.K: np.ndarray  = None
        self.V: np.ndarray  = None
        self.attention_weights: np.ndarray  = None
        self.attention_output: np.ndarray  = None
    
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        self.inputs = inputs
        
        B, T, C = inputs.shape
        
        Q = inputs @ self.W_q
        K = inputs @ self.W_k
        V = inputs @ self.W_v
        
        self.Q, self.K, self.V = (Q, K, V)
        
        scores = (Q @ K.transpose(0,2,1)) / np.sqrt(C)
        
        mask = np.tril(np.ones(T,T), dtype=bool)
        
        scores = np.where(mask, scores, -np.inf)  
        
        attention_weights = softmax(scores)
        self.attention_weights = attention_weights
        
        attention_output = attention_weights @ V
        self.attention_output = attention_output
        
        return attention_output @ self.W_o
    
    
    def backward(self, prev_grads: np.ndarray)-> np.ndarray:
        B, T, C = self.inputs.shape
        
        
        d_attention_output = prev_grads @ self.W_o.T
        self.W_o_grads = np.reshape(self.attention_output, (B*T,C)).T @ np.reshape(prev_grads, (B*T,C))
        
        d_attention_weights = d_attention_output @ self.V.transpose(0,2,1)
        d_V = self.attention_weights.transpose(0,2,1) @ d_attention_output
        
        d_scores = self.attention_weights * (d_attention_weights - np.sum(d_attention_weights * self.attention_weights, axis=-1, keepdims=True))

        
        d_Q = (d_scores / np.sqrt(C)) @ self.K
        d_K = (d_scores / np.sqrt(C)).transpose(0,2,1) @ self.Q
        
        self.W_q_grads = np.reshape(self.inputs,(B*T, C)).T @ np.reshape(d_Q, (B*T, C))
        self.W_k_grads = np.reshape(self.inputs,(B*T, C)).T @ np.reshape(d_K, (B*T, C))
        self.W_v_grads = np.reshape(self.inputs,(B*T, C)).T @ np.reshape(d_V, (B*T, C))
        
        d_input_Q = d_Q @ self.W_q.T
        d_input_K = d_K @ self.W_k.T
        d_input_V = d_V @ self.W_v.T
        
        d_input = d_input_Q + d_input_K + d_input_V
        
        return d_input