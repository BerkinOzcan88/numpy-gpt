import numpy as np
from feedforward import FeedForward
from attention import CausalSelfAttentionLayer
from layernorm import LayerNorm

class TransformerBlock:
    def __init__(self, C: int)-> None:
        self.attention = CausalSelfAttentionLayer(C)
        self.attention_norm = LayerNorm(C)
        
        self.feedforward = FeedForward(C)
        self.feedforward_norm = LayerNorm(C)
        
    
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        norm_attn_in = self.attention_norm.forward(inputs)
        attn_out = self.attention.forward(norm_attn_in)
        
        x_attn = inputs + attn_out
        
        norm_ffn_in = self.feedforward_norm.forward(x_attn)
        ffn_out = self.feedforward.forward(norm_ffn_in)
        
        x_ffn = x_attn + ffn_out
        
        return x_ffn
    
    def backward(self, prev_grads: np.ndarray)-> np.ndarray:
        d_norm_ffn_in= self.feedforward.backward(prev_grads)
        d_ffn_residual = self.feedforward_norm.backward(d_norm_ffn_in)
        
        d_x_attn = prev_grads + d_ffn_residual
        
        d_norm_attn_in = self.attention.backward(d_x_attn)
        d_attn_residual = self.attention_norm.backward(d_norm_attn_in)
        
        d_input = d_x_attn + d_attn_residual
        
        return d_input
    
    def parameters(self)-> list[dict]:
        params = []
        
        params += self.attention.parameters()
        params += self.attention_norm.parameters()
        params += self.feedforward.parameters()
        params += self.feedforward_norm.parameters()
        
        return params
        