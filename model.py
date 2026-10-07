import numpy as np
from embedding import TokenEmbedding, PositionalEmbedding
from transformer import TransformerBlock
from layernorm import LayerNorm
from linear import LinearLayer
from activation import softmax
class GPT:
    def __init__(self, num_layers: int, vocab_size: int, C: int, context_len: int)-> None:
        self.token_embedding = TokenEmbedding(vocab_size, C)
        self.pos_embedding = PositionalEmbedding(context_len, C)
        self.transformer_blocks = [TransformerBlock(C) for _ in range(num_layers)]
        self.final_norm = LayerNorm(C)
        self.lm_head = LinearLayer(C, vocab_size)
        
        self.context_len = context_len
        self.vocab_size = vocab_size

    def forward(self, token_ids: np.ndarray)-> np.ndarray:
        token_embeddings = self.token_embedding.forward(token_ids)
        pos_embeddings = self.pos_embedding.forward(token_ids)
        
        embeddings = token_embeddings + pos_embeddings
        
        x = embeddings
        for block in self.transformer_blocks:
            x = block.forward(x)
            
        norm_x = self.final_norm.forward(x)
        
        logits = self.lm_head.forward(norm_x)
        
        return logits

    def backward(self, prev_grads: np.ndarray)-> None:
        lm_head_grads = self.lm_head.backward(prev_grads)  
        norm_x_grads = self.final_norm.backward(lm_head_grads)
        
        x_grads = norm_x_grads
        for block in reversed(self.transformer_blocks):
            x_grads = block.backward(x_grads)
            
        self.token_embedding.backward(x_grads)
        self.pos_embedding.backward(x_grads)
    
    
    def parameters(self)-> list[dict]:
        params = []

        params += self.token_embedding.parameters()
        params += self.pos_embedding.parameters()

        for block in self.transformer_blocks:
            params += block.parameters()

        params += self.final_norm.parameters()
        params += self.lm_head.parameters()

        return params
    
    def generate(self, idx: np.ndarray, max_tokens: int, temp=1.0)-> np.ndarray:
        for _ in range(max_tokens):
            context_idx = idx[:, -self.context_len:]
            
            logits = self.forward(context_idx)
            
            logits = logits[:, -1, :]
        
            logits = logits / temp
            
            probs = softmax(logits)
            
            new_token = np.array([np.random.choice(self.vocab_size, p=p) for p in probs])[:, None]
            
            idx = np.concatenate([idx, new_token], axis=1)
            
        return idx

