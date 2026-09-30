import numpy as np
from embedding import TokenEmbedding, PositionalEmbedding
from transformer import TransformerBlock
from layernorm import LayerNorm
from linear import LinearLayer

class Model:
    def __init__(self, num_layers: int, vocab_size: int, C: int, context_len: int, eps: float)-> None:
        self.token_embedding = TokenEmbedding(vocab_size, C)
        self.pos_embedding = PositionalEmbedding(context_len, C)
        self.transformer_blocks = [TransformerBlock(eps, C) for _ in range(num_layers)]
        self.final_norm = LayerNorm(eps, C)
        self.out_projection = LinearLayer(C, vocab_size)

    def forward(self, token_ids: np.ndarray)-> np.ndarray:
        token_embeddings = self.token_embedding.forward(token_ids)
        pos_embeddings = self.pos_embedding.forward(token_ids)
        
        embeddings = token_embeddings + pos_embeddings
        
        x = embeddings
        for block in self.transformer_blocks:
            x = block.forward(x)
            
        norm_x = self.final_norm.forward(x)
        
        logits = self.out_projection.forward(norm_x)
        
        return logits

        
        

