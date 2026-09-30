import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size: int, C: int)-> None:
        self.embeddings = np.random.randn(vocab_size, C) * 0.02
        self.inputs = None
        
        self.embedding_grads = np.zeros_like(self.embeddings)
        
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        self.inputs = inputs
        return self.embeddings[inputs]
    
    def backward(self, prev_grads: np.ndarray)-> None:
        embedding_grads = np.zeros_like(self.embeddings)
        
        np.add.at(embedding_grads, self.inputs, prev_grads)
        
        self.embedding_grads += embedding_grads
        

class PositionalEmbedding:
    def __init__(self, context_len: int, C: int)-> None:
        self.pos_embeddings = np.random.randn(context_len, C) * 0.02
        
        self.pos_embedding_grads = np.zeros_like(self.pos_embeddings)
        
    def forward(self, inputs: np.ndarray)-> np.ndarray:
        sequence_length = inputs.shape[-1]
        return self.pos_embeddings[:sequence_length]
    
    def backward(self, prev_grads: np.ndarray)-> None:
        sequence_length = prev_grads.shape[1]
        
        pos_grads = np.sum(prev_grads, axis=0)
        self.pos_embedding_grads[:sequence_length] += pos_grads
        
    

