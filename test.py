import numpy as np
import pandas as pd
from model import Model
from tokenizer import Tokenizer
from activation import softmax

NUM_LAYERS = 2
VOCAB_SIZE = 2048
C = 512
CONTEXT_LEN = 16
EPS = 1e-5

DATA_PATH = r"C:\Users\Lenovo\Desktop\Projects\gpt\wikipedia-tr\data\train-00001.parquet"

df = pd.read_parquet(DATA_PATH, engine='pyarrow')

text = ". ".join(df["text"][:5].astype(str))

model = Model(num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE, C=C, context_len=CONTEXT_LEN, eps=EPS)

tokenizer = Tokenizer()

training_tokens = tokenizer.encode(text)

tokenizer.merge_top_pairs(training_tokens, VOCAB_SIZE)


text = "Merhaba dünya!"
token_ids = np.array(tokenizer.encode(text))
token_ids = np.array([token_ids])
for _ in range(80):
    input_ids = token_ids[:, -CONTEXT_LEN:]
    
    logits = model.forward(input_ids)
    probs = softmax(logits[0,-1,:])
    next_token = np.random.choice(len(probs), p=probs)
    token_ids = np.append(token_ids, np.array([[next_token]]), axis=1)
    
    text = tokenizer.decode(token_ids[0].tolist())  
    
    print(text) 
