import numpy as np
import pandas as pd
from tokenizer import Tokenizer

VOCAB_SIZE = 1000
CONTEXT_LEN = 32
BATCH_SIZE = 8

DATA_PATH = r"C:\Users\Lenovo\Desktop\Projects\gpt\wikipedia-tr\data\train-00001.parquet"

df = pd.read_parquet(DATA_PATH, engine='pyarrow')

text = ". ".join(df["text"][:5].astype(str))

    
tokenizer = Tokenizer()

tokens = tokenizer.merge_top_pairs(text, VOCAB_SIZE)

tokens = np.array(tokens, dtype=np.int64)

n = int(0.9 * len(tokens))

training_data = tokens[:n]
val_data = tokens[n:]

def get_batch(data):
    max_start = len(data) - CONTEXT_LEN - 1
    
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    
    x = np.stack([
        data[i:i + CONTEXT_LEN]
        for i in starts
    ])

    y = np.stack([
        data[i + 1:i + CONTEXT_LEN + 1]
        for i in starts
    ])

    return x, y



