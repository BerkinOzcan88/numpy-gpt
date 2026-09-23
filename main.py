import pandas as pd
from tokenizer import Tokenizer

DATA_PATH = r"C:\Users\Lenovo\Desktop\Projects\gpt\wikipedia-tr\data\train-00001.parquet"

df = pd.read_parquet(DATA_PATH, engine='pyarrow')

text = df["text"][0]

    
tokenizer = Tokenizer()

tokens = tokenizer.tokenize(text)

print(len(tokens))

tokens = tokenizer.merge_top_pairs(tokens, 400)

print(len(tokens))

