import pandas as pd
from tokenizer import Tokenizer

DATA_PATH = r"C:\Users\Lenovo\Desktop\Projects\gpt\wikipedia-tr\data\train-00001.parquet"

df = pd.read_parquet(DATA_PATH, engine='pyarrow')

text = df["text"][0]

    
tokenizer = Tokenizer()

training_tokens = tokenizer.encode(text)

tokenizer.merge_top_pairs(training_tokens, 300)





