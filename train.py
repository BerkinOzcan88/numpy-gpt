import  pandas as pd
import numpy as np
from tokenizer import Tokenizer
from model import GPT
from data import get_batch
from loss import CrossEntropyLoss
from optimizer import AdamW

import matplotlib.pyplot as plt

NUM_LAYERS = 4
VOCAB_SIZE = 700
C = 128
CONTEXT_LEN = 32
BATCH_SIZE = 8

DATA_PATH = r"C:\Users\Lenovo\Desktop\Projects\gpt\wikipedia-tr\data\train-00001.parquet"


df = pd.read_parquet(DATA_PATH, engine='pyarrow')

text = ". ".join(df["text"][:10].astype(str))

tokenizer = Tokenizer()

tokens = tokenizer.merge_top_pairs(text, VOCAB_SIZE)

tokens = np.array(tokens, dtype=np.int64)

n = int(0.9 * len(tokens))

training_data = tokens[:n]
val_data = tokens[n:]

loss_func = CrossEntropyLoss()
model = GPT(num_layers=NUM_LAYERS, vocab_size=VOCAB_SIZE, C=C, context_len=CONTEXT_LEN)

params = model.parameters()

optimizer = AdamW(parameters=params)


losses = []
val_losses = []
generated_list = []
start = np.array(tokenizer.encode("Marx"))[None,:]

steps = 300
for step in range(steps):
    x, y = get_batch(training_data, context_len=CONTEXT_LEN, batch_size=BATCH_SIZE)
    
    logits = model.forward(x)

    loss = loss_func.forward(logits=logits, targets=y)

    losses.append(loss)

    model.backward(loss_func.backward())

    optimizer.step()

    optimizer.zero_grad()
    
    if step % 10 == 0:
        print(f"{step} train_loss, {loss}")
        
    if step % 50 == 0:
        x_val, y_val = get_batch(val_data, context_len=CONTEXT_LEN, batch_size=BATCH_SIZE)

        val_logits = model.forward(x_val)

        val_loss = loss_func.forward(logits=val_logits, targets=y_val)
        
        val_losses.append(val_loss)
        
        print(f"{step} val_loss: {val_loss}")
    
    if step % 100 == 0:
        generated = model.generate(idx=start, max_tokens=100)
        
        generated_ids = generated[0].tolist()
        
        generated_text = tokenizer.decode(generated_ids)
        
        print(generated_text)
        generated_list.append(generated_text)

for text in generated_list:
    print(text)       

plt.plot(losses, label="loss")
plt.plot(val_losses, label="val loss")

plt.xlabel("steps")
plt.ylabel("loss")
plt.legend()
plt.show()