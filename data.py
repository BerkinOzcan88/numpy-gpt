import numpy as np



def get_batch(data, context_len: int, batch_size: int)-> tuple:
    max_start = len(data) - context_len - 1
    
    starts = np.random.randint(0, max_start, size=batch_size)
    
    x = np.stack([
        data[i:i + context_len]
        for i in starts
    ])

    y = np.stack([
        data[i + 1:i + context_len + 1]
        for i in starts
    ])

    return x, y



