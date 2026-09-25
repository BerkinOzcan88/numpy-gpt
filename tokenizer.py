class Tokenizer:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

    def encode(self, text: str)-> list[int]:
        token_ids = list(text.encode("utf-8"))
        
        for pair, new_token_id in self.merges.items():
            token_ids = self.merge(token_ids, pair, new_token_id)
        
        return token_ids

    def find_pairs(self, token_ids: list[int])-> dict:
        pair_counts = {}
        for pair in zip(token_ids, token_ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def merge(self, token_ids: list[int], pair: tuple[int, int], new_token_id: int)-> list[int]:
        new_tokens = []
        
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and pair[0] == token_ids[i] and pair[1] == token_ids[i+1]:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(token_ids[i])
                i += 1
        return new_tokens        

    def merge_top_pairs(self, token_ids: list[int], vocab_size: int)-> list[int]:
        new_token_ids = token_ids.copy()
        num_merges = vocab_size - 256
        
        for i in range(num_merges):
            pairs = self.find_pairs(new_token_ids)
            
            top_pair = max(pairs, key=pairs.get)
            new_token_id = 256 + i
            
            new_token_ids = self.merge(new_token_ids, top_pair, new_token_id)
        
            self.merges[top_pair] = new_token_id
            self.update_vocab(top_pair, new_token_id) 
        return new_token_ids
    
    def update_vocab(self, pair: tuple[int, int], new_token_id: int)-> None:
        self.vocab[new_token_id] =  self.vocab[pair[0]] + self.vocab[pair[1]]
    
    def decode(self, token_ids: list[int])-> str:
        tokens = b"".join(self.vocab[idx] for idx in token_ids)
        text = tokens.decode("utf-8", errors="replace")
        return  text

        