import pandas as pd

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self,s: str)-> list[int]:
        tokens = s.encode("utf-8")
        tokens = list(map(int, tokens))
        return tokens

    def find_pairs(self, tokens: list[int])-> dict:
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, tokens: list[int], pair: set[int], new_token: int)-> list[int]:
        new_tokens = []
        
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and pair[0] == tokens[i] and pair[1] == tokens[i+1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens        

    def merge_top_pairs(self, tokens: list[int], vocab_size: int)-> list[int]:
        new_tokens = tokens.copy()
        num_merges = vocab_size - 256
        for i in range(num_merges):
            pairs = self.find_pairs(new_tokens)
            top_pair = max(pairs, key=pairs.get)
            new_tokens = self.merge(new_tokens, top_pair, 256+i)
            
        return new_tokens

        