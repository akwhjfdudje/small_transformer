import re

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
        # Add special tokens
        if "<unk>" not in self.vocab:
            self.vocab["<unk>"] = len(self.vocab)
        if "<pad>" not in self.vocab:
            self.vocab["<pad>"] = len(self.vocab)

    def build_vocab(self, texts, min_freq=1):
        counts = {}
        for text in texts:
            for full_word in re.findall(r"\w+|[^\w\s]", text):
                for word in [full_word[0:len(full_word)//2], full_word[len(full_word)//2:]]:
                    counts[word] = counts.get(word, 0) + 1
        for word, freq in counts.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        full_words = re.findall(r"\w+|[^\w\s]", text)
        words = []
        for full_word in full_words:
            words += [full_word[0:len(full_word)//2], full_word[len(full_word)//2:]]
        return [self.vocab.get(w, self.vocab["<unk>"]) for w in words]

    def decode(self, ids):
        return " ".join([self.inv_vocab.get(i, "<unk>") for i in ids])
