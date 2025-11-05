import os
import sentencepiece as spm

class SPTokenizer:
    """
    SentencePiece-based tokenizer with auto-train + auto-load.
    Works with any English text dataset.
    """

    def __init__(self, model_dir="tokenizer", vocab_size=16000, dataset_path=None):
        """
        Args:
            model_dir: directory to store spm.model and spm.vocab
            vocab_size: vocabulary size
            dataset_path: text dataset to train tokenizer if model missing
        """
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "spm.model")
        self.vocab_path = os.path.join(model_dir, "spm.vocab")
        self.vocab_size = vocab_size

        # Automatically train if not found
        if not os.path.exists(self.model_path):
            if dataset_path is None:
                raise ValueError(
                    f"No tokenizer found at {self.model_path}. Please provide dataset_path to train."
                )
            print(f"[Tokenizer] Training new SentencePiece model (vocab={vocab_size})...")
            spm.SentencePieceTrainer.train(
                input=dataset_path,
                model_prefix=os.path.join(model_dir, "spm"),
                vocab_size=vocab_size,
                character_coverage=1.0,
                model_type="bpe",  # can be "unigram", "bpe", or "word"
                bos_id=1,
                eos_id=2,
                pad_id=0,
                unk_id=3
            )
            print("[Tokenizer] Training complete. Model saved to:", self.model_path)
        else:
            print(f"[Tokenizer] Loaded existing model from {self.model_path}")

        # Load trained tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def encode(self, text, add_bos=True, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos: ids = [self.sp.bos_id()] + ids
        if add_eos: ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids):
        # Filter out padding or invalid tokens
        ids = [i for i in ids if i > 0]
        return self.sp.decode(ids)

    def vocab_size(self):
        return self.sp.vocab_size()

    def save_vocab(self, path=None):
        """Optional: save vocab to text file for inspection."""
        path = path or self.vocab_path
        with open(path, "w", encoding="utf-8") as f:
            for i in range(self.sp.vocab_size()):
                f.write(f"{self.sp.id_to_piece(i)}\n")
        print(f"[Tokenizer] Vocabulary saved to {path}")

