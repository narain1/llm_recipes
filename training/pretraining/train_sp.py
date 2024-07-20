import sentencepiece as spm
import os

def gather_text_files(root_dir):
    """Recursively collects text file paths from the specified directory."""
    text_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'): 
                text_files.append(os.path.join(root, file))
    return text_files

def combine_text_files(text_files, combined_file_path):
    """Combines text from multiple files into a single file."""
    with open(combined_file_path, 'w', encoding='utf-8') as outfile:
        for fname in text_files:
            with open(fname, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

def train_sentencepiece(combined_file_path, model_prefix, vocab_size=32000):
    """Trains the SentencePiece model from the combined text file."""
    spm.SentencePieceTrainer.Train(
            input=combined_file_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            num_threads=os.cpu_count(),
            byte_fallback=True,
            allow_whitespace_only_pieces=True,
            normalization_rule_name="identity",)

def load_sentencepiece_model(model_path):
    """Loads the SentencePiece model from the specified path."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def tokenize_text(sp, text):
    """Tokenizes the given text using the loaded SentencePiece model."""
    tokens = sp.EncodeAsPieces(text)
    return tokens


os.makedirs("out", exist_ok=True)
root_dir = 'data/raw' 
combined_file_path = 'data/combined_tamil_corpus.txt'
model_prefix = 'out/tamil_spm'
vocab_size = 32_000

text_files = gather_text_files(root_dir)
combine_text_files(text_files, combined_file_path)
train_sentencepiece(combined_file_path, model_prefix, vocab_size=vocab_size)


tamil_sentence = 'அவள் புத்தகத்தை வாசிக்கிறாள்'
sp = load_sentencepiece_model(model_prefix + ".model")
tokens = sp.EncodeAsPieces(tamil_sentence)

print("Original Sentence:")
print(tamil_sentence)
print("\nTokenized Output:")
print(tokens)
