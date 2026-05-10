from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# # Step 1: Make a BPE tokenizer
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# # Step 2: Teach it from some sentences (like a tiny textbook)
# trainer = BpeTrainer(special_tokens=["[UNK]"])
# sentences = "unhappiness is real", "playing and learning", "tokenization is cool"
# tokenizer.train_from_iterator(sentences, trainer)

# # Step 3: Break a word into pieces!
# result = tokenizer.encode("went")
# print(result.tokens)
# Output → small letter chunks joined by common pairs

# from transformers import BertTokenizer

# # This is Google's pre-trained WordPiece tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Break a word into pieces — notice the ## signs!
# tokens = tokenizer.tokenize("unhappiness")
# print(tokens)
# # Output: ['un', '##happiness']  ← ## means "I am the middle part!"

# tokens2 = tokenizer.tokenize("tokenization is playing")
# print(tokens2)
# # Output: ['token', '##ization', 'is', 'playing']
# tokens3 = tokenizer.tokenize("Devpelors")
# print(tokens3)
# # Output: ['token', '##ization', 'is', 'playing']

import sentencepiece as spm

# Step 1: Train a tiny model on your sentences
with open("my_text.txt", "w") as f:
    f.write("unhappiness is real\n"
        "playing and learning is fun\n"
        "tokenization is cool\n"
        "natural language processing helps computers understand text\n"
        "machine learning models are trained on large datasets\n"
        "deep learning uses neural networks for feature extraction\n"
        "python is a popular programming language for data science\n"
        "transformers use attention mechanisms to process sequences\n")

spm.SentencePieceTrainer.train(
    input="my_text.txt",
    model_prefix="my_model",  # saves my_model.model
    vocab_size=50
)

# Step 2: Load and use it
sp = spm.SentencePieceProcessor()
sp.load("my_model.model")

tokens = sp.encode("unhappiness", out_type=str)
print(tokens)
# Output: ['▁un', 'happi', 'ness']  ← ▁ means "start of a word"