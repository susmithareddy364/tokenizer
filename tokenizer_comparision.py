# tokenizer_comparison.py

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer

# Sentence example
sentence = "Hello world!"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenize sentence
token_ids = tokenizer(sentence).input_ids

print("\nToken IDs:\n")
print(token_ids)

print("\nDecoded Tokens:\n")
for id in token_ids:
    print(tokenizer.decode(id))


# Colors for token visualization
colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]


# Function to visualize tokens
def show_tokens(sentence: str, tokenizer_name: str):
    """Show tokens with different colors"""

    print(f"\n----- {tokenizer_name} -----")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids

    print(f"Vocabulary Length: {len(tokenizer)}")

    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m'
            + tokenizer.decode(t)
            + '\x1b[0m',
            end=' '
        )

    print("\n")


# Text to test tokenization
text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""


# Run tokenizers
show_tokens(text, "bert-base-cased")

# Optional comparisons
show_tokens(text, "bert-base-uncased")
show_tokens(text, "Xenova/gpt-4")
show_tokens(text, "gpt2")
show_tokens(text, "google/flan-t5-small")
show_tokens(text, "bigcode/starcoder2-15b")
show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")
show_tokens(text, "Qwen/Qwen2-VL-7B-Instruct")
