import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------
# config
# -----------------------
dataset_name = "HuggingFaceH4/MATH-500"
out_dir = os.path.dirname(__file__)

enc = tiktoken.get_encoding("gpt2")

# -----------------------
# load dataset
# -----------------------
print(f"Loading dataset: {dataset_name}")
dataset = load_dataset(dataset_name)

# create train/val split if needed
if "train" in dataset and "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    dataset["val"] = dataset.pop("test")

# -----------------------
# clean text format (NO labels)
# -----------------------
def extract_text(example):
    problem = example["problem"]
    solution = example["solution"]

    # simple concatenation only
    text = problem.strip() + "\n" + solution.strip()

    return text

# -----------------------
# tokenize
# -----------------------
def encode(example):
    text = extract_text(example)
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    return ids

# -----------------------
# build dataset
# -----------------------
for split_name, split_data in dataset.items():
    print(f"\nProcessing {split_name}...")

    all_tokens = []

    for item in tqdm(split_data):
        all_tokens.extend(encode(item))

    arr = np.array(all_tokens, dtype=np.uint16)

    out_path = os.path.join(out_dir, f"{split_name}.bin")
    print(f"Writing {out_path} ({len(arr)} tokens)")

    with open(out_path, "wb") as f:
        arr.tofile(f)

print("\nDone: train.bin and val.bin created")