from datasets import load_dataset

# Load the LongBench v2 dataset (train split)
dataset = load_dataset("THUDM/LongBench-v2", split="train")

# Print the first sample to inspect the structure
print(dataset[0])
