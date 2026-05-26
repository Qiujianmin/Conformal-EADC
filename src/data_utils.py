"""
Data utilities for FineHarm streaming safety detection experiment.

Converts FineHarm dataset into streaming prefix format for probe training
and e-process evaluation.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data-FineHarm"


@dataclass
class StreamingSample:
    """A single prompt-response pair with streaming metadata."""
    idx: int
    prompt: str
    response: str
    tokens: List[str]           # flattened word-level tokens
    token_labels: List[int]     # 0=safe, 1=harmful at token level
    sentence_labels: List[int]  # sentence-level labels
    source: str
    is_harmful: bool            # response-level label
    harmful_onset: Optional[int]  # first token index where harmful content appears
    total_tokens: int
    num_sentences: int


def load_fineharm(split: str) -> List[StreamingSample]:
    """Load and parse a FineHarm split into StreamingSample objects."""
    path = DATA_DIR / f"FineHarm-{split}.json"
    with open(path) as f:
        raw = json.load(f)

    samples = []
    for d in raw:
        # Flatten words and word_labels from per-sentence lists
        tokens = []
        token_labels = []
        for sent_words, sent_labels in zip(d["words"], d["word_labels"]):
            tokens.extend(sent_words)
            token_labels.extend(sent_labels)

        is_harmful = any(l == 1 for l in d["sentence_labels"])

        # Find harmful onset: first token with label 1
        harmful_onset = None
        for i, l in enumerate(token_labels):
            if l == 1:
                harmful_onset = i
                break

        samples.append(StreamingSample(
            idx=d["idx"],
            prompt=d["prompt"],
            response=d["response"],
            tokens=tokens,
            token_labels=token_labels,
            sentence_labels=d["sentence_labels"],
            source=d["source"],
            is_harmful=is_harmful,
            harmful_onset=harmful_onset,
            total_tokens=len(tokens),
            num_sentences=d["sentence_num"],
        ))

    return samples


class PrefixDataset(Dataset):
    """
    Dataset for probe training: generates random prefixes from responses.

    Each sample is: [CLS] prompt [SEP] response_prefix [SEP]
    Label: 1 if prefix contains harmful content, 0 otherwise.
    """

    def __init__(
        self,
        samples: List[StreamingSample],
        tokenizer,
        max_length: int = 512,
        min_prefix_ratio: float = 0.05,
        max_prefix_ratio: float = 1.0,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_prefix_ratio = min_prefix_ratio
        self.max_prefix_ratio = max_prefix_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Randomly sample a prefix length
        min_len = max(1, int(sample.total_tokens * self.min_prefix_ratio))
        max_len = int(sample.total_tokens * self.max_prefix_ratio)
        prefix_len = random.randint(min_len, max_len)

        # Build prefix text
        prefix_tokens = sample.tokens[:prefix_len]
        prefix_text = " ".join(prefix_tokens)

        # Determine label for this prefix
        prefix_labels = sample.token_labels[:prefix_len]
        label = 1 if any(l == 1 for l in prefix_labels) else 0

        # Tokenize: prompt + response prefix
        encoding = self.tokenizer(
            sample.prompt,
            prefix_text,
            truncation="longest_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "prefix_len": prefix_len,
            "total_len": sample.total_tokens,
        }


class StreamingEvalDataset(Dataset):
    """
    Dataset for streaming evaluation: generates ALL prefixes for each response.

    Returns prefixes at regular intervals (e.g., every 5 tokens) for efficiency.
    """

    def __init__(
        self,
        samples: List[StreamingSample],
        tokenizer,
        max_length: int = 512,
        step_size: int = 5,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.step_size = step_size

        # Pre-compute all (sample_idx, prefix_len) pairs
        self.eval_points = []
        for s_idx, sample in enumerate(samples):
            for t in range(step_size, sample.total_tokens + 1, step_size):
                self.eval_points.append((s_idx, t))
            # Always include the last token
            if sample.total_tokens % step_size != 0:
                self.eval_points.append((s_idx, sample.total_tokens))

    def __len__(self):
        return len(self.eval_points)

    def __getitem__(self, idx):
        s_idx, prefix_len = self.eval_points[idx]
        sample = self.samples[s_idx]

        prefix_text = " ".join(sample.tokens[:prefix_len])
        prefix_labels = sample.token_labels[:prefix_len]
        label = 1 if any(l == 1 for l in prefix_labels) else 0

        encoding = self.tokenizer(
            sample.prompt,
            prefix_text,
            truncation="longest_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "sample_idx": s_idx,
            "prefix_len": prefix_len,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "is_harmful_response": int(sample.is_harmful),
            "harmful_onset": sample.harmful_onset if sample.harmful_onset is not None else -1,
        }


def split_train_calibration(
    train_samples: List[StreamingSample],
    cal_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[StreamingSample], List[StreamingSample]]:
    """Split training samples into probe training set and calibration set."""
    rng = random.Random(seed)

    harmful = [s for s in train_samples if s.is_harmful]
    safe = [s for s in train_samples if not s.is_harmful]

    rng.shuffle(harmful)
    rng.shuffle(safe)

    n_cal_h = int(len(harmful) * cal_ratio)
    n_cal_s = int(len(safe) * cal_ratio)

    cal_samples = harmful[:n_cal_h] + safe[:n_cal_s]
    train_samples_remaining = harmful[n_cal_h:] + safe[n_cal_s:]

    rng.shuffle(cal_samples)
    rng.shuffle(train_samples_remaining)

    return train_samples_remaining, cal_samples


def get_prefix_label(token_labels: List[int], prefix_len: int) -> int:
    """Get binary label for a prefix of given length."""
    return 1 if any(l == 1 for l in token_labels[:prefix_len]) else 0


def print_dataset_stats(samples: List[StreamingSample], name: str = ""):
    """Print statistics of a sample list."""
    harmful = [s for s in samples if s.is_harmful]
    safe = [s for s in samples if not s.is_harmful]
    lengths = [s.total_tokens for s in samples]

    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"Total samples: {len(samples)}")
    print(f"Harmful: {len(harmful)} ({len(harmful)/len(samples)*100:.1f}%)")
    print(f"Safe: {len(safe)} ({len(safe)/len(samples)*100:.1f}%)")
    print(f"Token length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    if harmful:
        onsets = [s.harmful_onset for s in harmful if s.harmful_onset is not None]
        if onsets:
            rel_onsets = [s.harmful_onset / s.total_tokens for s in harmful if s.harmful_onset is not None]
            print(f"Harmful onset: mean={np.mean(onsets):.0f}, median={np.median(onsets):.0f} tokens")
            print(f"Onset relative: mean={np.mean(rel_onsets):.2f}, median={np.median(rel_onsets):.2f}")


if __name__ == "__main__":
    train = load_fineharm("train")
    val = load_fineharm("val")
    test = load_fineharm("test")

    print_dataset_stats(train, "Train")
    print_dataset_stats(val, "Val")
    print_dataset_stats(test, "Test")

    train_probe, cal = split_train_calibration(train, cal_ratio=0.2)
    print_dataset_stats(train_probe, "Probe Training (after split)")
    print_dataset_stats(cal, "Calibration")

    # Save split info
    split_info = {
        "probe_train_indices": [s.idx for s in train_probe],
        "calibration_indices": [s.idx for s in cal],
        "cal_ratio": 0.2,
    }
    out_path = BASE_DIR / "outputs" / "data_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit info saved to {out_path}")
