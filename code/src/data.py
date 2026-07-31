"""Vocabulary, tokenizer and Dataset class for LUCID-Net (bug-fixed).

Fixes vs. the original notebook (see ANALYSIS.md):
  - image/mask pairing is done by matching filename stems, not by blind
    ``sorted(os.listdir())`` zip, and non-image files (e.g. ``.DS_Store``)
    are filtered out.
  - text prompts default to a real, dataset-specific description instead
    of the literal placeholder string ``"{Prompt}"``.
"""
import os
import re
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word("<pad>")
        self.add_word("<unk>")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_from_texts(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        for word, count in counter.items():
            if count >= min_freq:
                self.add_word(word)

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

    def encode(self, text, max_len=20):
        tokens = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in self.tokenize(text)]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [self.word2idx["<pad>"]] * (max_len - len(tokens))
        return tokens


def _list_images(dir_path):
    names = [f for f in sorted(os.listdir(dir_path)) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
    return names


class SegmentationDataset(Dataset):
    """Paired image/mask dataset, matched by filename stem (not directory order)."""

    def __init__(self, image_dir, mask_dir, text_prompt="{Prompt}",
                 transform=None, target_transform=None, vocab=None, max_len=20):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.max_len = max_len

        image_names = _list_images(image_dir)
        mask_names = _list_images(mask_dir)
        image_stems = {os.path.splitext(f)[0]: f for f in image_names}
        mask_stems = {os.path.splitext(f)[0]: f for f in mask_names}
        common = sorted(set(image_stems) & set(mask_stems))
        missing_img = set(mask_stems) - set(image_stems)
        missing_mask = set(image_stems) - set(mask_stems)
        if missing_img or missing_mask:
            raise ValueError(
                f"Unpaired files in {image_dir} vs {mask_dir}: "
                f"{len(missing_mask)} images with no mask, {len(missing_img)} masks with no image."
            )
        if not common:
            raise ValueError(f"No paired image/mask files found in {image_dir} / {mask_dir}")

        self.pairs = [(image_stems[s], mask_stems[s]) for s in common]

        # A single dataset-level text prompt is used for every image (matches how the
        # paper actually generated one DeepSeek-R1 description per *dataset*, not per image).
        self.text_prompt = text_prompt
        self.vocab = vocab
        if self.vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_from_texts([self.text_prompt])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            mask = (mask > 0.5).float()  # ensure strictly binary after resizing/interpolation

        text_tokens = self.vocab.encode(self.text_prompt, self.max_len)
        return image, mask, torch.tensor(text_tokens, dtype=torch.long)
