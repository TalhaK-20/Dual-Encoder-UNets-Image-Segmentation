"""Training / evaluation engine shared by all three model variants.

Fixes vs. the original notebook (see ANALYSIS.md):
  - real train/val/test loop with early stopping + best-checkpoint saving
    (the original trained blindly for 1000 epochs with no val set, no
    checkpointing at all).
  - full reproducibility seeding.
  - test-time evaluation NEVER sees the ground-truth mask for the LUCID-Net
    variant (spatial conditioning uses a zero mask at eval, matching the
    paper's own claim that "only the image and no mask is available" at
    inference) — the original notebook fed the real test mask into
    spatial_encoder() during evaluate(), which is label leakage.
  - single sigmoid activation (the original applied sigmoid once inside the
    model AND again in evaluate()).
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SegmentationDataset
from .diffusion import Diffusion
from .losses import BCEDiceLoss
from .metrics import MetricAccumulator
from .models import AttentionUNet, LUCIDNet, SpatialEncoder, TextEncoder, TimmEncoderUNet

try:
    import torchvision.transforms as T
except ImportError:  # pragma: no cover
    T = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(train=True, image_size=128):
    img_ops = [T.Resize((image_size, image_size))]
    if train:
        img_ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15)]
    img_ops += [T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]

    mask_ops = [T.Resize((image_size, image_size)), T.ToTensor()]
    # NOTE: flips/rotations must be applied identically to image+mask to stay aligned.
    # We therefore do joint augmentation manually in a wrapper instead of torchvision
    # Compose (which would sample independent random params per call). See JointTransform.
    return img_ops, mask_ops


class JointTransform:
    """Applies the *same* random flip/rotation to an (image, mask) pair."""

    def __init__(self, train, image_size=128):
        self.train = train
        self.image_size = image_size
        self.img_norm = T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    def __call__(self, image, mask):
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))
        if self.train:
            if random.random() < 0.5:
                image, mask = T.functional.hflip(image), T.functional.hflip(mask)
            if random.random() < 0.5:
                image, mask = T.functional.vflip(image), T.functional.vflip(mask)
            angle = random.uniform(-15, 15)
            image = T.functional.rotate(image, angle)
            mask = T.functional.rotate(mask, angle)
        image = self.img_norm(T.functional.to_tensor(image))
        mask = T.functional.to_tensor(mask)
        return image, mask


class _JointDataset(torch.utils.data.Dataset):
    """Wraps SegmentationDataset(transform=None) and applies JointTransform per-sample."""

    def __init__(self, base, joint_transform):
        self.base = base
        self.joint_transform = joint_transform
        self.vocab = base.vocab

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask, text = self.base[idx]
        image, mask = self.joint_transform(image, mask)
        mask = (mask > 0.5).float()
        return image, mask, text


def make_datasets(data_root, text_prompt, image_size=128):
    def _base(split):
        return SegmentationDataset(
            os.path.join(data_root, split, "images"),
            os.path.join(data_root, split, "masks"),
            text_prompt=text_prompt,
            transform=None,
            target_transform=None,
            vocab=None,
        )

    train_base = _base("train")
    val_base = SegmentationDataset(
        os.path.join(data_root, "val", "images"), os.path.join(data_root, "val", "masks"),
        text_prompt=text_prompt, vocab=train_base.vocab,
    )
    test_base = SegmentationDataset(
        os.path.join(data_root, "test", "images"), os.path.join(data_root, "test", "masks"),
        text_prompt=text_prompt, vocab=train_base.vocab,
    )

    train_ds = _JointDataset(train_base, JointTransform(train=True, image_size=image_size))
    val_ds = _JointDataset(val_base, JointTransform(train=False, image_size=image_size))
    test_ds = _JointDataset(test_base, JointTransform(train=False, image_size=image_size))
    return train_ds, val_ds, test_ds


class ModelBundle:
    """Wraps whichever sub-modules a given model variant needs behind one call signature."""

    def __init__(self, variant, vocab_size, device):
        self.variant = variant
        self.device = device
        if variant == "lucidnet":
            self.unet = LUCIDNet().to(device)
            self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=128).to(device)
            self.spatial_encoder = SpatialEncoder(in_channels=1, base_channels=64).to(device)
            self.modules = [self.unet, self.text_encoder, self.spatial_encoder]
        elif variant == "attn_unet":
            self.unet = AttentionUNet().to(device)
            self.modules = [self.unet]
        elif variant == "timm_unet":
            self.unet = TimmEncoderUNet(backbone="resnet34", pretrained=True).to(device)
            self.modules = [self.unet]
        else:
            raise ValueError(variant)

    def parameters(self):
        for m in self.modules:
            yield from m.parameters()

    def train(self):
        for m in self.modules:
            m.train()
        if self.variant == "lucidnet":
            # DenseNet-style frozen-BN-in-eval isn't applicable here (nothing pretrained/frozen
            # in LUCID-Net's own encoder), all submodules train normally.
            pass

    def eval(self):
        for m in self.modules:
            m.eval()

    def forward(self, images, masks, texts, is_train):
        if self.variant == "lucidnet":
            text_embed = self.text_encoder(texts)
            if is_train:
                spatial_embed, _ = self.spatial_encoder(masks)
            else:
                # Bug fix: no ground-truth mask at test time (label leakage in the
                # original notebook). Condition on a zero mask instead.
                zero_mask = torch.zeros_like(masks)
                spatial_embed, _ = self.spatial_encoder(zero_mask)
            return self.unet(images, text_embed, spatial_embed)
        return self.unet(images)

    def state_dicts(self):
        return {f"module_{i}_{type(m).__name__}": m.state_dict() for i, m in enumerate(self.modules)}

    def load_state_dicts(self, state):
        for i, m in enumerate(self.modules):
            key = f"module_{i}_{type(m).__name__}"
            m.load_state_dict(state[key])


def train_and_evaluate(
    dataset_name, data_root, text_prompt, variant,
    out_dir, max_epochs=300, patience=40, batch_size=4, lr=1e-3,
    diffusion_T=1000, image_size=128, seed=42, device=None,
):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    train_ds, val_ds, test_ds = make_datasets(data_root, text_prompt, image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    bundle = ModelBundle(variant, vocab_size=len(train_ds.vocab.word2idx), device=device)
    diffusion = Diffusion(T=diffusion_T, device=device) if variant == "lucidnet" else None

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(bundle.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=15)

    history = []
    best_val_dice = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    ckpt_path = os.path.join(out_dir, "best.pt")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        bundle.train()
        train_loss_sum, n_batches = 0.0, 0
        for images, masks, texts in train_loader:
            images, masks, texts = images.to(device), masks.to(device), texts.to(device)

            if diffusion is not None:
                t = torch.randint(0, diffusion.T, (images.size(0),), device=device)
                noisy_images, _ = diffusion.q_sample(images, t)
            else:
                noisy_images = images

            pred = bundle.forward(noisy_images, masks, texts, is_train=True)
            loss = criterion(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        # ---- validation ----
        bundle.eval()
        val_acc = MetricAccumulator()
        val_loss_sum, n_val_batches = 0.0, 0
        with torch.no_grad():
            for images, masks, texts in val_loader:
                images, masks, texts = images.to(device), masks.to(device), texts.to(device)
                pred = bundle.forward(images, masks, texts, is_train=False)  # clean image, no mask leak
                val_loss_sum += criterion(pred, masks).item()
                n_val_batches += 1
                val_acc.update(pred, masks)
        val_loss = val_loss_sum / max(n_val_batches, 1)
        val_summary = val_acc.summary()
        val_dice = val_summary["dice_mean"]

        scheduler.step(val_dice)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_dice": val_dice, "val_iou": val_summary["iou_mean"], "seconds": elapsed})

        improved = val_dice > best_val_dice
        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save({"state": bundle.state_dicts(), "epoch": epoch, "val_dice": val_dice}, ckpt_path)
        else:
            epochs_without_improve += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"[{dataset_name}/{variant}] epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}"
                  f"{'  *best*' if improved else ''}")

        if epochs_without_improve >= patience:
            print(f"[{dataset_name}/{variant}] early stopping at epoch {epoch} "
                  f"(no val_dice improvement for {patience} epochs; best={best_val_dice:.4f} @ epoch {best_epoch})")
            break

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ---- final test evaluation using the BEST checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=device)
    bundle.load_state_dicts(ckpt["state"])
    bundle.eval()

    test_acc = MetricAccumulator()
    qualitative = []
    with torch.no_grad():
        for bi, (images, masks, texts) in enumerate(test_loader):
            images, masks, texts = images.to(device), masks.to(device), texts.to(device)
            pred = bundle.forward(images, masks, texts, is_train=False)
            test_acc.update(pred, masks)
            if bi == 0:
                qualitative = (images.cpu(), masks.cpu(), pred.cpu())

    test_summary = test_acc.summary()
    test_summary["best_epoch"] = ckpt["epoch"]
    test_summary["dataset"] = dataset_name
    test_summary["variant"] = variant
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    return {
        "history": history,
        "test_summary": test_summary,
        "confusion_matrix": test_acc.confusion_matrix(),
        "roc": test_acc.roc_curve(),
        "pr": test_acc.pr_curve(),
        "qualitative": qualitative,
        "ckpt_path": ckpt_path,
    }
