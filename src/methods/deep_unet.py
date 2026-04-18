"""Deep-learning segmentation: U-Net with ImageNet-pretrained encoder.

Implemented via `segmentation_models_pytorch` (model + Dice loss),
PyTorch (training loop), and `albumentations` (data augmentation).

Pipeline:
  Training:
    1. Resize every image/mask to IMG_SIZE (must be divisible by 32 for
       a 5-level U-Net, so 352 with 350x350 inputs).
    2. Augment: flips, 90-degree rotations, small shift/scale/rotate,
       brightness/contrast jitter.
    3. Train a U-Net with a ResNet18 ImageNet-pretrained encoder using
       a weighted sum of BCE-with-logits and Dice losses.
    4. After each epoch, evaluate on the validation split; keep the
       checkpoint with the best val IoU.

  Inference:
    1. Resize image to IMG_SIZE, normalise with ImageNet stats.
    2. Forward pass -> logits -> sigmoid -> 0.5 threshold.
    3. Resize mask back to the original H, W with nearest-neighbour
       so binary values are preserved.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data_utils import load_image, load_mask
from src.methods.base import Segmenter


# ImageNet normalisation stats. Using them matches what the pretrained
# ResNet18 encoder was trained on, so its early-layer features transfer
# cleanly.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Size at which we run the U-Net. Must be divisible by 2^(num-downsamples)
# = 2^5 = 32 for the default 5-level U-Net. 352 is the smallest multiple
# of 32 at or above the native EWS size of 350.
IMG_SIZE = 352


def get_device() -> torch.device:
    """Pick the best available device: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def read_split_rows(csv_path: Path) -> list[dict]:
    with Path(csv_path).open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_train_transform(image_size: int = IMG_SIZE) -> A.Compose:
    """Augmentations used during training.

    Flips/rotations are safe because plant shapes are orientation-free
    from above. Brightness/contrast jitter helps generalise to lighting
    conditions not in train.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15, contrast_limit=0.15, p=0.5
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(image_size: int = IMG_SIZE) -> A.Compose:
    """No augmentation — just resize + normalise. Used for val and test."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class EWSDataset(Dataset):
    """Loads paired (image, mask) samples from a split CSV.

    Returns a dict per sample with preprocessed torch tensors ready for
    the DataLoader.
    """

    def __init__(self, rows: list[dict], transform: A.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # load_image with normalize=False returns float32 in [0, 255].
        # albumentations expects uint8 images for Normalize to behave
        # the way the Normalize stats assume, so cast back here.
        image = load_image(
            Path(row['image_path']), color_space='rgb', normalize=False
        ).clip(0, 255).astype(np.uint8)
        mask = load_mask(Path(row['mask_path']), as_binary=True)

        out = self.transform(image=image, mask=mask)
        return {
            'image': out['image'],              # (3, H, W) float
            'mask': out['mask'].float(),        # (H, W)    float in {0, 1}
        }


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_fn: nn.BCEWithLogitsLoss,
    dice_fn: nn.Module,
    bce_weight: float,
    dice_weight: float,
) -> torch.Tensor:
    # BCE pushes the per-pixel logit toward the correct side of zero.
    # Dice directly optimises overlap, which matters more on imbalanced
    # masks. A weighted sum of the two is a standard, robust choice.
    return bce_weight * bce_fn(logits, targets) + dice_weight * dice_fn(logits, targets)


@torch.no_grad()
def compute_val_iou(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 1e-7,
) -> float:
    """Compute mean IoU across the validation split at model resolution."""
    model.eval()
    total_intersection = 0.0
    total_union = 0.0
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)                   # (N, H, W)
        logits = model(images)                             # (N, 1, H, W)
        preds = (torch.sigmoid(logits).squeeze(1) > 0.5).float()
        intersection = (preds * masks).sum().item()
        union = ((preds + masks) > 0).float().sum().item()
        total_intersection += intersection
        total_union += union
    return total_intersection / (total_union + eps)


class UNetSegmenter(Segmenter):
    """U-Net (ResNet18 encoder, ImageNet pretrained) plant segmenter."""

    name = 'dl_unet_r18'

    def __init__(
        self,
        encoder_name: str = 'resnet18',
        encoder_weights: str = 'imagenet',
        num_epochs: int = 30,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        image_size: int = IMG_SIZE,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        num_workers: int = 0,
        random_state: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        # Encoder choice. ResNet18 is small enough to train on CPU in
        # minutes and large enough to give competitive IoU on 150 images.
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        # Training hyperparameters.
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Input resolution. Leave at 352 unless you know what you're doing
        # — it must stay divisible by 32 for a 5-level U-Net.
        self.image_size = image_size

        # Loss weights. 0.5 each is a safe default.
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        # DataLoader workers. Keep 0 on macOS / small datasets to avoid
        # multiprocessing startup overhead swamping the actual training.
        self.num_workers = num_workers

        self.random_state = random_state
        self.device = device or get_device()

        # Populated by fit(). Used by predict().
        self.model: Optional[nn.Module] = None
        self.eval_transform = build_eval_transform(self.image_size)

    def _build_model(self) -> nn.Module:
        return smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=3,
            classes=1,
        )

    def fit(
        self,
        train_csv: Path,
        val_csv: Optional[Path] = None,
    ) -> float:
        """Train on `train_csv`, optionally tracking best by val IoU.

        Returns wall-clock training time in seconds (loading + training)
        so the caller can record it in the evaluation summary row.
        """
        # Seed every RNG we can reach for reproducibility.
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        print(f'Using device: {self.device}')

        # --- Data loaders -------------------------------------------------
        train_rows = read_split_rows(train_csv)
        train_ds = EWSDataset(train_rows, build_train_transform(self.image_size))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda'),
        )

        val_loader: Optional[DataLoader] = None
        if val_csv is not None:
            val_rows = read_split_rows(val_csv)
            val_ds = EWSDataset(val_rows, build_eval_transform(self.image_size))
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device.type == 'cuda'),
            )

        # --- Model, loss, optimiser ---------------------------------------
        model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        bce_fn = nn.BCEWithLogitsLoss()
        dice_fn = smp.losses.DiceLoss(mode='binary')

        # --- Training loop ------------------------------------------------
        # We keep the best-by-val-IoU checkpoint in memory so that after
        # training we can restore it (protects against overfitting in
        # later epochs).
        best_val_iou = -1.0
        best_state: Optional[dict] = None

        t_start = time.perf_counter()
        for epoch in range(1, self.num_epochs + 1):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                images = batch['image'].to(self.device)
                # Model outputs (N, 1, H, W); masks are (N, H, W). Add a
                # channel dim so losses broadcast cleanly.
                masks = batch['mask'].to(self.device).unsqueeze(1)

                optimizer.zero_grad()
                logits = model(images)
                loss = dice_bce_loss(
                    logits, masks, bce_fn, dice_fn,
                    self.bce_weight, self.dice_weight,
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_ds)

            val_iou: Optional[float] = None
            if val_loader is not None:
                val_iou = compute_val_iou(model, val_loader, self.device)
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    # Clone weights to CPU so we don't keep extra GPU memory.
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

            msg = f'Epoch {epoch}/{self.num_epochs} | train_loss={epoch_loss:.4f}'
            if val_iou is not None:
                msg += f' | val_iou={val_iou:.4f} (best={best_val_iou:.4f})'
            print(msg)

        # Restore best checkpoint if we tracked one.
        if best_state is not None:
            model.load_state_dict(best_state)

        self.model = model
        train_time = time.perf_counter() - t_start
        print(f'Training complete in {train_time:.1f} s')
        if best_val_iou >= 0:
            print(f'Best val IoU: {best_val_iou:.4f}')
        return train_time

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                'UNetSegmenter.predict() called before fit(). '
                'Call .fit(train_csv) first.'
            )

        # Defensive normalisation mirroring the other methods.
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max() > 1.5:
            image = image / 255.0

        h, w, _ = image.shape

        # Albumentations expects uint8 RGB for its default Normalize
        # statistics to apply correctly (the stats are derived from
        # [0, 255] ImageNet inputs rescaled to [0, 1] internally).
        img_u8 = (image * 255.0).clip(0, 255).astype(np.uint8)
        transformed = self.eval_transform(image=img_u8)
        tensor = transformed['image'].unsqueeze(0).to(self.device)  # (1, 3, IMG, IMG)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)                            # (1, 1, IMG, IMG)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0)    # (IMG, IMG)
            probs = probs.cpu().numpy()

        mask = (probs > 0.5).astype(np.uint8)

        # Resize back to the original (H, W) if we upsampled/downsampled.
        # Nearest-neighbour preserves the {0, 1} binary values.
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return mask
