
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class MSFFANDataset(Dataset):

    def __init__(self, project_root, split="train"):

        self.project_root = Path(project_root)
        self.dataset_root = self.project_root / "data" / "dataset"

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.split = split

        # Load metadata
        metadata_path = self.dataset_root / "dataset_metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}"
            )

        metadata = pd.read_csv(metadata_path)

        # Keep only the requested split
        self.metadata = metadata[
            metadata["split"] == split
        ].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise RuntimeError(
                f"No samples found for split: {split}"
            )

        self.hr_dir = self.dataset_root / split / "HR"
        self.lr_dir = self.dataset_root / split / "LR"

        if not self.hr_dir.exists():
            raise FileNotFoundError(
                f"HR directory not found: {self.hr_dir}"
            )

        if not self.lr_dir.exists():
            raise FileNotFoundError(
                f"LR directory not found: {self.lr_dir}"
            )

    def __len__(self):
        return len(self.metadata)

    def _find_image(self, directory, sample_id):

        matches = list(
            directory.glob(f"{sample_id}_*.png")
        )

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No image found for {sample_id}"
            )

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple images found for {sample_id}"
            )

        return matches[0]

    def _load_image(self, path):

        image = Image.open(path).convert("RGB")

        array = np.asarray(
            image,
            dtype=np.float32
        )

        # H,W,C -> C,H,W
        tensor = torch.from_numpy(
            array.transpose(2, 0, 1)
        ).contiguous()

        # [0,255] -> [0,1]
        tensor = tensor / 255.0

        return tensor

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        sample_id = row["sample_id"]

        # Locate images
        hr_path = self._find_image(
            self.hr_dir,
            sample_id
        )

        lr_path = self._find_image(
            self.lr_dir,
            sample_id
        )

        # Load images
        hr = self._load_image(hr_path)
        lr = self._load_image(lr_path)

        # Metadata
        metadata = {
            "sample_id": sample_id,
            "split": row["split"],
            "location": row["location"],
            "x": int(row["x"]),
            "y": int(row["y"]),
            "patch_size": int(row["patch_size"]),
            "scale_factor": int(row["scale_factor"]),
            "blur_type": row["blur_type"],
            "jpeg_quality": int(row["jpeg_quality"]),
            "contrast_factor": float(row["contrast_factor"]),
            "source_file": row["source_file"],
            "seed": int(row["seed"]),
            "hr_path": str(hr_path),
            "lr_path": str(lr_path),
        }

        return {
            "lr": lr,
            "hr": hr,
            "metadata": metadata,
        }
