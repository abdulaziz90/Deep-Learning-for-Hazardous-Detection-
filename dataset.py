import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from torchvision.datasets import ImageFolder
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.datasets.utils import list_files
from PIL import Image
from sklearn.model_selection import train_test_split
import h5py
import time
import subprocess

class CustomNpyDataset(Dataset):
    def __init__(self, data_list, target_list):
        self.data_list = data_list
        self.target_list = target_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.target_list[idx]

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        temporal_resolution: int,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.temporal_resolution = temporal_resolution


    def setup_stage1(self, stage: Optional[str] = None) -> None:

        print(f"======= Loading Data ... This might takes a few minutes =======")
        dataset_name = os.path.join(self.data_dir, 'dataset_T='+str(self.temporal_resolution)+'.h5')

        if not os.path.exists(dataset_name) or os.path.getsize(dataset_name) == 0:
            print(f"{dataset_name} does not exist. Downloading now...")
            if self.temporal_resolution == 20:
                url = 'https://www.dropbox.com/scl/fi/9ckf53m1g2zhl0792gg9w/dataset_T-20.h5?rlkey=2mordixt76ykhqg8wy4hr2i2k&dl=1'
                subprocess.run(['wget', '-O', dataset_name, url], check=True)
                print(f"Download complete.")
            elif self.temporal_resolution == 10:
                url = 'https://www.dropbox.com/scl/fi/pzsb352ern6jpazqrgbk7/dataset_T-10.h5?rlkey=xpncti80xqot9w0iba8gaux8o&dl=1'
                subprocess.run(['wget', '-O', dataset_name, url], check=True)
                print(f"Download complete.")
            else: 
                raise ValueError(f"Invalid temporal resolution: {self.temporal_resolution}. Expected 10 or 20.")
        try:
            with h5py.File(dataset_name, 'r') as hf:
                X_train = torch.tensor(hf['time_series_cubes_train'][:], dtype=torch.float32)
                y_train = torch.tensor(hf['clouds_train'][:], dtype=torch.float32)
                X_test = torch.tensor(hf['time_series_cubes_test'][:], dtype=torch.float32)
                y_test = torch.tensor(hf['clouds_test'][:], dtype=torch.float32)
        except OSError as e:
            print(f"An error occurred: {e}")

        y_train = y_train.unsqueeze(1)
        y_test = y_test.unsqueeze(1)

        X_train = X_train.permute(0, 4, 1, 2, 3)
        X_test = X_test.permute(0, 4, 1, 2, 3)

        print(f"X_train: {X_train.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_test: {y_test.shape}")

        # create datasets
        self.train_dataset = CustomNpyDataset(X_train, y_train)
        self.val_dataset = CustomNpyDataset(X_test, y_test)

    def setup_stage2(self, stage: Optional[str] = None) -> None:

        print(f"======= Loading Data ... This might takes a few minutes =======")
        dataset_name = os.path.join(self.data_dir, 'dataset_T='+str(self.temporal_resolution)+'.h5')

        if not os.path.exists(dataset_name) or os.path.getsize(dataset_name) == 0:
            print(f"{dataset_name} does not exist. Downloading now...")
            if self.temporal_resolution == 20:
                url = 'https://www.dropbox.com/scl/fi/9ckf53m1g2zhl0792gg9w/dataset_T-20.h5?rlkey=2mordixt76ykhqg8wy4hr2i2k&dl=1'
                subprocess.run(['wget', '-O', dataset_name, url], check=True)
                print(f"Download complete.")
            elif self.temporal_resolution == 10:
                url = 'https://www.dropbox.com/scl/fi/pzsb352ern6jpazqrgbk7/dataset_T-10.h5?rlkey=xpncti80xqot9w0iba8gaux8o&dl=1'
                subprocess.run(['wget', '-O', dataset_name, url], check=True)
                print(f"Download complete.")
            else: 
                raise ValueError(f"Invalid temporal resolution: {self.temporal_resolution}. Expected 10 or 20.")
        try:
            file_path = os.path.join(self.data_dir, 'outputs_stage1_T=' + str(self.temporal_resolution) + '.h5')
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                print(f"The estimated clouds of stage 1 do not exist ... Downloading now....")
                if self.temporal_resolution == 20:
                    url = 'https://www.dropbox.com/scl/fi/mokzf9auwgs57440pyfw3/outputs_stage1_T-20.h5?rlkey=tiv1ee0n4u1qyqr8owlez0b9x&dl=1'
                    subprocess.run(['wget', '-O', file_path, url], check=True)  # Changed dataset_name to file_path
                elif self.temporal_resolution == 10:
                    url = 'https://www.dropbox.com/scl/fi/g8kbeuw1i1gqgh7j39xgn/outputs_stage1_T-10.h5?rlkey=qh008w7s8jdknmf0432lq8wpg&dl=1'
                    subprocess.run(['wget', '-O', file_path, url], check=True)  # Changed dataset_name to file_path
                else: 
                    raise ValueError(f"Invalid temporal resolution: {self.temporal_resolution}. Expected 10 or 20.")

            print(f"Loading estimated clouds of stage 1")
            with h5py.File(file_path, 'r') as hf:  # 'r' for read mode
                X_test = torch.tensor(hf['output_test'][:], dtype=torch.float32)
            with h5py.File(dataset_name, 'r') as hf:
                y_train = torch.tensor(hf['params_train'][:], dtype=torch.float32)
                y_test = torch.tensor(hf['params_test'][:], dtype=torch.float32) 
                X_train = torch.tensor(hf['clouds_train'][:], dtype=torch.float32)
                X_train = X_train.unsqueeze(1)
        except (OSError, ValueError) as e:  # Combined both error types
            print(f"An error occurred: {e}")
            
        # create datasets
        self.train_dataset = CustomNpyDataset(X_train, y_train)
        self.val_dataset = CustomNpyDataset(X_test, y_test)

#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
     