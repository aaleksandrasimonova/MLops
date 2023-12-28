import os

import dvc.api
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()

        MNISTDataModule.load_data_dvc()

        self.batch_size = batch_size
        mnist_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=False,
            transform=torchvision.transforms.ToTensor(),
        )
        self.train_dataset, self.val_dataset = random_split(
            mnist_dataset, [50000, 10000]
        )

        self.test_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=False,
            transform=torchvision.transforms.ToTensor(),
        )

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return dataloader

    @staticmethod
    def load_data_dvc():
        if os.path.exists("./data/MNIST"):
            return
        url = 'https://github.com/aaleksandrasimonova/MLops'
        fs = dvc.api.DVCFileSystem(url, rev='main')
        fs.get("./data", "./", recursive=True, download=True)
