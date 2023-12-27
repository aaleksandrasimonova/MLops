import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size
        mnist_dataset = torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        self.train_dataset, self.val_dataset = random_split(
            mnist_dataset, [50000, 10000]
        )

        self.test_dataset = torchvision.datasets.MNIST(
            root="data",
            train=False,
            download=True,
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
