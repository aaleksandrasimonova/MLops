import git
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# from dvc import api
from hydra import compose, initialize

from .data_module import MNISTDataModule
from .model import MnistModel


class MNISTTraineModule(pl.LightningModule):
    def __init__(self, conf, git_id):
        super().__init__()
        self.save_hyperparameters()

        self.model = MnistModel(conf.model.input_size, conf.model.num_classes)

        self.loss = F.cross_entropy
        self.lr = conf.model.lr

    @staticmethod
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        loss = self.loss(out, labels)
        acc = MNISTTraineModule.accuracy(out.detach(), labels.detach())

        metrics = {'loss': loss.detach(), 'accuracy': acc}
        self.log_dict(
            metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        loss = self.loss(out, labels)
        acc = MNISTTraineModule.accuracy(out.detach(), labels.detach())

        metrics = {'loss': loss.detach(), 'accuracy': acc}
        self.log_dict(
            metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        return optimizer


def export_to_onnx(model, path_to_onnx):
    model.eval()
    test_input = torch.ones((1, 784))
    torch.onnx.export(
        model,
        test_input,
        path_to_onnx,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "BATCH_SIZE"},
            "output": {0: "BATCH_SIZE"},
        },
    )


def try_get_commit_id():
    try:
        repo = git.Repo(search_parent_directories=True)
        git_commit_id = repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        git_commit_id = "unknown"
    return git_commit_id


def train_model():
    initialize(version_base="1.3", config_path="../config")
    config = compose("config.yaml")

    data_module = MNISTDataModule(batch_size=config.model.batch_size)
    train_module = MNISTTraineModule(config, try_get_commit_id())

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config.logger.exp,
        tracking_uri=config.logger.path,
    )

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        max_epochs=config.model.num_epochs,
        logger=logger,
    )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    trainer.fit(train_module, train_dataloader, val_dataloader)

    torch.save(train_module.model.state_dict(), config.model.model_path)

    export_to_onnx(train_module.model, config.model.model_path_onnx)


if __name__ == "__main__":
    train_model()
