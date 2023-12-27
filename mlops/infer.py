import pandas as pd
import torch
import torchvision
from hydra import compose, initialize

from .data_module import MNISTDataModule
from .model import MnistModel


def evaluate(dataloader, model):
    model.eval()
    y_pred, y_true = torch.Tensor(), torch.Tensor()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            images, labels = data
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            y_pred = torch.cat([y_pred, predicted.cpu()])
            y_true = torch.cat([y_true, labels.cpu()])
    return y_pred, y_true


def infer():
    initialize(version_base="1.3", config_path="../config")
    config = compose("config.yaml")
    model_file = config.model.model_path
    model = MnistModel(config.model.input_size, config.model.num_classes)
    model.load_state_dict(torchvision.torch.load(model_file))
    dataloader = MNISTDataModule(256).test_dataloader()
    y_pred, y_true = evaluate(dataloader, model)

    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df.to_csv("predict.csv", index=False)
    print('accuracy: ', (y_pred == y_true).sum() / len(y_true))


if __name__ == '__main__':
    infer()
