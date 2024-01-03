import mlflow
import numpy as np
import onnx
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


def get_model_mlflow_from_onnx(
    input_size, num_classes, input, model_path_onnx
):
    model_onnx = onnx.load_model(model_path_onnx)

    model = MnistModel(input_size, num_classes)
    model.eval()

    signature = mlflow.models.infer_signature(
        input.numpy(),
        model(input).detach().numpy(),
    )
    model_info = mlflow.onnx.log_model(
        model_onnx,
        "model",
        signature=signature,
    )

    return model_info


def run_server():
    initialize(version_base="1.3", config_path="../config")
    cfg = compose("config.yaml")

    mlflow.set_tracking_uri(cfg.logger.path)
    mlflow.set_experiment(cfg.logger.exp)

    with mlflow.start_run():
        export_input = torch.ones((1, 784))
        model_info = get_model_mlflow_from_onnx(
            cfg.model.input_size,
            cfg.model.num_classes,
            export_input,
            cfg.model.model_path_onnx,
        )

        onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)

        data = MNISTDataModule(cfg.model.batch_size)

        test_dataloader = data.test_dataloader()
        y_true, y_pred = np.array([]), np.array([])
        for images, labels in test_dataloader:
            images = torch.nn.Flatten()(images)
            images = images.detach().numpy()
            predicted = onnx_pyfunc.predict(images)['output']

            y_true = np.append(y_true, labels)
            y_pred = np.append(y_pred, predicted.argmax(axis=1))

    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df.to_csv("predict_onnx.csv", index=False)
    print('accuracy: ', (y_pred == y_true).sum() / len(y_true))

    pass


if __name__ == '__main__':
    infer()
