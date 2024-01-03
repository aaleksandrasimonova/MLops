from functools import lru_cache

import numpy as np
import torch
from hydra import compose, initialize
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype

from .data_module import MNISTDataModule


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_infer(input):
    triton_client = get_client()

    triton_input = InferInput(
        name="input",
        shape=input.shape,
        datatype=np_to_triton_dtype(input.dtype),
    )

    triton_input.set_data_from_numpy(input, binary_data=True)

    triton_output = InferRequestedOutput("output", binary_data=True)
    query_response = triton_client.infer(
        "mnist_model", [triton_input], outputs=[triton_output]
    )

    output = query_response.as_numpy("output")

    return output


def test_triton():
    MNISTDataModule.load_data_dvc()
    initialize(version_base="1.3", config_path="../config")
    config = compose("config.yaml")

    data = MNISTDataModule(batch_size=config.model.batch_size)

    test_dataloader = data.test_dataloader()
    y_true, y_pred = np.array([]), np.array([])
    for images, labels in test_dataloader:
        images = torch.nn.Flatten()(images)
        images = images.detach().numpy()
        predicted = call_triton_infer(images)

        y_true = np.append(y_true, labels)
        y_pred = np.append(y_pred, predicted.argmax(axis=1))

    print('accuracy: ', (y_pred == y_true).sum() / len(y_true))


if __name__ == "__main__":
    test_triton()
