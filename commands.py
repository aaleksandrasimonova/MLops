import fire

import mlops.infer
import mlops.train
import mlops.triton


def hello():
    print('hello')


def train():
    mlops.train.train_model()
    pass


def infer():
    mlops.infer.infer()
    pass


def test_dvc():
    import mlops.data_module

    mlops.data_module.MNISTDataModule.load_data_dvc()


def run_server():
    mlops.infer.run_server()


def test_triton():
    mlops.triton.test_triton()


if __name__ == '__main__':
    fire.Fire()
