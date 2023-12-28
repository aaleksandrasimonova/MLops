import fire

import mlops.infer
import mlops.train


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


if __name__ == '__main__':
    fire.Fire()
