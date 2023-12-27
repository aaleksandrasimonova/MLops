import fire

import mlops.train
import mlops.infer 


def hello():
    print('hello')


def train():
    mlops.train.train_model()
    pass

def infer():
    mlops.infer.infer()
    pass






if __name__ == '__main__':
    fire.Fire()
