import os
import time
import pathlib
import numpy as np
import math
from absl import flags, app,logging
import torch 
import torch.nn as nn 

flags = flags.FLAGS


class MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.optimiser = None
        self.scheduler = None
        self.train_loader = None 
        self.validationset_loader = None 

    def fit(self, train_data, epochs, batch_size):

        if self.train_loader is None:
            self.train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size= batch_size 
                shuffle = True
            )
        pass

    def forward(self):
        pass


def main(_):
    logging.info('Welcome to the world of AI')


if __name__ == '__main__':
    app.run(main)



