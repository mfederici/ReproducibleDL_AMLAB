import hydra

import pytorch_lightning as pl

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from callbacks import SamplingCallback, ReconstructionCallback
from model import VAE


@hydra.main(config_path='config', config_name='config.yaml')
def main(conf):

    dataset = MNIST('/data', transform=ToTensor())
    train_loader = DataLoader(dataset,
                              batch_size=conf.batch_size,
                              shuffle=True,
                              num_workers=conf.num_workers)

    model = VAE(z_dim=conf.z_dim, beta=conf.beta, lr=conf.lr)
    print(model)

    trainer = pl.Trainer(callbacks=[SamplingCallback(), ReconstructionCallback()])

    for callback in trainer.callbacks:
        print(callback)

    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == '__main__':
    main()
