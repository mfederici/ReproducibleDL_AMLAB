import pytorch_lightning as pl

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from model import VAE

def main():
    batch_size = 64
    z_dim = 64
    beta = 0.1
    lr = 1e-3
    num_workers = 6

    dataset = MNIST('/data', transform=ToTensor())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = VAE(z_dim=z_dim, beta=beta, lr=lr)
    print(model)

    trainer = pl.Trainer()

    trainer.fit(model, train_dataloader=train_loader)

if __name__ == '__main__':
    main()
