import hydra
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger

from utils import pprint, add_config


@hydra.main(config_path='config', config_name='config.yaml')
def main(conf):
    pprint(conf)

    train_loader = instantiate(conf.train_loader)

    model = instantiate(conf.model)
    print(model)

    trainer = instantiate(conf.trainer)

    if isinstance(trainer.logger, WandbLogger):
        add_config(trainer.logger.experiment, conf)

    for callback in trainer.callbacks:
        print(callback)

    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == '__main__':
    main()
