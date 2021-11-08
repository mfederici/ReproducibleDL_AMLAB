import hydra
from hydra.utils import instantiate

from utils import pprint


@hydra.main(config_path='config', config_name='config.yaml')
def main(conf):
    pprint(conf)

    train_loader = instantiate(conf.train_loader)

    model = instantiate(conf.model)
    print(model)

    trainer = instantiate(conf.trainer)

    for callback in trainer.callbacks:
        print(callback)

    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == '__main__':
    main()
