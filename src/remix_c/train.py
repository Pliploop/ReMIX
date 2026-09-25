"""Train ReMIX-C. Every setting is a hydra key (conf/), e.g.

  PYTHONPATH=src python -m remix_c.train model=small objective=lejepa data.filter_threshold=4
  PYTHONPATH=src python -m remix_c.train --cfg job      # print the resolved config
"""
import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from remix_c.model import RemixC


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)                                   # checkpoints store plain values, no interpolations
    L.seed_everything(cfg.seed, workers=True)
    logger = WandbLogger(**cfg.logger, config=OmegaConf.to_container(cfg, resolve=True))
    trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=[instantiate(c) for c in cfg.callbacks.values()])
    model, data = RemixC(cfg.model, cfg.objective, cfg.optim), instantiate(cfg.data)
    if cfg.validate_first and not cfg.resume:
        trainer.validate(model, data)                        # full val pass logged at step 0
    trainer.fit(model, data, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
