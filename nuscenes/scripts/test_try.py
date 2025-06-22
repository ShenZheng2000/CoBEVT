from pathlib import Path
import pytorch_lightning as pl
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy

from cross_view_transformer.common import setup_config, setup_experiment


CONFIG_PATH = Path.cwd() / 'config'  # Adjust if needed
CONFIG_NAME = 'config.yaml'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Load model and datamodule
    model_module, data_module, _ = setup_experiment(cfg)

    # # Load latest checkpoint if it exists
    # ckpt_path = (
    #     Path(cfg.experiment.save_dir) / cfg.experiment.uuid / "checkpoints" / "model.ckpt"
    # )
    # ckpt_path = str(ckpt_path) if ckpt_path.exists() else None

    # NOTE: hardcode checkpoint path for now
    ckpt_path = "/home/shenzheng_google_com/Projects/Inf_Perception/Methods/CoBEVT/nuscenes/logs/sinbevt_nuscenes_vehicle_50k.ckpt"

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        **cfg.trainer
    )

    # trainer.test(model_module, datamodule=data_module, ckpt_path=ckpt_path)
    trainer.validate(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
