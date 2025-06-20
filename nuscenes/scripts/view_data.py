import hydra
import numpy as np
import cv2

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.common import setup_config, setup_data_module, setup_viz


def setup(cfg):
    print('See training set by adding +split=train')
    print('Shuffle samples by adding +shuffle=false')

    cfg.loader.batch_size = 1

    if 'split' not in cfg:
        cfg.split = 'val'

    if 'shuffle' not in cfg:
        cfg.shuffle = False


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    setup_config(cfg, setup)

    data = setup_data_module(cfg)
    viz = setup_viz(cfg)
    loader = data.get_split(cfg.split, shuffle=cfg.shuffle)

    print(f'{cfg.split}: {len(loader)} total samples')

    # 🔸 ADD: create folder to save output
    save_dir = Path(hydra.utils.get_original_cwd()) / 'debug_output'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 🔸 ADD: counter for filenames
    i = 0

    for batch in tqdm(loader):
        img = np.vstack(viz(batch))

        # 🔸 REMOVE or COMMENT OUT display lines:
        # cv2.imshow('debug', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

        # 🔸 ADD: save image to disk
        out_path = save_dir / f'frame_{i:04d}.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        i += 1


if __name__ == '__main__':
    main()
