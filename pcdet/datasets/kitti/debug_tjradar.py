from tjradar_dataset import create_tjradar_infos

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[0] == 'create_tjradar_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[1])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_tjradar_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )