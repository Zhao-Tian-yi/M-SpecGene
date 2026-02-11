# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--rgbt-mode',
        choices=['rgbt', 'rgb', 'ir'],
        default='rgbt',
        help='Input mode for RGBT pipelines.')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _match_channel_values(values, channels):
    if values is None:
        return values
    if isinstance(values, (int, float)):
        return values
    if len(values) == channels:
        return values
    if len(values) == 3 and channels == 6:
        return list(values) * 2
    if len(values) == 6 and channels == 3:
        return list(values)[:3]
    if len(values) == 4 and channels == 6:
        return list(values[:3]) + [values[3]] * 3
    if len(values) == 4 and channels == 3:
        return list(values)[:3]
    return values


def _update_pipeline_rgbt_mode(pipeline, mode):
    channels = 6 if mode == 'rgbt' else 3
    for transform in pipeline:
        if transform.get('type') == 'LoadRGBTImageFromFile':
            transform['mode'] = mode
        if transform.get('type') == 'Pad' and 'pad_val' in transform:
            transform['pad_val'] = _match_channel_values(
                transform['pad_val'], channels)


def _update_rgbt_mode(cfg, mode):
    channels = 6 if mode == 'rgbt' else 3
    for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        if key not in cfg:
            continue
        data_cfg = cfg[key].get('dataset')
        while isinstance(data_cfg, dict) and 'dataset' in data_cfg:
            data_cfg = data_cfg['dataset']
        if isinstance(data_cfg, dict) and 'pipeline' in data_cfg:
            _update_pipeline_rgbt_mode(data_cfg['pipeline'], mode)

    if 'tta_pipeline' in cfg:
        _update_pipeline_rgbt_mode(cfg.tta_pipeline, mode)

    data_preprocessor = cfg.get('model', {}).get('data_preprocessor', None)
    if isinstance(data_preprocessor, dict):
        data_preprocessor['mean'] = _match_channel_values(
            data_preprocessor.get('mean'), channels)
        data_preprocessor['std'] = _match_channel_values(
            data_preprocessor.get('std'), channels)
        data_preprocessor['pad_val'] = _match_channel_values(
            data_preprocessor.get('pad_val'), channels)


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    _update_rgbt_mode(cfg, args.rgbt_mode)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
