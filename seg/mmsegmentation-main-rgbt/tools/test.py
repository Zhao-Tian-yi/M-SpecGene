# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
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
    for key in ('val_dataloader', 'test_dataloader'):
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


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
