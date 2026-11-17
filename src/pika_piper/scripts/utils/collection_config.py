"""Resolve and validate acquisition configuration without starting ROS."""

import datetime
import uuid
from pathlib import Path

import numpy as np
import yaml

from .configuration import load_config, package_file


def load_settings(args):
    pipeline_path = args.config
    collection_path = args.collection_config
    supplied = load_config(Path(pipeline_path).expanduser()) if pipeline_path else None
    if (
        supplied is not None
        and 'arm' not in supplied
        and any(key in supplied for key in ('lerobot_python', 'action_type', 'image_height'))
    ):
        if collection_path:
            raise ValueError(
                '--config already specifies a collection YAML; use '
                '--config pipeline.yaml --collection-config collection.yaml'
            )
        # Accept the common single-config invocation, including IDE launches.
        collection_path, pipeline_path = pipeline_path, None
        supplied = None
    pipeline_path = (
        Path(pipeline_path or package_file('config/pipeline.yaml')).expanduser().resolve()
    )
    collection_path = (
        Path(collection_path or package_file('config/collection.yaml')).expanduser().resolve()
    )
    pipeline = supplied if supplied is not None else load_config(pipeline_path)
    settings = yaml.safe_load(collection_path.read_text())
    if not isinstance(settings, dict):
        raise ValueError('Collection config must be a YAML mapping')
    # Later controller-param merging must use the resolved PIPELINE path.
    args.config, args.collection_config = str(pipeline_path), str(collection_path)
    cfg = dict(pipeline, **settings)
    output_base = Path(cfg['output_dir']).expanduser()
    if not output_base.is_absolute():
        output_base = collection_path.parent / output_base
    cfg['output_dir'] = str(output_base.resolve())
    # Ignore the obsolete setting in older YAMLs; the launching interpreter
    # is the sole authority for both acquisition and encoding.
    cfg.pop('lerobot_python', None)
    for name in ('repo_id', 'action_type', 'task'):
        if getattr(args, name, None) is not None:
            cfg[name] = getattr(args, name)
    if cfg['action_type'] not in ('joint', 'eef6d'):
        raise ValueError('action_type must be joint or eef6d')
    for name in (
        'fps',
        'image_height',
        'image_width',
        'max_timesteps',
        'encoder_threads',
        'encoder_queue_maxsize',
        'view_max_width',
    ):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], int) or cfg[name] <= 0:
            raise ValueError(name + ' must be a positive integer')
    if cfg['image_height'] % 2 or cfg['image_width'] % 2:
        raise ValueError('Video image dimensions must be even')
    for name in (
        'alignment_delay',
        'alignment_timeout',
        'status_timeout',
        'source_timeout',
        'camera_tolerance',
        'state_gap',
        'command_age',
    ):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(name + ' must be finite and positive')
    if not cfg['cameras'] or len(set(cfg['cameras'].values())) != len(cfg['cameras']):
        raise ValueError('Configure at least one camera and distinct camera topics')
    cfg['subtasks'] = {str(k): str(v) for k, v in cfg.get('subtasks', {}).items()}
    if any(len(k) != 1 or k not in '123456789' for k in cfg['subtasks']):
        raise ValueError('Subtask keys must be single digits 1..9')
    if len(set(cfg['subtasks'].values())) != len(cfg['subtasks']):
        raise ValueError('Subtask names must be unique')
    if args.resume and not args.output:
        raise ValueError('--resume requires --output pointing to an existing dataset')
    if args.output:
        root = Path(args.output).expanduser().resolve()
    else:
        name = (
            'dataset_'
            + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            + '_'
            + uuid.uuid4().hex[:8]
        )
        root = (Path(cfg['output_dir']).expanduser() / name).resolve()
    return cfg, root
