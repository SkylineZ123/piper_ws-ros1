"""Replay options and validation; relative YAML paths use the YAML directory."""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

from .configuration import package_file

DEFAULT_OPTIONS = dict(
    dataset='latest',
    search_dir='../../../data',
    repo_id=None,
    episodes='all',
    segment_id=None,
    start_frame=0,
    end_frame=None,
    speed=1.0,
    output_dir=None,
    save_video=True,
    save_plots=True,
    display=False,
    camera_names=None,
    video_max_width=1280,
    execute=False,
    pipeline_config='pipeline.yaml',
    control_python=None,
)


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Replay YAML; defaults to this package config/replay.yaml')
    parser.add_argument('--dataset', help='Dataset directory, or latest')
    parser.add_argument('--episode', type=int, help='Override YAML episodes with one episode')
    parser.add_argument('--segment-id', type=int)
    parser.add_argument('--output', help='Override preview output directory')
    parser.add_argument('--speed', type=float)
    parser.add_argument('--pipeline-config', help='Robot/controller YAML for physical execution')
    parser.add_argument(
        '--control-python', help='Optional physical worker interpreter; default is sys.executable'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', dest='execute', default=None)
    mode.add_argument('--preview-only', action='store_false', dest='execute')
    parser.add_argument('--display', action='store_true', default=None)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    default_path = package_file('config/replay.yaml')
    defaults = DEFAULT_OPTIONS
    path = resolve_path(args.config, Path.cwd()) if args.config else default_path
    supplied = yaml.safe_load(path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError('Replay YAML must be a mapping')
    # Preserve the old --dataset ... --config pipeline.yaml --execute syntax.
    legacy_pipeline = 'arm' in supplied and 'cameras' in supplied
    if legacy_pipeline:
        options, base = dict(defaults), default_path.parent
        options['pipeline_config'] = str(path)
    else:
        unknown = set(supplied) - set(defaults)
        if unknown:
            raise ValueError('Unknown replay settings: ' + ', '.join(sorted(unknown)))
        options, base = dict(defaults, **supplied), path.parent
    for key in ('search_dir', 'output_dir', 'pipeline_config'):
        if options[key] is not None:
            options[key] = str(resolve_path(options[key], base))
    if options['control_python'] is not None:
        if not isinstance(options['control_python'], str) or not options['control_python'].strip():
            raise ValueError('control_python must be null or an interpreter path')
        if '/' in options['control_python']:
            options['control_python'] = str(resolve_path(options['control_python'], base))
    dataset = options['dataset']
    options['dataset'] = str(resolve_path(dataset, base)) if dataset != 'latest' else dataset
    for key, value in (
        ('dataset', args.dataset),
        ('output_dir', args.output),
        ('pipeline_config', args.pipeline_config),
    ):
        if value is not None:
            options[key] = (
                value
                if key == 'dataset' and value == 'latest'
                else str(resolve_path(value, Path.cwd()))
            )
    for key in ('speed', 'execute', 'display', 'control_python', 'segment_id'):
        value = getattr(args, key)
        if value is not None:
            options[key] = (
                str(resolve_path(value, Path.cwd()))
                if key == 'control_python' and '/' in value
                else value
            )
    if args.episode is not None:
        options['episodes'] = args.episode
    if args.no_video:
        options['save_video'] = False
    if args.no_plots:
        options['save_plots'] = False
    for key in ('execute', 'display', 'save_video', 'save_plots'):
        if not isinstance(options[key], bool):
            raise ValueError(key + ' must be a YAML boolean')
    speed = options['speed']
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not np.isfinite(speed)
        or speed <= 0
    ):
        raise ValueError('speed must be finite and positive')
    for key in ('start_frame', 'end_frame', 'segment_id'):
        value = options[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(key + ' must be a nonnegative integer or null')
    if options['start_frame'] is None:
        raise ValueError('start_frame must be a nonnegative integer')
    if options['end_frame'] is not None and options['end_frame'] <= options['start_frame']:
        raise ValueError('end_frame must be greater than start_frame (exclusive)')
    width = options['video_max_width']
    if isinstance(width, bool) or not isinstance(width, int) or width < 320:
        raise ValueError('video_max_width must be an integer >= 320')
    names = options['camera_names']
    if names is not None and (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError('camera_names must be null or a nonempty list of unique names')
    if options['control_python'] is None:
        options['control_python'] = os.environ.get('PIKA_CONTROL_PYTHON') or None
    return options
