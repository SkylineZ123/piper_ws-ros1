#!/usr/bin/env python3
"""LeRobot v3 replay/quality preview, configured by config/replay.yaml."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import yaml

from pika_piper.configuration import package_file, control_environment
from pika_piper.core import validate_metadata, read_trajectory
from pika_piper.replay_tools import latest_dataset, select_episodes, episode_arrays, quality_report, replay_span
from pika_piper.replay_render import write_csv, plot_tracking, render_video, write_index

DEFAULT_OPTIONS = dict(dataset='latest', search_dir='../../../data', repo_id=None,
                       episodes='all', segment_id=None, start_frame=0, end_frame=None,
                       speed=1., output_dir=None, save_video=True, save_plots=True,
                       display=False, camera_names=None, video_max_width=1280,
                       execute=False, pipeline_config='pipeline.yaml', control_python=None)


class ReplaySetupError(RuntimeError):
    """Actionable configuration/dependency failure before robot control."""


def check_control_environment(python):
    """Import the worker's dependencies without starting a ROS node or moving."""
    executable = shutil.which(str(python))
    if executable is None:
        raise ReplaySetupError('Control Python is not executable: %s; set control_python in replay.yaml' % python)
    source = str(Path(__file__).resolve().parents[1] / 'src')
    probe = (
        'import sys,json; sys.path.insert(0,' + repr(source) + '); '
        'import rospy,casadi,pinocchio; '
        'from pinocchio import casadi as cpin; '
        'from pika_piper.kinematics import Kinematics; '
        'from pika_piper.configuration import package_file; '
        'urdf=package_file("urdf/piper_description.urdf",package="piper_description"); '
        'srdf=package_file("config/piper_description.srdf",package="piper_with_gripper_moveit"); '
        'print(json.dumps(dict(python=sys.executable,casadi=casadi.__version__,'
        'pinocchio=pinocchio.__version__,urdf=str(urdf),srdf=str(srdf))))'
    )
    try:
        result = subprocess.run([executable, '-c', probe], env=control_environment(executable),
                                text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReplaySetupError('Cannot check control_python=%s: %s' % (executable, exc)) from exc
    if result.returncode:
        lines = (result.stderr or result.stdout).strip().splitlines()
        cause = next((line for line in reversed(lines)
                      if any(name in line for name in ('Error:', 'Exception:', 'ResourceNotFound:'))),
                     lines[-1] if lines else 'dependency import failed')
        raise ReplaySetupError(
            'Control environment is incompatible: %s\n%s\n'
            'The physical worker requires casadi AND pinocchio.casadi. '
            'Its URDF and SRDF must also be available in the ROS/source workspace. '
            'Set control_python in replay.yaml to the validated controller environment '
            '(or use --control-python). No robot command was sent.' % (executable, cause))
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ReplaySetupError('Unexpected control environment probe output: ' + result.stdout) from exc


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
    parser.add_argument('--control-python', help='Optional physical worker interpreter; default is sys.executable')
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
    for key, value in (('dataset', args.dataset), ('output_dir', args.output),
                       ('pipeline_config', args.pipeline_config)):
        if value is not None:
            options[key] = value if key == 'dataset' and value == 'latest' else str(resolve_path(value, Path.cwd()))
    for key in ('speed', 'execute', 'display', 'control_python', 'segment_id'):
        value = getattr(args, key)
        if value is not None:
            options[key] = (str(resolve_path(value, Path.cwd()))
                            if key == 'control_python' and '/' in value else value)
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
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not np.isfinite(speed) or speed <= 0:
        raise ValueError('speed must be finite and positive')
    if options['execute'] and speed > 1:
        raise ValueError('Physical replay speed must be in (0, 1]')
    for key in ('start_frame', 'end_frame', 'segment_id'):
        value = options[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(key + ' must be a nonnegative integer or null')
    if options['start_frame'] is None:
        raise ValueError('start_frame must be a nonnegative integer')
    if options['end_frame'] is not None and options['end_frame'] <= options['start_frame']:
        raise ValueError('end_frame must be greater than start_frame (exclusive)')
    width = options['video_max_width']
    if isinstance(width, bool) or not isinstance(width, int) or width < 320:
        raise ValueError('video_max_width must be an integer >= 320')
    names = options['camera_names']
    if names is not None and (not isinstance(names, list) or not names or
                              any(not isinstance(n, str) for n in names) or len(set(names)) != len(names)):
        raise ValueError('camera_names must be null or a nonempty list of unique names')
    return options


def run(options):
    root = latest_dataset(options['search_dir']) if options['dataset'] == 'latest' else Path(options['dataset'])
    info = json.loads((root / 'meta/info.json').read_text())
    meta = json.loads((root / 'meta/pika_pipeline.json').read_text())
    validate_metadata(meta)
    episodes = select_episodes(options['episodes'], info['total_episodes'])
    if not episodes:
        raise ValueError('Dataset has no saved episodes')
    if options['execute'] and len(episodes) != 1:
        raise ValueError('Physical replay requires exactly one episode; set episodes: 0 (or another index)')
    if options['execute'] and not Path(options['pipeline_config']).is_file():
        raise ValueError('Physical replay requires a valid pipeline_config')
    control_runtime = None
    if options['execute']:
        # Fail before video export, ROS initialization, or acquiring control.
        control_runtime = check_control_environment(options['control_python'] or sys.executable)
        print('Validated control Python: ' + control_runtime['python'], flush=True)
    output = Path(options['output_dir']) if options['output_dir'] else root / 'replay_preview'
    output.mkdir(parents=True, exist_ok=True)
    print('Dataset: %s\nEpisodes: %s\nOutput: %s\nMode: %s' % (
        root, episodes, output, 'PHYSICAL EXECUTION' if options['execute'] else 'offline preview'), flush=True)
    reports = []
    execution = None
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    with tempfile.TemporaryDirectory(prefix='pika-replay-cache-') as temporary:
        os.environ['HF_DATASETS_CACHE'] = temporary
        os.environ.setdefault('MPLCONFIGDIR', str(Path(temporary) / 'matplotlib'))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        repo_id = options['repo_id'] or meta.get('config', {}).get('repo_id', 'local/pika_piper')
        ds = LeRobotDataset(repo_id=repo_id, root=root, episodes=episodes,
                           revision=info['codebase_version'], video_backend='pyav')
        # Numeric data through LeRobot's public table; videos are decoded
        # sequentially with its metadata offsets instead of seeking every row.
        table = ds.hf_dataset.with_format(None).to_dict()
        subtasks = meta.get('collection_contract', {}).get('subtasks', {})
        subtask_names = dict(enumerate(subtasks.values()))
        if (root / 'meta/subtasks.parquet').is_file():
            import pandas as pd
            subtable = pd.read_parquet(root / 'meta/subtasks.parquet')
            subtask_names = {int(index): str(name) for name, index in subtable['subtask_index'].items()}
        for episode in episodes:
            print('Processing episode %d ...' % episode, flush=True)
            arrays = episode_arrays(table, episode, options['start_frame'], options['end_frame'], options['segment_id'])
            if meta['action_type'] == 'eef6d' and 'observation.eef_state' not in arrays:
                raise ValueError('EEF replay requires observation.eef_state')
            if options['execute']:
                execution = replay_span(arrays, ds.fps)
            prefix = 'episode_%03d' % episode
            report = quality_report(arrays, meta, ds.fps)
            report.update(episode=episode, first_frame=int(arrays['frame_index'][0]),
                          last_frame=int(arrays['frame_index'][-1]), playback_speed=options['speed'])
            artifacts = dict(csv=prefix + '.csv', json=prefix + '_quality.json', video=None, plots=[])
            write_csv(output / artifacts['csv'], arrays, meta['action_type'])
            if options['save_plots']:
                pairs = [('joint_command', 'observation.state', False),
                         ('eef_command', 'observation.eef_state', True)]
                for command, state, eef in pairs:
                    if command in arrays and state in arrays:
                        name = prefix + ('_eef.png' if eef else '_joint.png')
                        plot_tracking(output / name, arrays, command, state, ds.fps, eef)
                        artifacts['plots'].append(name)
            if options['save_video'] or options['display']:
                name = prefix + '.mp4'
                report['cameras'] = render_video(ds, episode, arrays, options, output / name,
                                                meta.get('config', {}).get('task', ''), subtask_names,
                                                meta['action_type'])
                if options['save_video']:
                    artifacts['video'] = name
            else:
                report['cameras'] = None  # video decoding was not requested
            report['artifacts'] = artifacts
            (output / artifacts['json']).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            reports.append(report)
            print('Episode %d: %d frames, %d continuous spans' % (
                episode, report['frames'], len(report['continuous_spans'])), flush=True)
    summary = dict(dataset=str(root), fps=info['fps'], action_type=meta['action_type'],
                   units=meta['units'], urdf_sha256=meta['urdf_sha256'],
                   episodes=reports, options=options,
                   control_runtime=control_runtime,
                   tracking_note='Synchronous target-feedback errors include robot response lag; '
                                 'they do not alone determine dataset quality.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_index(output, root, reports)
    print('Preview report: ' + str(output / 'index.html'), flush=True)
    if options['execute']:
        actions, stamps = execution
        path = output / ('episode_%03d_trajectory.npz' % episodes[0])
        np.savez(path, actions=actions, timestamps=stamps)
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
        read_trajectory(path)
        worker = Path(__file__).with_name('replay_worker.py')
        python = control_runtime['python']
        try:
            subprocess.run([python, str(worker), '--trajectory', str(path),
                            '--config', options['pipeline_config'], '--speed', str(options['speed'])],
                           env=control_environment(python), check=True)
        except subprocess.CalledProcessError as exc:
            raise ReplaySetupError('Physical worker exited with code %d; see its diagnostic above.' % exc.returncode) from exc
    return summary


def main(argv=None):
    return run(load_options(argv))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Replay cancelled.', file=sys.stderr)
        sys.exit(130)
    except ReplaySetupError as exc:
        print('Replay error: ' + str(exc), file=sys.stderr)
        sys.exit(2)
