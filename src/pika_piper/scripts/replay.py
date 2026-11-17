#!/usr/bin/env python3
"""LeRobot v3 replay and quality preview, configured by config/replay.yaml."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from utils.configuration import control_environment
from utils.core import read_trajectory, validate_metadata
from utils.replay_config import DEFAULT_OPTIONS as DEFAULT_OPTIONS
from utils.replay_config import load_options
from utils.replay_render import plot_tracking, render_video, write_csv, write_index
from utils.replay_tools import (
    episode_arrays,
    latest_dataset,
    quality_report,
    replay_span,
    select_episodes,
)
from utils.reproducibility import runtime_snapshot
from utils.runtime import ReplaySetupError, check_control_environment


def run(options):
    root = (
        latest_dataset(options['search_dir'])
        if options['dataset'] == 'latest'
        else Path(options['dataset'])
    )
    info = json.loads((root / 'meta/info.json').read_text())
    meta = json.loads((root / 'meta/pika_pipeline.json').read_text())
    validate_metadata(meta)
    episodes = select_episodes(options['episodes'], info['total_episodes'])
    if not episodes:
        raise ValueError('Dataset has no saved episodes')
    if options['execute'] and len(episodes) != 1:
        raise ValueError(
            'Physical replay requires exactly one episode; set episodes: 0 (or another index)'
        )
    if options['execute'] and not Path(options['pipeline_config']).is_file():
        raise ValueError('Physical replay requires a valid pipeline_config')
    control_runtime = None
    if options['execute']:
        # Fail before video export, ROS initialization, or acquiring control.
        control_runtime = check_control_environment(options['control_python'] or sys.executable)
        print('Validated control Python: ' + control_runtime['python'], flush=True)
    output = Path(options['output_dir']) if options['output_dir'] else root / 'replay_preview'
    output.mkdir(parents=True, exist_ok=True)
    print(
        'Dataset: %s\nEpisodes: %s\nOutput: %s\nMode: %s'
        % (
            root,
            episodes,
            output,
            'PHYSICAL EXECUTION' if options['execute'] else 'offline preview',
        ),
        flush=True,
    )
    reports = []
    execution = None
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    with tempfile.TemporaryDirectory(prefix='pika-replay-cache-') as temporary:
        os.environ['HF_DATASETS_CACHE'] = temporary
        os.environ.setdefault('MPLCONFIGDIR', str(Path(temporary) / 'matplotlib'))
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id = options['repo_id'] or meta.get('config', {}).get('repo_id', 'local/pika_piper')
        ds = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            revision=info['codebase_version'],
            video_backend='pyav',
        )
        # Numeric data through LeRobot's public table; videos are decoded
        # sequentially with its metadata offsets instead of seeking every row.
        table = ds.hf_dataset.with_format(None).to_dict()
        subtasks = meta.get('collection_contract', {}).get('subtasks', {})
        subtask_names = dict(enumerate(subtasks.values()))
        if (root / 'meta/subtasks.parquet').is_file():
            import pandas as pd

            subtable = pd.read_parquet(root / 'meta/subtasks.parquet')
            subtask_names = {
                int(index): str(name) for name, index in subtable['subtask_index'].items()
            }
        for episode in episodes:
            print('Processing episode %d ...' % episode, flush=True)
            arrays = episode_arrays(
                table, episode, options['start_frame'], options['end_frame'], options['segment_id']
            )
            if meta['action_type'] == 'eef6d' and 'observation.eef_state' not in arrays:
                raise ValueError('EEF replay requires observation.eef_state')
            if options['execute']:
                execution = replay_span(arrays, ds.fps)
            prefix = 'episode_%03d' % episode
            report = quality_report(arrays, meta, ds.fps)
            report.update(
                episode=episode,
                first_frame=int(arrays['frame_index'][0]),
                last_frame=int(arrays['frame_index'][-1]),
                playback_speed=options['speed'],
            )
            artifacts = dict(
                csv=prefix + '.csv', json=prefix + '_quality.json', video=None, plots=[]
            )
            write_csv(output / artifacts['csv'], arrays, meta['action_type'])
            if options['save_plots']:
                pairs = [
                    ('joint_command', 'observation.state', False),
                    ('eef_command', 'observation.eef_state', True),
                ]
                for command, state, eef in pairs:
                    if command in arrays and state in arrays:
                        name = prefix + ('_eef.png' if eef else '_joint.png')
                        plot_tracking(output / name, arrays, command, state, ds.fps, eef)
                        artifacts['plots'].append(name)
            if options['save_video'] or options['display']:
                name = prefix + '.mp4'
                report['cameras'] = render_video(
                    ds,
                    episode,
                    arrays,
                    options,
                    output / name,
                    meta.get('config', {}).get('task', ''),
                    subtask_names,
                    meta['action_type'],
                )
                if options['save_video']:
                    artifacts['video'] = name
            else:
                report['cameras'] = None  # video decoding was not requested
            report['artifacts'] = artifacts
            (output / artifacts['json']).write_text(
                json.dumps(report, indent=2, allow_nan=False) + '\n'
            )
            reports.append(report)
            print(
                'Episode %d: %d frames, %d continuous spans'
                % (episode, report['frames'], len(report['continuous_spans'])),
                flush=True,
            )
    summary = dict(
        dataset=str(root),
        fps=info['fps'],
        action_type=meta['action_type'],
        units=meta['units'],
        urdf_sha256=meta['urdf_sha256'],
        episodes=reports,
        options=options,
        control_runtime=control_runtime,
        runtime=runtime_snapshot(),
        tracking_note='Synchronous target-feedback errors include robot response lag; '
        'they do not alone determine dataset quality.',
    )
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
            subprocess.run(
                [
                    python,
                    str(worker),
                    '--trajectory',
                    str(path),
                    '--config',
                    options['pipeline_config'],
                    '--speed',
                    str(options['speed']),
                ],
                env=control_environment(python),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ReplaySetupError(
                'Physical worker exited with code %d; see its diagnostic above.' % exc.returncode
            ) from exc
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
