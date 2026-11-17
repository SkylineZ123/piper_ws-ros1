"""Headless plots/video and an optional main-thread OpenCV playback window."""
import csv
from fractions import Fraction
import html
import json
import os
from pathlib import Path
import time

import numpy as np

from .collection import JOINT_NAMES, EEF_NAMES
from .replay_tools import CameraFrames, continuous_spans


def write_csv(path, arrays, action_type='joint'):
    keys = list(arrays)
    columns = []
    for key in keys:
        value = arrays[key]
        names = EEF_NAMES if (key in ('observation.eef_state', 'eef_command') or
                              (key == 'action' and action_type == 'eef6d')) else JOINT_NAMES
        columns.extend([key + '.' + name for name in names] if value.ndim == 2 else [key])
    with Path(path).open('w', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        for i in range(len(arrays['action'])):
            row = []
            for key in keys:
                row.extend(np.atleast_1d(arrays[key][i]).tolist())
            writer.writerow(row)


def plot_tracking(path, arrays, command_key, state_key, fps, eef=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = EEF_NAMES if eef else JOINT_NAMES
    spans = continuous_spans(arrays, fps)
    times = arrays['timestamp']
    fig, axes = plt.subplots(7, 1, figsize=(14, 15), sharex=True)
    try:
        for dim, axis in enumerate(axes):
            for a, b in spans:
                for key, color, label in ((state_key, 'tab:blue', 'state'),
                                          (command_key, 'tab:orange', 'action / command')):
                    values = arrays[key][a:b, dim].astype(float).copy()
                    if eef and 3 <= dim < 6:
                        values[np.flatnonzero(abs(np.diff(values)) > np.pi) + 1] = np.nan
                    axis.plot(times[a:b], values, color=color, label=label if a == 0 else None,
                              linewidth=.85, drawstyle='steps-post' if key == command_key else 'default')
                if a:
                    axis.axvline(times[a], color='grey', linestyle='--', linewidth=.8)
            back = arrays.get('back_event', np.zeros(len(times)))
            for i in np.flatnonzero(back):
                axis.axvspan(times[i], times[i] + 1/fps, color='red', alpha=.12, linewidth=0)
            unit = 'm' if dim == 6 or (eef and dim < 3) else 'rad'
            axis.set_ylabel('%s\n(%s)' % (names[dim], unit))
            axis.grid(alpha=.2)
        axes[0].legend(loc='upper right')
        axes[0].set_title('Episode tracking: %s vs %s; dashed = span boundary, red = back_event%s' % (
            command_key, state_key, '; RPY lines break at wrap' if eef else ''))
        axes[-1].set_xlabel('LeRobot episode timestamp (s; pauses are compressed)')
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)


def canvas(images, names, arrays, i, episode, task, subtask_names, max_width, action_type):
    import cv2
    height = min(image.shape[0] for image in images)
    tiles = []
    for image, name in zip(images, names):
        tile = cv2.resize(image[:, :, ::-1], (int(image.shape[1]*height/image.shape[0]), height))
        cv2.putText(tile, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 240, 30), 2)
        tiles.append(tile)
    strip = np.concatenate(tiles, axis=1)
    if strip.shape[1] > max_width:
        strip = cv2.resize(strip, (max_width, int(strip.shape[0]*max_width/strip.shape[1])))
    panel = np.zeros((156, strip.shape[1], 3), np.uint8)

    def text(value, y, color=(235, 235, 235)):
        value = str(value).encode('ascii', errors='replace').decode('ascii')
        scale = .55
        while scale > .25 and cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > panel.shape[1]-20:
            scale -= .025
        cv2.putText(panel, value, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    frame = int(arrays['frame_index'][i])
    stamp = float(arrays['timestamp'][i])
    segment = int(arrays['segment_id'][i]) if 'segment_id' in arrays else 0
    back = int(arrays['back_event'][i]) if 'back_event' in arrays else 0
    subtask = int(arrays['subtask_index'][i]) if 'subtask_index' in arrays else -1
    skill = subtask_names.get(subtask, str(subtask) if subtask >= 0 else '')
    text('episode %d | frame %d | t %.3fs | segment %d | BACK=%d' % (
        episode, frame, stamp, segment, back), 25, (50, 70, 240) if back else (60, 230, 60))
    source = arrays.get('source_timestamp', arrays['timestamp'])
    text('TASK: %s | SUBTASK: %s | source elapsed %.3fs' % (task, skill, source[i]-source[0]), 54)
    text('ACTION (%s): ' % action_type + ' '.join('%+.4f' % v for v in arrays['action'][i]), 88, (0, 180, 255))
    state_key = 'observation.eef_state' if action_type == 'eef6d' else 'observation.state'
    text('STATE: ' + ' '.join('%+.4f' % v for v in arrays[state_key][i]), 116, (255, 180, 80))
    text('Recorded data preview | SPACE pause, q/ESC cancel (window only)', 143, (150, 150, 150))
    image = np.concatenate([strip, panel], axis=0)
    return cv2.copyMakeBorder(image, 0, image.shape[0] % 2, 0, image.shape[1] % 2,
                              cv2.BORDER_CONSTANT, value=0)


def render_video(ds, episode, arrays, options, output, task, subtask_names, action_type='joint'):
    import av
    import cv2
    keys = [key for key, feature in ds.features.items() if key.startswith('observation.images.')
            and feature['dtype'] == 'video']
    requested = options['camera_names']
    if requested is not None:
        keys = ['observation.images.' + name for name in requested]
        if any(key not in ds.features or ds.features[key]['dtype'] != 'video' for key in keys):
            raise ValueError('camera_names contains a missing or non-video camera')
    if not keys:
        raise ValueError('No RGB video cameras in this dataset')
    names = [key.removeprefix('observation.images.') for key in keys]
    readers, offsets = [], []
    container = stream = None
    path = Path(output)
    temporary = path.with_suffix('.tmp.mp4')
    show = options['display']
    if show and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise ValueError('display=true requires a graphical display; use display=false for export')
    previous = [None] * len(keys)
    identical = [0] * len(keys)
    completed = False
    deadline = time.monotonic()
    try:
        ep_meta = next(row for row in ds.meta.episodes if int(row['episode_index']) == episode)
        for key in keys:
            offset = float(ep_meta['videos/' + key + '/from_timestamp'])
            offsets.append(offset)
            video_path = Path(ds.root) / ds.meta.get_video_file_path(episode, key)
            readers.append(CameraFrames(video_path, offset + float(arrays['timestamp'][0]), ds.fps))
        if show:
            cv2.namedWindow('Pika dataset replay', cv2.WINDOW_NORMAL)
        for i in range(len(arrays['action'])):
            images = [reader.frame_at(offset + float(arrays['timestamp'][i]))
                      for reader, offset in zip(readers, offsets)]
            for j, image in enumerate(images):
                if previous[j] is not None and np.array_equal(image, previous[j]):
                    identical[j] += 1
                previous[j] = image
            image = canvas(images, names, arrays, i, episode, task, subtask_names, options['video_max_width'], action_type)
            if options['save_video']:
                if container is None:
                    container = av.open(str(temporary), mode='w')
                    stream = container.add_stream('libx264', rate=Fraction(str(ds.fps*options['speed'])).limit_denominator(10000))
                    stream.width, stream.height = image.shape[1], image.shape[0]
                    stream.pix_fmt = 'yuv420p'
                    stream.options = {'crf': '23', 'preset': 'veryfast', 'threads': '2'}
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='bgr24')):
                    container.mux(packet)
            if show:
                cv2.imshow('Pika dataset replay', image)
                deadline += 1/(ds.fps*options['speed'])
                key = cv2.waitKey(max(1, int(1000*(deadline-time.monotonic())))) & 255
                if key == ord(' '):
                    while True:
                        key = cv2.waitKey(30) & 255
                        if key in (ord(' '), ord('q'), 27):
                            break
                    deadline = time.monotonic()
                if key in (ord('q'), 27) or cv2.getWindowProperty('Pika dataset replay', cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt('Preview cancelled')
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
            temporary.replace(path)
        completed = True
        return {name: dict(matched_frames=reader.matched_frames,
                           identical_consecutive_frames=identical[j])
                for j, (name, reader) in enumerate(zip(names, readers))}
    finally:
        for reader in readers:
            reader.close()
        if container is not None:
            container.close()
        if not completed and temporary.exists():
            temporary.unlink()
        if show:
            try:
                cv2.destroyWindow('Pika dataset replay')
            except cv2.error:
                pass  # The operator may already have closed the window.


def write_index(output, dataset, reports):
    blocks = []
    for report in reports:
        artifacts = report['artifacts']
        links = ['<a href="%s">%s</a>' % (html.escape(name, quote=True), html.escape(key))
                 for key, name in artifacts.items() if key not in ('video', 'plots')]
        video = ('<video controls preload="metadata" src="%s"></video>' % html.escape(artifacts['video'], quote=True)
                 if artifacts.get('video') else '')
        plots = ''.join('<img loading="lazy" src="%s">' % html.escape(name, quote=True)
                        for name in artifacts['plots'])
        tracking = report.get('joint_tracking')
        error = (' | joint RMSE max %.4f rad | gripper RMSE %.2f mm' % (
            max(tracking['rmse'][:6]), tracking['rmse'][6]*1000)) if tracking else ''
        blocks.append('<section><h2>Episode %d</h2><p>%d frames | %.2f s | %d spans | back_event %d%s</p>%s<p>%s</p>%s</section>' % (
            report['episode'], report['frames'], report['duration_s'], len(report['continuous_spans']),
            report['back_event_frames'], error, video, ' · '.join(links), plots))
    page = '<!doctype html><meta charset="utf-8"><title>Pika LeRobot replay</title><style>body{max-width:1280px;margin:24px auto;font-family:sans-serif}video,img{width:100%%}section{margin-bottom:40px}p{line-height:1.6}</style><h1>数据集回放与质量检查</h1><p>%s</p><p>蓝线：真实 state；橙线：实际 command。跟踪误差包含机器人响应延迟，不能单独作为数据质量合格判据。视频中的 ACTION/STATE 遵循数据集的 action_type：joint 为关节角(rad)+夹爪(m)，eef6d 为位置(m)+RPY(rad)+夹爪(m)。</p>%s' % (html.escape(str(dataset)), ''.join(blocks))
    (Path(output) / 'index.html').write_text(page, encoding='utf-8')
