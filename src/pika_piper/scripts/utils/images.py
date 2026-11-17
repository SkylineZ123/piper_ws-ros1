"""Decode ROS Image bytes to RGB without cv_bridge or native ROS extensions."""

import numpy as np


def decode_image(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError('Expected rgb8/bgr8 Image, got ' + msg.encoding)
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    return image[:, :, ::-1].copy() if msg.encoding == 'bgr8' else image.copy()
