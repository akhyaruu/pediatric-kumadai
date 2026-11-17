"""
Unicode-safe replacements for cv2.imread / cv2.imwrite.

On Windows, OpenCV cannot open paths that contain non-ASCII characters (e.g. "764.1 Schiessscheibenphänomen.bmp").
These helpers read and write the file bytes with NumPy and decode/encode them with OpenCV, which works for any path.
They behave like the OpenCV functions: imread returns None if the file cannot be read or decoded.
"""

import os

import cv2
import numpy as np


def imread(filename, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(filename), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(filename, img, params=None):
    filename = str(filename)
    ext = os.path.splitext(filename)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params if params is not None else [])
    if not ok:
        return False
    buf.tofile(filename)
    return True
