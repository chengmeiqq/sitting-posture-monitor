# -*- coding: utf-8 -*-
"""显示输出 (阶段7): 通过管道把标注帧喂给 ffplay 在 X11(:0) 显示.
避免安装桌面版 opencv (与 headless 版冲突), 零新依赖。
"""
import subprocess
import numpy as np


class PipeDisplay:
    def __init__(self, width, height, fps=15):
        self.w, self.h, self.fps = width, height, fps
        self.proc = subprocess.Popen(
            ['ffplay', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'bgr24',
             '-video_size', '%dx%d' % (width, height), '-framerate', str(fps), '-i', 'pipe:0'],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def put(self, frame_bgr):
        f = frame_bgr
        if f.shape[1] != self.w or f.shape[0] != self.h:
            import cv2
            f = cv2.resize(f, (self.w, self.h))
        try:
            self.proc.stdin.write(f.tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError('display window closed')

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
