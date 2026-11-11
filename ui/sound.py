# -*- coding: utf-8 -*-
"""声音提示 (本地 ALSA, 无新依赖)

实现选择(踩过两个坑后):
1. 每次合成波形再起 aplay: 可行, 启动开销约 150ms
2. 常驻 aplay 持续喂静音流想做到零延迟: 在这块板子的 ES8323 + I2S 驱动上
   aplay 会卡在 PREPARED 不触发(缓冲区写满也不 start), 不可靠 -> 放弃

所以用方案 1 的改良版: 启动时把提示音预渲染成 wav 文件放内存盘,
提醒时用后台线程起 aplay 播放, 不阻塞主监控循环。

策略:
- 只对"已确认的异常姿态"响; NORMAL/UNKNOWN/AWAY 不响(未校准阶段不乱叫)
- 升级立即响; 持续同一异常按 repeat_s 重复, 不连响
- 红(急促三连) > 黄(温和双音)
- 任何异常都降级为静音, 绝不影响监控
"""
import math
import os
import struct
import subprocess
import tempfile
import threading
import time
import wave

SR = 48000
CH = 2
FADE = 0.008


def _render(freqs, vol):
    """freqs: [(频率Hz, 时长s), ...]; 频率 0 表示静音。返回 int16 帧数据。"""
    frames = bytearray()
    for freq, dur in freqs:
        n = int(SR * dur)
        if freq <= 0:
            frames += b'\x00\x00\x00\x00' * n
            continue
        for i in range(n):
            env = min(1.0, i / (SR * FADE), (n - i) / (SR * FADE))
            v = int(32767 * vol * env * math.sin(2 * math.pi * freq * i / SR))
            frames += struct.pack('<hh', v, v)
    return bytes(frames)


def _write_wav(path, data):
    w = wave.open(path, 'wb')
    w.setnchannels(CH)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(data)
    w.close()


class AlertSound:
    def __init__(self, device='hw:0,0', enabled=True, volume=0.35,
                 repeat_s=30.0, min_gap_s=1.5):
        self.device = device
        self.enabled = enabled
        self.volume = volume
        self.repeat_s = repeat_s
        self.min_gap_s = min_gap_s
        self._lock = threading.Lock()
        self._last_beep = 0.0
        self._last_level = None
        self._alert_since = 0.0
        self._ok = True
        self._err = None
        self._playing = None
        self._play_lock = threading.Lock()

        self._dir = tempfile.mkdtemp(prefix='posture_snd_')
        self._files = {}
        specs = {
            # 黄色: 温和双音 (提示)
            'yellow': [(660, 0.10), (0, 0.05), (660, 0.10), (0, 0.05)],
            # 红色: 急促三连 (异常)
            'red': [(1050, 0.09), (0, 0.04), (1050, 0.09), (0, 0.04),
                    (1050, 0.09), (0, 0.05)],
            # 恢复正常: 单音
            'ok': [(880, 0.09), (0, 0.04)],
        }
        for name, spec in specs.items():
            path = os.path.join(self._dir, name + '.wav')
            try:
                _write_wav(path, _render(spec, volume))
                self._files[name] = path
            except Exception as e:
                self._ok = False
                self._err = 'render %s: %s' % (name, e)

    def _play_file(self, path):
        """后台线程里起 aplay, 不阻塞调用方。
        设备是独占的: 上一次 aplay 没退干净会 Device busy, 所以串行化并等待前一个结束。
        """
        with self._play_lock:
            prev = self._playing
            if prev is not None and prev.poll() is None:
                try:
                    prev.wait(timeout=2.0)
                except Exception:
                    try:
                        prev.kill()
                        prev.wait(timeout=1.0)
                    except Exception:
                        pass
            try:
                p = subprocess.Popen(
                    ['aplay', '-D', self.device, '-q', path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._playing = p
                self._ok = True
                self._err = None
                p.wait(timeout=8.0)      # 让设备彻底放开
            except Exception as e:
                self._ok = False
                self._err = str(e)

    def beep(self, kind='yellow'):
        if not self.enabled:
            return False
        path = self._files.get(kind) or self._files.get('yellow')
        if not path:
            return False
        now = time.time()
        with self._lock:
            if now - self._last_beep < self.min_gap_s:
                return False
            self._last_beep = now
        threading.Thread(target=self._play_file, args=(path,), daemon=True).start()
        return True

    def update(self, level, state):
        """每个判定周期调用, 返回本周期是否触发声音。"""
        if not self.enabled:
            return False
        if level == 'green' or state in ('UNKNOWN', 'AWAY', 'INIT'):
            self._last_level = level
            self._alert_since = 0.0
            return False

        now = time.time()
        kind = 'red' if level == 'red' else 'yellow'
        escalated = (self._last_level != level)
        self._last_level = level

        if escalated or self._alert_since == 0.0:
            self._alert_since = now
            return self.beep(kind)
        if now - self._alert_since >= self.repeat_s:
            self._alert_since = now
            return self.beep(kind)
        return False

    def set_enabled(self, on):
        with self._lock:
            self.enabled = bool(on)
            if not self.enabled:
                self._alert_since = 0.0
                self._last_level = None

    def status(self):
        return {'enabled': self.enabled, 'ok': self._ok, 'error': self._err,
                'device': self.device, 'volume': self.volume,
                'repeat_s': self.repeat_s}

    def close(self):
        p = self._playing
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
        for path in self._files.values():
            try:
                os.unlink(path)
            except Exception:
                pass
        try:
            os.rmdir(self._dir)
        except Exception:
            pass
