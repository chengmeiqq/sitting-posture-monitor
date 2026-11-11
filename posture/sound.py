# -*- coding: utf-8 -*-
"""声音提醒 (板载 es8388 喇叭, aplay 播放, 不依赖外网/云端)

音效策略 (防刷屏):
  - 进入红色姿态异常 >= confirm_s: 急促三连"嘀-嘀-嘀"; 红色持续期间每 repeat_s 重复一次
  - 红色恢复正常: 柔和两声"叮咚"(正反馈, 给小朋友)
  - 进入黄色姿态提醒(歪肩): 单声轻"嘀" (仅一次)
  - 离座/未知: 不发声 (孩子起身是正常的)
音量/开关可由 web /api/sound 控制; 关闭后本模块静默。
"""
import os
import subprocess
import threading
import time
import wave

import numpy as np

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'audio')
RATE = 48000   # es8388 仅支持常见标准采样率(48k/44.1k), 22050 会被拒


def _tone(freq, dur_s, vol=0.55, ramp=0.008):
    n = int(RATE * dur_s)
    t = np.arange(n) / RATE
    y = np.sin(2 * np.pi * freq * t)
    # 淡入淡出防爆音
    r = int(RATE * ramp)
    env = np.ones(n)
    if r * 2 < n:
        env[:r] = np.linspace(0, 1, r)
        env[-r:] = np.linspace(1, 0, r)
    y = y * env * vol
    return (y * 32767).astype(np.int16)


def _seq_write(path, parts, gap_s=0.09):
    gap = np.zeros(int(RATE * gap_s), np.int16)
    frames = []
    for i, p in enumerate(parts):
        if i:
            frames.append(gap)
        frames.append(p)
    data = np.concatenate(frames) if frames else np.zeros(1, np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(data.tobytes())


def build_assets():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    files = {}
    files['alarm'] = os.path.join(AUDIO_DIR, 'alarm.wav')       # 红色急促三连
    files['warn'] = os.path.join(AUDIO_DIR, 'warn.wav')         # 黄色轻嘀
    files['good'] = os.path.join(AUDIO_DIR, 'good.wav')         # 恢复正常叮咚
    _seq_write(files['alarm'], [_tone(1180, 0.13, 0.6), _tone(1180, 0.13, 0.6), _tone(1180, 0.2, 0.6)], 0.07)
    _seq_write(files['warn'], [_tone(760, 0.14, 0.35)], 0.05)
    _seq_write(files['good'], [_tone(660, 0.14, 0.45), _tone(990, 0.24, 0.45)], 0.06)
    return files


class SoundAlert:
    RED_POSTURE = {'HEAD_FWD', 'SLUMP', 'BODY_FWD', 'HEAD_NEAR_DESK', 'SIDE_LEAN'}
    YELLOW_POSTURE = {'TILT_L', 'TILT_R'}

    def __init__(self, enabled=True, confirm_s=4.0, repeat_s=30.0, min_gap_s=2.0):
        self.files = build_assets()
        self.enabled = enabled
        self.ok = None                # None=未测, True/False=喇叭可用否
        self.confirm_s = confirm_s      # 红色持续这么久才响 (与状态机确认一致)
        self.repeat_s = repeat_s        # 红色持续期间重复提醒间隔
        self.min_gap_s = min_gap_s
        self._last_play = 0.0
        self._red_since = 0.0
        self._last_alarm = 0.0
        self._cur_state = None
        self._tilt_notified = False
        self._good_pending = False
        self._lock = threading.Lock()

    def set_enabled(self, on):
        self.enabled = bool(on)

    def probe(self):
        """启动时探测: 播放一段全零静音 wav(不会出声), 验证声卡通路可用."""
        import tempfile
        try:
            fd, tmp = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            _seq_write(tmp, [np.zeros(int(RATE * 0.1), np.int16)], 0.01)
            r = subprocess.run(['aplay', '-q', tmp], timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.ok = (r.returncode == 0)
            os.unlink(tmp)
        except Exception:
            self.ok = False
        return self.ok

    def play_test(self, kind='yellow'):
        key = {'yellow': 'warn', 'red': 'alarm', 'good': 'good'}.get(kind, 'warn')
        self._last_play = 0.0
        self._play(key)
        return key

    def _play(self, key):
        now = time.time()
        if now - self._last_play < self.min_gap_s:
            return
        self._last_play = now
        path = self.files[key]
        def _run():
            try:
                r = subprocess.run(['aplay', '-q', path], timeout=5,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.ok = (r.returncode == 0)
            except Exception as e:
                self.ok = False
                print('[sound] aplay 失败:', e)
        threading.Thread(target=_run, daemon=True).start()

    def update(self, state, level, now):
        """每帧调用; 返回本次触发的音效名或 None (供日志/UI显示)"""
        if state != self._cur_state:
            self._cur_state = state
            if state in self.RED_POSTURE:
                self._red_since = now
                self._last_alarm = 0.0
            elif state in self.YELLOW_POSTURE:
                self._tilt_notified = False
            if state == 'NORMAL':
                self._good_pending = True
            else:
                self._good_pending = False
        if not self.enabled:
            return None
        fired = None
        if state in self.RED_POSTURE:
            if now - self._red_since >= self.confirm_s:
                if now - self._last_alarm >= self.repeat_s:
                    self._last_alarm = now
                    self._play('alarm')
                    fired = 'alarm'
        elif state == 'NORMAL' and self._good_pending:
            self._good_pending = False
            self._play('good')
            fired = 'good'
        elif state in self.YELLOW_POSTURE and not self._tilt_notified:
            self._tilt_notified = True
            self._play('warn')
            fired = 'warn'
        elif state not in self.YELLOW_POSTURE:
            self._tilt_notified = False
        return fired


if __name__ == '__main__':
    sa = SoundAlert()
    t = time.time()
    sa.update('HEAD_FWD', 'red', t)            # 刚进入红色: 未到4s不响
    assert sa.update('HEAD_FWD', 'red', t + 4.1) == 'alarm'
    assert sa.update('HEAD_FWD', 'red', t + 20) is None       # 30s内不重复
    assert sa.update('HEAD_FWD', 'red', t + 34.5) == 'alarm'
    assert sa.update('NORMAL', 'green', t + 30) == 'good'   # 转为正常立即播鼓励音
    assert sa.update('NORMAL', 'green', t + 31) is None      # 只播一次
    assert sa.update('TILT_R', 'yellow', t + 40) == 'warn'
    assert sa.update('TILT_R', 'yellow', t + 41) is None     # 只响一次
    assert sa.update('HEAD_FWD', 'red', t + 60) is None      # 红色刚进不响(未到4s)
    assert sa.update('AWAY', 'yellow', t + 65) is None        # 离座不发声
    print('SOUND_POLICY_OK')
