#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""坐姿监控 Web 实时界面 v2 (局域网 8100) — 低延迟版

v1 卡 7s 的根源及对策:
  1. 单循环 read+推理+渲染 ~6fps < 相机 30fps, 驱动缓冲积压旧帧
     -> 独立采集线程死循环读帧, 只保留"最新一帧"(新帧覆盖旧帧)
  2. 浏览器 fetch 轮询全图 blob -> 改用原生 multipart MJPEG 流 <img>
  3. 1280x720 全帧 PIL 中文渲染 -> 先缩到 800x450 再画骨架/HUD
  4. NPU 单核 -> 尝试三核 NPU_CORE_0_1_2, 失败自动回退 AUTO
另: 集成板载喇叭声音提醒 (posture/sound.py), /api/sound 可开关。
隐私: 帧只在内存, 不落盘; 仅局域网。
"""
import json, os, sys, threading, time
import cv2, numpy as np
from flask import Flask, Response, jsonify, request, render_template

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from pose.engine import YoloPoseRKNN, KP
from posture.features import compute_features
from posture.judge import PostureJudge, GREEN, YELLOW, RED
from posture.temporal import StateMachine
from posture.sound import SoundAlert
from calibration.calibrate import Calibrator, load_calibration
from main import draw_hud, STATE_NAMES

CAM_DEV = 17
W, H = 1280, 720
DISP_W, DISP_H = 800, 450
PORT = 8100
LOG_PATH = os.path.join(BASE, 'logs', 'web_runtime.jsonl')

LEVEL_NAMES = {GREEN: '正常', YELLOW: '注意', RED: '异常'}
SOUND_STATE_FILE = os.path.join(BASE, 'config', 'sound.json')


def load_sound_flag():
    try:
        with open(SOUND_STATE_FILE) as f:
            return json.load(f).get('enabled', True)
    except Exception:
        return True


def save_sound_flag(on):
    try:
        os.makedirs(os.path.dirname(SOUND_STATE_FILE), exist_ok=True)
        with open(SOUND_STATE_FILE, 'w') as f:
            json.dump({'enabled': bool(on)}, f)
    except Exception:
        pass


class Capture(threading.Thread):
    """死循环读相机, 永远把最新帧放进 slot; 下游按需取, 取过的不重复取."""
    daemon = True

    def __init__(self):
        super().__init__()
        self.running = True
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.t_taken = 0.0   # 帧采集时刻(time.time()近似)

    def open(self):
        cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def run(self):
        cap = self.open()
        while cap is None and self.running:
            time.sleep(2)
            cap = self.open()
        fails = 0
        while self.running:
            ret, fr = cap.read()
            if not ret:
                fails += 1
                if fails > 30:
                    print('[cap] 连续读失败, 重开设备')
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(2)
                    cap = self.open()
                    fails = 0
                    if cap is None:
                        while cap is None and self.running:
                            time.sleep(2)
                            cap = self.open()
                continue
            fails = 0
            with self.cond:
                self.frame = fr
                self.t_taken = time.time()
                self.seq += 1
                self.cond.notify_all()

    def take_latest(self, last_seq, block=0.5):
        """返回(seq, frame, 采集时刻); 若没有比 last_seq 更新的帧则等待."""
        with self.cond:
            deadline = time.time() + block
            while self.seq <= last_seq and self.running:
                left = deadline - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)
            if not self.running or self.frame is None:
                return None
            return self.seq, self.frame, self.t_taken

    def peek_latest(self):
        with self.cond:
            if self.frame is None:
                return None
            return self.seq, self.frame, self.t_taken


class Pipeline(threading.Thread):
    """取最新帧->NPU->判定->轻量渲染->发布 jpeg."""
    daemon = True

    def __init__(self, cap_thread):
        super().__init__()
        self.running = True
        self.capth = cap_thread
        self.cond = threading.Condition()
        self.frame_jpeg = None
        self.fseq = 0
        self.state = {'state': 'INIT', 'state_cn': '启动中', 'level': 'yellow',
                      'dur_s': 0, 'fps': 0, 'npu_ms': 0, 'persons': 0,
                      'calibrated': False, 'cal_progress': None,
                      'feats': {}, 'lat_ms': 0, 'sound': load_sound_flag(),
                      'ts': time.strftime('%H:%M:%S')}
        self.recent = []
        self.eng = None
        self.cal_active = False
        self.sound_seq = 0
        self.sound_evt = None
        self.sound = SoundAlert(enabled=load_sound_flag())
        self.sound_ok = self.sound.probe()
        print('[web] 音频设备:', 'OK' if self.sound_ok else '不可用')

    def start_calibrate(self):
        with self.cond:
            if self.cal_active:
                return False, '校准已在进行中'
            self.cal_active = True
            self.capt = Calibrator(min_frames=60, duration_s=15.0)
            self.capt.start()
            return True, '开始校准: 请保持端正坐姿 15 秒'

    def stop_calibrate(self):
        with self.cond:
            self.cal_active = False
            return True, '校准已取消'

    def _apply_calibration(self):
        cal = load_calibration()
        if not cal:
            return
        self.judge = PostureJudge(ref=cal['ref'], calibrated=True,
                                  facing=cal['facing'])
        self.judge.calibrated = True
        with self.cond:
            self.state['calibrated'] = True
            self.state['cal_msg'] = '校准已保存 (%s, %d帧, 朝桌=%s)' % (
                cal['ts'], cal['n_frames'], '画面右侧' if cal['facing'] == 'right' else '画面左侧')

    def run(self):
        print('[web] 加载 NPU 模型...')
        # 实测: 本模型单输出同步推理, 三核无收益(41ms vs 56ms, 多核反而加同步开销), 用 AUTO
        self.eng = YoloPoseRKNN()
        cal = load_calibration()
        self.judge = PostureJudge(ref=cal['ref'] if cal else None,
                                  calibrated=cal is not None,
                                  facing=cal['facing'] if cal else None)
        judge = self.judge
        sm = StateMachine()
        instant_ref_set = False
        last_person_t = time.time()
        fps_avg, last_log, last_sound_log = 0.0, 0.0, 0.0
        last_cap_seq = 0
        loop_t0 = time.time()
        scales = (DISP_W / W, DISP_H / H)

        with open(LOG_PATH, 'a', buffering=1) as logger:
            while self.running:
                got = self.capth.take_latest(last_cap_seq, block=0.5)
                if not got:
                    continue
                seq, frame, t_taken = got
                last_cap_seq = seq
                now = time.time()

                boxes = self.eng.infer(frame)
                person = max(boxes, key=lambda b: (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])) if boxes else None
                feats = None
                if person is not None:
                    last_person_t = now
                    feats = compute_features(person['kpts'])
                    if feats.get('valid') and not judge.calibrated and not instant_ref_set:
                        judge.set_instant_ref(feats)
                        instant_ref_set = True
                    state, level, det = judge.judge(feats)
                else:
                    if now - last_person_t > sm.miss_s:
                        state, level, det = 'AWAY', YELLOW, {}
                    else:
                        state, level, det = 'UNKNOWN', YELLOW, {}
                shown, shlevel, dur = sm.update(state, level, now)

                # 声音提醒: 板载喇叭 + 浏览器页(WebAudio, 靠事件 seq 同步)
                fired = self.sound.update(shown, shlevel, now)
                if fired:
                    self.sound_seq += 1
                    self.sound_evt = {'kind': fired, 'seq': self.sound_seq, 't': time.strftime('%H:%M:%S')}

                # 校准采样
                cp = None
                if self.cal_active:
                    if feats and feats.get('valid'):
                        self.capt.add(feats)
                    cp = self.capt.progress()
                    if self.capt.ready():
                        out, msg = self.capt.finish()
                        self.cal_active = False
                        if out:
                            self._apply_calibration()
                        else:
                            with self.cond:
                                self.state['cal_msg'] = msg

                cyc = max(1e-3, now - loop_t0)
                loop_t0 = now
                fps_avg = (1 / cyc) if fps_avg == 0 else fps_avg * 0.9 + (1 / cyc) * 0.1

                # —— 轻量渲染: 先缩小, 再画 ——
                small = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
                sx, sy = scales
                if person is not None:
                    p2 = dict(person)
                    k = person['kpts'].copy()
                    k[:, 0] *= sx
                    k[:, 1] *= sy
                    p2['kpts'] = k
                    p2['xmin'] *= sx; p2['ymin'] *= sy; p2['xmax'] *= sx; p2['ymax'] *= sy
                    self.eng.draw(small, [p2])
                small = draw_hud(small, shown, shlevel, dur, fps_avg,
                                 self.eng.last_infer_ms, feats,
                                 judge.calibrated, cal_prog=cp)
                if self.cal_active:
                    cv2.putText(small, 'CALIBRATING 15s', (DISP_W - 290, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                ok, jpeg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                lat_ms = (now - t_taken) * 1000 + (time.time() - now) * 1000

                fsummary = {}
                if feats and feats.get('valid'):
                    for kk in ('torso_tilt_deg', 'shoulder_line_deg', 'head_fwd_norm',
                               'head_down_norm', 'trunk_off_norm', 'eye_hip_ratio'):
                        if feats.get(kk) is not None:
                            fsummary[kk] = feats[kk]

                st = {'state': shown, 'state_cn': STATE_NAMES.get(shown, shown),
                      'level': shlevel, 'level_cn': LEVEL_NAMES.get(shlevel, shlevel),
                      'dur_s': round(dur, 1), 'fps': round(fps_avg, 1),
                      'npu_ms': round(self.eng.last_infer_ms, 1),
                      'lat_ms': round(lat_ms), 'persons': len(boxes),
                      'calibrated': judge.calibrated,
                      'cal_progress': round(cp, 3) if cp is not None else None,
                      'sound': {'enabled': self.sound.enabled, 'ok': self.sound.ok},
                      'sound_evt': self.sound_evt,
                      'pending_p': (round(sm.pending_progress, 2)
                                    if sm.pending_state and sm.pending_state != shown else None),
                      'pending_cn': STATE_NAMES.get(sm.pending_state, sm.pending_state)
                                    if sm.pending_state and sm.pending_state != shown else None,
                      'feats': fsummary,
                      'det': {kk: v for kk, v in det.items() if kk != 'all_flags'},
                      'ts': time.strftime('%H:%M:%S')}
                with self.cond:
                    if ok:
                        self.frame_jpeg = jpeg.tobytes()
                        self.fseq += 1
                    self.state = st
                    self.cond.notify_all()   # 关键: v2 漏了这行导致推流被 wait(0.5) 限到2fps
                    if shown != getattr(self, '_last_shown', None) or now - last_log > 5:
                        self._last_shown = shown
                        self.recent.append(st)
                        self.recent = self.recent[-60:]
                if now - last_log > 2:
                    last_log = now
                    logger.write(json.dumps(st, ensure_ascii=False) + '\n')

    def close(self):
        self.running = False
        if self.eng:
            try:
                self.eng.release()
            except Exception:
                pass


app = Flask(__name__, template_folder=os.path.join(BASE, 'ui', 'templates'))
capth = Capture()
pipe = Pipeline(capth)


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/stream')
def stream():
    def gen():
        last = -1
        while pipe.running and capth.running:
            with pipe.cond:
                while pipe.fseq == last and pipe.running:
                    pipe.cond.wait(0.5)
                if not pipe.running:
                    return
                jpg, last = pipe.frame_jpeg, pipe.fseq
            if jpg:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/frame.jpg')
def frame_jpg():
    with pipe.cond:
        jpg = pipe.frame_jpeg
    if jpg is None:
        return 'no frame yet', 503
    return Response(jpg, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.route('/api/state')
def state():
    with pipe.cond:
        return jsonify(pipe.state)


@app.route('/api/recent')
def recent():
    with pipe.cond:
        return jsonify(list(pipe.recent))


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    action = (request.get_json(silent=True) or {}).get('action', 'start')
    ok, msg = pipe.start_calibrate() if action == 'start' else pipe.stop_calibrate()
    return jsonify({'ok': ok, 'msg': msg})


@app.route('/api/sound', methods=['POST'])
def sound():
    body = request.get_json(silent=True) or {}
    if body.get('test'):
        kind = body.get('kind', 'yellow')
        key = pipe.sound.play_test(kind)
        return jsonify({'ok': True, 'played': key, 'enabled': pipe.sound.enabled})
    on = body.get('enabled')
    if on is None:
        return jsonify({'ok': False, 'enabled': pipe.sound.enabled, 'ok_dev': pipe.sound.ok})
    pipe.sound.set_enabled(on)
    save_sound_flag(pipe.sound.enabled)
    msg = '声音提醒已开启' if pipe.sound.enabled else '声音提醒已静音'
    return jsonify({'ok': True, 'enabled': pipe.sound.enabled, 'msg': msg})


def main():
    capth.start()
    pipe.start()
    app.run(host='0.0.0.0', port=PORT, threaded=True)


if __name__ == '__main__':
    main()
