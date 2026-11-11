#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地儿童学习坐姿 AI 监控 — 主程序 (RK3588S + NPU)
用法:
  python3 main.py                 # 正常运行(用已有校准, 无校准则默认基准+提示)
  python3 main.py --calibrate     # 校准模式: 保持正常坐姿 15s
  python3 main.py --no-display    # 无界面(仅日志)
  python3 main.py --res 1920x1080 # 摄像头分辨率
隐私: 视频只在内存中实时处理, 不保存。
"""
import argparse, json, os, sys, time
import cv2, numpy as np

try:   # 中文 HUD 需 PIL + CJK 字体 (cv2.putText 不支持中文)
    from PIL import Image, ImageDraw, ImageFont
    _FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    _FONTS = {}
    _TEXT_CACHE = {}
    def _font(px):
        if px not in _FONTS:
            _FONTS[px] = ImageFont.truetype(_FONT_PATH, px)
        return _FONTS[px]
    def draw_text_cn(img, xy, text, px, color_bgr):
        # 文字只在状态变化时变化, 缓存渲染结果 -> 稳态下 0 次 PIL 转换
        key = (text, tuple(color_bgr), px)
        hit = _TEXT_CACHE.get(key)
        if hit is None:
            font = _font(px)
            tmp = Image.new('RGB', (8, 8))
            bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
            tw, th = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
            timg = Image.new('RGB', (tw, th), (0, 0, 0))
            ImageDraw.Draw(timg).text((-bbox[0], -bbox[1]), text, font=font,
                                      fill=color_bgr[::-1])
            arr = cv2.cvtColor(np.array(timg), cv2.COLOR_RGB2BGR)
            hit = (arr, tw, th)
            if len(_TEXT_CACHE) > 64:
                _TEXT_CACHE.clear()
            _TEXT_CACHE[key] = hit
        arr, tw, th = hit
        h, w = img.shape[:2]
        ty = xy[1] - px
        y0, x0 = ty, xy[0]
        y1, x1 = min(h, y0 + th), min(w, x0 + tw)
        if y1 <= y0 or x1 <= x0:
            return img
        # 只把文字块内"亮像素"贴到帧上, 保留背景 (避免黑底方块)
        patch = arr[:y1 - y0, :x1 - x0]
        mask = patch.any(axis=2)
        dst = img[y0:y1, x0:x1]
        dst[mask] = patch[mask]
        return img

    CN = True
except Exception:
    CN = False
    def draw_text_cn(img, xy, text, px, color_bgr):
        cv2.putText(img, text.encode('ascii', 'ignore').decode(), (xy[0], xy[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, px / 32.0, color_bgr, 2, cv2.LINE_AA)
        return img

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from pose.engine import YoloPoseRKNN, KP
from posture.features import compute_features
from posture.judge import PostureJudge, GREEN, YELLOW, RED
from posture.temporal import StateMachine
from calibration.calibrate import Calibrator, load_calibration

CAM_DEV = 17
LOG_DIR = os.path.join(BASE, 'logs')
STATE_NAMES = {
    'NORMAL': '正常', 'HEAD_FWD': '头部前倾', 'BODY_FWD': '身体前倾',
    'SLUMP': '驼背趋势', 'HEAD_NEAR_DESK': '头近桌面', 'TILT_L': '左倾',
    'TILT_R': '右倾', 'SIDE_LEAN': '身体侧歪', 'AWAY': '离开座位', 'UNKNOWN': '未知'}
LEVEL_RGB = {GREEN: (0, 200, 0), YELLOW: (0, 200, 255), RED: (0, 0, 255)}


def draw_hud(img, state, level, dur, fps, npu_ms, feats, calibrated, cal_prog=None):
    color = LEVEL_RGB.get(level, (200, 200, 200))
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (14, 46), color, -1)
    txt = '%s' % STATE_NAMES.get(state, state)
    if dur > 0.5:
        txt += '  %.0fs' % dur
    if CN:
        draw_text_cn(img, (26, 40), txt, 30, color)
    else:
        cv2.putText(img, txt, (26, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    info = 'FPS %.1f  NPU %.0fms  %s' % (fps, npu_ms, 'CAL' if calibrated else 'NO-CAL')
    cv2.putText(img, info, (w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    if cal_prog is not None:
        cv2.rectangle(img, (0, h - 18), (int(w * cal_prog), h), (255, 200, 0), -1)
        cv2.putText(img, 'Calibrating... keep normal posture', (10, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # 特征小字
    if feats and feats.get('valid'):
        line = 'tilt %.0f  shline %.0f  head_fwd %.2f  head_dn %.2f' % (
            feats['torso_tilt_deg'], feats['shoulder_line_deg'],
            feats['head_fwd_norm'], feats['head_down_norm'])
        cv2.putText(img, line, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--no-display', action='store_true')
    ap.add_argument('--res', default='1280x720')
    args = ap.parse_args()
    W, H = map(int, args.res.lower().split('x'))

    os.makedirs(LOG_DIR, exist_ok=True)
    logger = open(os.path.join(LOG_DIR, 'runtime.jsonl'), 'a', buffering=1)

    print('加载 NPU 模型...')
    eng = YoloPoseRKNN()
    cal = load_calibration()
    calibrated = cal is not None
    judge = PostureJudge(ref=cal['ref'] if calibrated else None,
                         calibrated=calibrated,
                         facing=cal['facing'] if calibrated else None)
    instant_ref_set = False
    if calibrated:
        print('已加载个人校准: %s facing=%s (%s)' % (cal['ts'], cal['facing'], cal['n_frames']))
    else:
        print('!! 无个人校准: 使用默认保守基准, 建议先运行 --calibrate')
    sm = StateMachine()
    capt = Calibrator()
    if args.calibrate:
        capt.start()

    disp = None
    if not args.no_display:
        try:
            from ui.display import PipeDisplay
            disp = PipeDisplay(960, int(960 * H / W), fps=15)
        except Exception as e:
            print('显示初始化失败(%s), 转 --no-display' % e)
            disp = None

    def open_cam():
        cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap if cap.isOpened() else None

    cap = open_cam()
    assert cap, '摄像头打开失败'
    print('摄像头 %dx%d 已打开' % (W, H))

    fps_avg = 0.0
    last_person_t = time.time()
    n_iter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                t_fail = time.time()
                while time.time() - t_fail < 2:
                    ret, frame = cap.read()
                    if ret:
                        break
                    time.sleep(0.1)
                else:
                    pass
                if not ret:
                    print('摄像头读帧失败, 重连...')
                    cap.release()
                    for _ in range(3):
                        time.sleep(1.5)
                        cap = open_cam()
                        if cap:
                            break
                    if cap is None:
                        print('重连失败, 3s 后重试主循环')
                        cap = open_cam() or cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
                    continue
            t_prev = time.time()
            now = t_prev
            fps = 1.0 / max(1e-3, now - getattr(main, '_lt', now - 0.06))
            main._lt = now
            fps_avg = fps if fps_avg == 0 else fps_avg * 0.9 + fps * 0.1

            boxes = eng.infer(frame)

            # 选最近(最大)的一人
            person = max(boxes, key=lambda b: (b['xmax']-b['xmin'])*(b['ymax']-b['ymin'])) if boxes else None
            feats = None
            if person is not None:
                last_person_t = now
                feats = compute_features(person['kpts'])
                if feats.get('valid') and not calibrated and not instant_ref_set:
                    judge.set_instant_ref(feats)
                    instant_ref_set = True
                    print('!! 未校准: 已用当前姿态作临时基准(保守阈值), 请尽快 --calibrate')
                state, level, det = judge.judge(feats)
            else:
                if now - last_person_t > sm.miss_s:
                    state, level, det = 'AWAY', YELLOW, {}
                else:
                    state, level, det = 'UNKNOWN', YELLOW, {'reason': 'no_person_transient'}
            shown, shlevel, dur = sm.update(state, level, now)

            if args.calibrate:
                if feats and feats.get('valid'):
                    capt.add(feats)
                p = capt.progress()
                if capt.ready():
                    out, msg = capt.finish()
                    print('校准完成:', msg)
                    if out:
                        print(json.dumps(out, ensure_ascii=False)[:400])
                    return 0
            elif n_iter % 30 == 0:  # 0.5s@60fps ~ 记录
                rec = dict(ts=time.strftime('%H:%M:%S'), state=shown, level=shlevel,
                           dur_s=round(dur, 1), fps=round(fps_avg, 1),
                           npu_ms=round(eng.last_infer_ms, 1))
                if feats and feats.get('valid'):
                    rec.update({k: feats[k] for k in
                                ('torso_tilt_deg', 'shoulder_line_deg',
                                 'head_fwd_norm', 'head_down_norm', 'trunk_off_norm',
                                 'eye_hip_ratio')
                                if k in feats and feats[k] is not None})
                logger.write(json.dumps(rec, ensure_ascii=False) + '\n')

            if person is not None:
                frame = eng.draw(frame, [person])
            frame = draw_hud(frame, shown, shlevel, dur, fps_avg, eng.last_infer_ms,
                             feats, calibrated,
                             cal_prog=(capt.progress() if args.calibrate else None))
            if disp is not None and n_iter % 4 == 0:  # ~15fps 显示
                disp.put(cv2.resize(frame, (960, int(960 * H / W))))
            n_iter += 1
    except KeyboardInterrupt:
        print('退出')
    finally:
        if disp:
            disp.close()
        cap.release()
        eng.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())
