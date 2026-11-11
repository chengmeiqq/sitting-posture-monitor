# -*- coding: utf-8 -*-
"""状态时间平滑 (阶段6): 多帧确认 + 恢复滞回
规则:
- 异常状态须连续持续 hold_s(默认4s, 3~5s 可调) 才升级显示
- 短暂异常(< hold_s)自动消化, 不报警
- 恢复须连续 NORMAL 达 recover_s(默认6s) 才回落, 防抖动
- 连续 miss_s(默认2s) 检测不到人 -> AWAY; 关键点无效 -> UNKNOWN(不计正常)
"""
import time
from posture.judge import GREEN, YELLOW, RED

SEVERITY = {GREEN: 0, YELLOW: 1, RED: 2}


class StateMachine:
    def __init__(self, hold_s=4.0, recover_s=6.0, miss_s=2.0):
        self.hold_s = hold_s
        self.recover_s = recover_s
        self.miss_s = miss_s
        self.shown_state = 'UNKNOWN'      # 当前界面显示的状态
        self.shown_level = YELLOW
        self._cand = None                 # 候选(未确认)状态
        self._cand_since = None
        self._normal_since = None
        self._last_seen = None            # 上次检测到有效人体的时刻
        self.abnormal_since = None        # 当前异常显示开始时刻(用于异常持续时长)
        self.pending_state = None         # 未确认候选状态(供 UI 显示倒计时)
        self.pending_progress = 0.0

    def update(self, state, level, t=None):
        """t: time.time() 默认. 返回 (shown_state, shown_level, dur_s)"""
        if t is None:
            t = time.time()
        if state == 'AWAY':
            # 无人直接切走(去抖由相机层保证 miss_s)
            self.shown_state, self.shown_level = 'AWAY', YELLOW
            self._cand = None
            self.pending_state, self.pending_progress = None, 0.0
            self.abnormal_since = None
            return self.shown_state, self.shown_level, 0.0

        if state == 'UNKNOWN':
            # 不允许误判为正常: 立即降为 UNKNOWN 显示, 但不清除异常候选进度
            self.shown_state, self.shown_level = 'UNKNOWN', YELLOW
            self.abnormal_since = None
            return self.shown_state, self.shown_level, 0.0

        if state == 'NORMAL':
            self._cand = None
            self.pending_state, self.pending_progress = None, 0.0
            if SEVERITY.get(self.shown_level, 2) > 0 or self.shown_state != 'NORMAL':
                # 需要连续 NORMAL recover_s 才回落
                if self._normal_since is None:
                    self._normal_since = t
                if t - self._normal_since >= self.recover_s:
                    self.shown_state, self.shown_level = 'NORMAL', GREEN
                    self.abnormal_since = None
                    self._normal_since = None
            else:
                self.shown_state, self.shown_level = 'NORMAL', GREEN
                self.abnormal_since = None
            dur = (t - self.abnormal_since) if self.abnormal_since else 0.0
            return self.shown_state, self.shown_level, dur

        # 异常状态
        self._normal_since = None
        if self._cand == state:
            if t - self._cand_since >= self.hold_s and self.shown_state != state:
                self.shown_state = state
                self.shown_level = level
                if self.abnormal_since is None:
                    self.abnormal_since = self._cand_since
            elif self.shown_state == state and SEVERITY.get(level, 0) > SEVERITY.get(self.shown_level, 0):
                # 同状态严重度升级立即生效(降级仍需滞回)
                self.shown_level = level
        else:
            self._cand = state
            self._cand_since = t
        if self.shown_state != self._cand:
            self.pending_state = self._cand
            self.pending_progress = min(1.0, max(0.0, (t - self._cand_since) / self.hold_s))
        else:
            self.pending_state, self.pending_progress = None, 0.0
        if self.shown_state == state and self.abnormal_since is None:
            self.abnormal_since = t
        dur = (t - self.abnormal_since) if self.abnormal_since else (t - self._cand_since)
        return self.shown_state, self.shown_level, dur
