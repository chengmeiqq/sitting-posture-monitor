# PROJECT_STATUS.md — RK3588S 本地儿童学习坐姿监控

## 阶段进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 硬件及环境检查 | ✅ 完成（2026-11-10） |
| 2 | 摄像头测试 | ✅ 完成（2026-11-10） |
| 3 | YOLO Pose 部署 | ✅ 完成（2026-11-10） |
| 4 | 学习坐姿识别算法 | ✅ 代码完成+单元测试全绿（2026-11-10） |
| 5 | 个人坐姿校准 | 🟡 代码完成，**未用小朋友实测**（上学不在家） |
| 6 | 异常判断 | ✅ 代码完成（4s 确认 / 6s 滞回 / 2s 离座） |
| 7 | 实时监控界面 | ✅ Web 8100 已跑通（X11 ffplay 备用方案未做） |
| 8 | 性能测试 | ⏳ 未做（待 30min 长跑 + 内存/断流测试） |
| 9 | 声音提示（附加需求） | ✅ 本机 ALSA 实测发声，未校准自动静音（2026-11-11） |

---

## 阶段1：环境检查报告（2026-11-10）

### 系统
- Ubuntu 20.04.5 LTS (focal)，内核 5.10.66 SMP，aarch64
- 主机名 `rockchip`；SoC：`9tripod,rk3588s` + `rockchip,rk3588`（香橙派/9tripod RK3588S 板）
- CPU：big.LITTLE 8 核：cpu4-7 Cortex-A76 最高 2.256GHz，cpu0-3 Cortex-A55 最高 1.8GHz（lscpu 的 Model name 只显示 cpu0，即小核；实测 scaling_max_freq 确认大小核均在）；governor=ondemand
- 内存：7.5Gi total，可用 4.8Gi
- 存储：/ 共 14G，已用 5.9G（45%），剩 7.3G
- 网络：互联网可达（仅用于下载模型/工具，实时推理不依赖网络）

### NPU
- 内核驱动：RKNPU **v0.9.8**，设备 `fdab0000.npu`；debugfs 可读，NPU 三核负载实时显示 `Core0/1/2: 0%`（空闲正常）
- dmesg 有 `failed to find power_model node`：仅影响功耗估算，不影响推理
- /dev/dri 下有 renderD128(Mali GPU)/renderD129；`v4l2-ctl --info` 对 renderD129 无输出，以 debugfs version/load 为 NPU 存在性证据，阶段3 用真实模型推理做最终确证
- 运行时 `librknnrt.so`：**2.5.0** (74ea630ce8@2026-09-30)，位于 /usr/lib
- Python 端：**rknn-toolkit-lite2 2.3.2**，`from rknnlite.api import RKNNLite` 导入成功
- NPU 调频：devfreq 当前 1000MHz（可用档位 300MHz-1GHz）
- ⚠️ lite2(2.3.2) 与 librknnrt(2.5.0) 主版本不同：转换模型时优先用 rknn-toolkit2 **2.3.2**（与 lite2 严格一致）；若官方预转换 RKNN 模型是 2.5.x/2.3.x 针对 rk3588 的，均可加载（2.x 前向兼容）。阶段3 实测验证。
- ⚠️ 2.3.2 版 RKNNLite 实例无 `RKNN_NPU_CORE_AUTO` 类常量（旧版 API），init_runtime 参数在阶段3 实测确认。

### 摄像头
- 设备：UGREEN Camera 4K，USB (usb-fc880000.usb-1.1)
- 节点：`/dev/video17`（采集）、`/dev/video18`（元数据）、`/dev/media2`
- 支持格式（MJPG）：4000x3000@30 / 3840x2160@30 / 2560x1440 / 1920x1080@60 / 1280x720@60 / 640x480@30 等
- 推荐坐姿监控档位：**1920x1080@30**（模型输入会缩到 640x640，1080p 足够且省带宽）

### 软件栈
- Python 3.8.10（系统自带，不重装）
- numpy 1.24.4 / opencv-python-headless 5.0.0.93（已装）
- gcc 9.4.0 / cmake 3.16.3（可编译 3rdparty demo）
- v4l-utils 1.18.0 已装

### 显示与网络（阶段7/3 相关）
- `DISPLAY=:0`，X11 已在运行（/tmp/.X11-unix/X0 存在）→ 桌面显示方案可行
- 互联网可达（github.com HTTP 200），仅用于下载模型/工具；实时推理不依赖网络

### 遗留/风险
1. 磁盘仅剩 7.3G，克隆仓库/模型时注意（rknn_model_zoo 全量 ~2G 含预编译，浅克隆可省）
2. opencv 是 headless 版：`cv2.imshow` 不可用 → 板上有 X11（:0），阶段7 可安装桌面版 `opencv-python`（pip 级别小改动，届时先报告再动）或改用其他方式显示
3. 本机为 aarch64，**rknn-toolkit2 (x86 PC 端转换工具) 无法在本机安装**；若无预转换 RKNN 模型，需借 x86 主机或云端转换（届时先汇报方案再动手）

### 本阶段未做任何系统修改
仅创建目录 `~/posture_monitor/{models,camera,pose,posture,calibration,ui,config,logs,tests}`。

---

## 阶段2：摄像头测试（2026-11-10，进行中）

测试脚本：`camera/test_camera.py`（OpenCV V4L2 后端，MJPG，实时读帧统计，不保存视频）

| 分辨率 | 时长 | 平均FPS | 中位瞬时FPS | 失败帧 | 亮度均值 | 对比度std |
|---|---|---|---|---|---|---|
| 1920x1080 | 10s | 30.0 | 30.1 | 0/300 | 114.6 | 72.5 |
| 1280x720 | 8s | 30.0 | 29.6 | 0/241 | 115.5 | 72.0 |
| 640x480 | 8s | 30.0 | 30.0 | 0/240 | 105.0 | 79.0 |

- 设备实际协商分辨率与请求一致，fourcc=MJPG，驱动报告 30fps
- 画面非全黑/全白，内容稳定（亮度均值波动 <2），镜头对着有光环境的场景
- P5 帧率 20~25 属 USB MJPG 偶尔抽帧，属正常；日志在 logs/stage2_*.log
- 1080p 连续 5 分钟稳定性长跑：进行中（后台）
- 实时画面验证：opencv 为 headless 版，改用 ffplay/X11 验证（待长跑结束后进行，避免占用设备冲突）


### 实时画面与稳定性（阶段2收尾）
- ffplay 在 X11(:0) 上打开 /dev/video17 实时显示，双帧截屏对比确认画面连续动态更新（运动区域占窗口绝大部分）
- headless opencv 不影响 ffplay/X11 显示路线；阶段7 显示方案可沿用 X11 窗口
- 1080p 连续 300s：8951 帧，0 失败，平均 29.8 FPS，无断流 → 摄像头稳定性通过
- 结论：摄像头推荐档 1280x720@30 或 1920x1080@30 均可稳定供帧

---

## 阶段3：YOLO Pose 部署（2026-11-10）✅

### 模型获取与转换
- 官方 ONNX：ultralytics_yolov8 的 yolov8n-pose.onnx（13MB，md5 7b46c2…，来自 rknn_model_zoo download_model.sh 直链）
- 官方仓库**不提供预转换 .rknn**（model 目录仅 onnx），必须自行转换
- 板上 aarch64 实际**有** rknn-toolkit2 2.3.2 wheel（PyPI aarch64 版，阿里镜像下载）→ 已安装于板端（环境变更点①：pip 安装 rknn-toolkit2 及其依赖 torch2.2/onnx1.16 等；未动系统 Python/驱动）
- 转换：混合量化 step1+step2（官方 convert.py 同参数，COCO 20 图校准，校准图来自 cocodataset.org）
  - 产物 models/yolov8n-pose-int8.rknn，5.0MB，静态 640x640，耗时 46s
  - 注：官方脚本 hybrid 路径缺 load_onnx 步骤（上游 bug），已补上

### NPU 推理确证（非 CPU）
- 板端 rknnlite 2.3.2 + librknnrt 2.5.0 + 驱动 0.9.8，init_runtime(target=None) 直接绑定本机 RKNPU
- 运行时日志确认：`RKNN Model Information ... target: RKNPU v2, target platform: rk3588`
- 推理期间 /sys/kernel/debug/rknpu/load 实测 Core0 25~35% 负载（空闲时 0%）→ 确证走 NPU
- core_mask 实测：AUTO≈47ms、CORE0≈48ms、0_1≈84ms、0_1_2≈64ms（模型小，多核调度反而慢）→ 采用 AUTO

### 性能（bus.jpg 60 次连续推理）
- NPU 推理延迟 avg 48.0ms / p50 48.9 / p95 54.0 → 纯推理 ≈21 FPS
- 端到端(含采集/letterbox/后处理) 1280x720 实时 30s：16.8 FPS（瓶颈为后处理 Python 循环，阶段4 再优化；对坐姿监控 5-10FPS 已足够）
- 报告：logs/stage3_bench.json / stage3_coremask.json / stage3_live.json

### 正确性验证
- bus.jpg 检出 3 人（score 0.85-0.89），每人 17 关键点全可见
- 关键点语义校验通过（鼻/眼/耳/肩/髋/膝相对位置全部合理）
- 实时摄像头帧检出在座人员，肩髋几何合理；骨架图 tests/bus_pose_result.jpg、logs/stage3_live_frame.jpg
- 曾发现并修复：Lite2 输入需 4D、关键点通道布局需按 (1,17,3,8400) 处理

### 代码
- pose/engine.py：YoloPoseRKNN（加载/推理/letterbox/向量化后处理/NMS/骨架绘制）
- tests/test_pose_npu.py、tests/test_live_pose.py

---

## 参考项目吸收 + 儿童适配（2026-11-10 晚）

用户指定参考 BNDSer/Sitting-posture-detection-system（高中生研学项目, 目标为一年级女生）：

### 借鉴点
- 其核心特征：眼-耳水平距 / 肩宽, 固定灵敏度 1.3 报警（Windows+GPU, ultralytics, winsound 蜂鸣）
- 我们吸收为 **eye_hip_ratio**（眼(或鼻)中点-髋中心距 / 肩宽, 归一化）, 加入 features.compute_features
- 不照抄其"固定系数+单特征+立即蜂鸣"：改为
  1. 相对**个人校准基准**的偏移量（阶段5 校准会采集 eye_hip_ratio 中位数）
  2. **双证据**才报 HEAD_FWD：比值明显下降 且 头颈下沉趋势 slump_dev 超阈
  3. 仍走阶段6 多帧确认(4s)+滞回, 不因翻书拿笔误报

### 儿童(一年级女生)适配
- DEFAULT_REF.eye_hip_ratio=3.2 按儿童头身比/窄肩经验值; 校准后覆盖
- HUD 中文化：状态用中文大字(PIL+Noto Sans CJK), cv2.putText 不支持中文已解决
- 实测: 摄像头前成人 sit 时 eye_hip_ratio≈2.3~3.2, 合成儿童几何正坐 4.7/前伸 4.4, 特征链路有效
- tests/test_posture_logic.py 全绿（新特征未破坏原6类姿态判定）

### 校准待做
- 等孩子实际坐在书桌前, 运行 `python3 main.py --calibrate`（15s 正常坐姿）建立个人基准
- 未校准时系统用"当前帧临时基准+保守阈值", 显示 NO-CAL 角标

### 已知边界
- 参考项目的 winsound/pyautogui/ultralytics 依赖均不需要（我们走 RKNN NPU）
- 不引入语音播放（超出授权范围, 需用户明确要求）

---

## Web 实时测试界面（用户明确要求，属对原规则6"不做网页服务"的一次授权豁免）

- `web.py`：与 main.py 同一算法管线（NPU YOLO Pose → features → judge → 状态机），Flask 局域网 8100 端口，不依赖外网
- 页面 `/`：实时画面(骨架+中文HUD) + 大字号状态 + 特征偏移表 + 校准按钮(15s 进度条) + 状态事件流
- `/api/state` `/api/frame.jpg` `/api/stream`(multipart) `/api/calibrate` `/api/recent`
- 隐私：帧只在内存，不落盘；无录像无云端
- 访问：http://192.168.x.x:8100 （板端 setsid 启动, 日志 logs/web.log, 运行态 logs/web_runtime.jsonl）
- 实测：FPS 15~18，NPU 37~40ms；校准 start/stop API 流程通过；人在画面走动且未校准时保守显示"未知"(黄)，符合"不允许误判为正常"
- 等待：小朋友放学后正式 --calibrate；用户先用自己试流程

---

## 2026-11-11 02:50 UTC 进度核查（板子重启后）

- 板子在 02:16 重启过，web.py 未自启（无 systemd 单元）→ 已 `setsid python3 web.py` 重新拉起，PID 46392
- 重启后实测：`/` 200 (8.7KB)，`/api/state` 正常，FPS ~15，NPU 32~39ms
- NPU 负载连续三次采样 Core0 23~27%，Core1/2 0% → 仍确证走 NPU，非 CPU 冒充
- 摄像头节点未变：`/dev/video17`（UGREEN 4K，按 by-id 绑定）
- 仍未校准：`config/calibration.json` 不存在，state 里 `calibrated:false`
- 当前画面里坐着人（persons=1），未校准下判 HEAD_NEAR_DESK(红) —— 属临时基准误报，校准后应消失
- 历史运行态日志 970 帧：UNKNOWN 720 / NORMAL 198 / AWAY 27 / HEAD_FWD 25（多为未校准阶段样本）

---

## 2026-11-11 03:12 UTC 延迟专项排查与优化（用户反馈"感觉延迟有点高"）

### 实测拆解（优化前）
| 环节 | 耗时 | 说明 |
|---|---|---|
| 采集 read() | 8~10ms | BUFFERSIZE=2，但主循环边读边推理，帧在 V4L2 队列里堆积 |
| NPU 推理 | 40~50ms | 正常 |
| 骨架+HUD 绘制 @960 | 12~20ms | 中文 HUD 每帧整帧 BGR↔RGB 往返 |
| JPEG 编码 | 4~6ms | 正常 |
| **单帧循环合计** | **~75ms** | → 13fps，队列越堆越久 → 画面明显滞后 |
| 前端轮询 frame.jpg | 120ms/次 | 每次都新建 TCP+HTTP，浏览器解码排队 → 雪球式延迟 |

### 优化项（均已实测）
1. **独立采集线程**：只保留最新一帧覆盖（丢队列），V4L2 缓冲不再堆积；采集实测稳定 29~30fps
2. **独立渲染线程 @30fps**：缩放/骨架/HUD/编码不再占用推理主循环；推理降到 11~15fps 也不影响视频流畅度
3. **推理节流** 15fps（坐姿判定不需要更高），把 CPU 让给渲染
4. **先缩放再绘制**：在 960x540 上画骨架和 HUD，省 ~8ms
5. **中文 HUD 走文字位图缓存**：状态文字不变时 0 次 PIL 转换（12.7ms → 0.1ms 命中）
6. **HUD 层缓存 0.4s + 只在顶栏/底栏叠加**，不再整帧布尔掩码
7. **cv2.setNumThreads(1)**：实测小图绘制多线程反而慢 3.5x（4.3ms → 1.2ms）
8. **前端改 multipart /api/stream** 长连接，不再 120ms 轮询 frame.jpg

### 优化后（当前运行态）
- 采集 29~30fps；推理 10.7~11.9fps（NPU 50~60ms，板子跑热后略降）
- `frame_age_ms` 0~26ms（推理取到的帧有多新）
- `pipe_ms` 43~81ms（采集→状态输出）
- `overlay_age_ms` 55~68ms（骨架比当前画面滞后多少）
- 渲染分解 `render_parts` = [resize, draw, hud, encode] ≈ [5, 5, 3, 10]ms
- JPEG 960x540 q78 ≈ 68KB/帧

### 剩余的"感觉延迟"来源（非 bug）
1. **4 秒多帧确认**（阶段6 设计）：异常必须持续 4s 才升级显示。已在 `/api/state` 增加 `pending / pending_cn / pending_p`，页面显示"已检测到 X，还需 N 秒确认"，避免误以为卡死
2. **6 秒恢复滞回**：恢复也要 6s 才回落
3. 未校准导致误报（eye_hip_ratio 实测 0.78 vs 默认基准 3.2，偏差巨大）→ 校准后误报和状态跳变会明显减少

### 当前运行
- http://192.168.x.x:8100 ，已由 systemd 托管（`deploy/posture-web.service`，enabled+running，`Restart=always`）
- 当前实例 PID 见 `ps -eo pid,cmd | grep '[w]eb.py'`（每次重启会变，故不写死）
- 管理：`systemctl status|restart posture-web`


---

## 阶段 9 追加：声音提示（2026-11-11 04:2x UTC）

> ⚠️ **本节为历史记录，实现已被下方「v2 低延迟 + 声音提醒」取代。**
> 当前生效的是 `posture/sound.py` + `config/sound.json` + `audio/*.wav`（音高 1180/760/660Hz，48kHz 单声道）。
> 本节对应的 `ui/sound.py` 已废弃且未被任何代码引用，`config/alert.json` 仅作旧参数留档。
> 保留本节是因为三次尝试的**失败记录**（尤其“常驻 aplay 喂静音流会卡在 PREPARED”）仍然有价值。

### 为什么用直接 ALSA，不用 PulseAudio
`pactl` 连接被拒（PulseAudio 不可用），板子声卡为 card 0 `rockchipes8388`（ES8323 codec），
`aplay -l` 有 `dailink-multicodecs`。直接走 `hw:0,0` 纯 ALSA 即可，无需安装任何新依赖。

### 实现路径的三次尝试（前两次失败，记录以免重走）
1. **每次提示新建 aplay 播 wav** —— 能响，但进程启动 + 打开 PCM 约 150~560ms，
   提示音明显滞后于画面判定。放弃。
2. **常驻 aplay 吃字节流 + 后台持续喂静音** —— 想做到零延迟插播。
   在这块板子的 ES8323+I2S 驱动上，aplay 写满缓冲区后**始终停在 `PREPARED` 不触发**
   （`hw_ptr` 永远 0，`appl_ptr` 停在 18000），提示音实际不出声。放弃。
3. **采用方案 1 的改良版（当前实现）** —— 启动时把三种提示音预渲染成 wav 放临时目录，
   提醒时用**后台线程**起 `aplay` 播放，不阻塞主监控循环；并用锁串行化，
   避免上一次 aplay 未退干净导致下一次 `Device or resource busy`。

### 三种提示音
| 名称 | 波形 | 用途 |
|------|------|------|
| `yellow` | 660Hz 双短音 | 黄色注意 |
| `red` | 1050Hz 三连急促音 | 红色异常 |
| `ok` | 880Hz 单音 | 恢复（当前仅手动试听） |

音量 0.35，带 8ms 淡入淡出防爆音，48kHz 双声道 S16_LE。

### 触发策略
- 只对**已确认**的异常姿态响（`HEAD_FWD / BODY_FWD / SLUMP / HEAD_NEAR_DESK / TILT_L / TILT_R / SIDE_LEAN` 且等级为黄/红）
- 等级升级（黄→红）立即响；持续同一异常每 `repeat_s = 30s` 重复一次；`min_gap_s = 1.5s` 防连响
- `NORMAL / UNKNOWN / AWAY / INIT` 一律不响
- **未校准时静音**（`silent_when_uncalibrated = true`）：当前 `calibrated=false`，
  未校准会大量误报成红色 HEAD_FWD，若照常发声就是无意义乱叫。校准完成后自动恢复发声
- 音频设备故障自动降级静音，`status().ok=false`，绝不影响摄像头与推理主流程

### 代码改动（历史：`ui/sound.py` 方案，已废弃）
- ~~新增 `ui/sound.py`（`AlertSound`：`beep() / update(level,state) / set_enabled() / status() / close()`）~~ → 现为 `posture/sound.py`（`SoundAlert`）
- ~~新增 `config/alert.json`~~ → 现为 `config/sound.json`（仅 `{"enabled": bool}`）+ 代码内默认参数
- `web.py`：接入 `AlertSound`，在 `sm.update()` 之后按状态触发；`/api/state` 增加 `sound` 字段；
  新增 `GET|POST /api/sound`（`enabled` 开关 / `test` 试听 / `kind` 音型 / `volume`）；`close()` 释放 aplay
- `ui/templates/dashboard.html`：新增「声音提示」卡片（开/关按钮、试听按钮、音频不可用提示）

### 实测结论
- 模块自测：`beep('yellow'/'red'/'ok')` 均返回 True 且连续调用不报错；
  `update('yellow',HEAD_FWD)` → True，`update('green',NORMAL)` → False，符合设计
- **发声真正确认**：播放期间读 `/proc/asound/card0/pcm0p/sub0/status`，
  状态为 `DRAINING`（`RUNNING` 后正常排空），`hw_ptr` 在推进 → 音频确实送进 DAC，不是假发声
- 混音器：`Headphone` 已 unmute 到 100%，`Playback De-emphasis` 设为 48Khz（与采样率一致）
- 服务侧：`/api/sound` 返回 `ok:true`；`/api/state` 的 `sound` 字段正确；
  点「试听」后 `events` 记录 `{level:red, state:HEAD_FWD, ts:...}`，串行化后无 Device busy
- 未校准静音验证：服务跑 20s 后 `state=UNKNOWN yellow calibrated=False`，`sound.events` 为 **0 条**（未乱响）

---

## v2 低延迟 + 声音提醒（2026-11-11 凌晨, 用户外出期间自主完成）

用户反馈: ①网页约7s延迟 ②没有声音。处理:

### 延迟 7s → ~0.1s (实测 lat_ms 中位 70ms)
1. **根因**: 单循环 read→推理→渲染 ~6fps < 相机30fps, V4L2 驱动缓冲积压旧帧, 越积越深
   - **修复**: 独立 Capture 线程死循环读帧只留最新一帧(seq+Condition), 推理线程按需 take_latest()
2. 浏览器 120ms fetch-blob 轮询 → 原生 multipart MJPEG `<img>` 长连接(/api/stream), 断线自动降级轮询
3. 渲染先缩到 800x450 再画骨架/HUD/中文(PIL), JPEG 质量 75
4. NPU 三核实测无收益(同步推理 41ms vs 56ms, 反而变慢), 保持 AUTO 单核
- 稳态: FPS 14~15, NPU 41~47ms, 端到端延迟中位 70ms (最大150ms)

### 声音提醒 (posture/sound.py, 板载 es8388 喇叭)
- numpy 合成 48kHz wav(alarm急促三连/good恢复叮咚/warn轻嘀), aplay 播放; es8388 不支持 22050 采样率
- 策略: 红色异常确认(4s)后响, 持续每30s重复; 恢复正常播鼓励音一次; 歪肩黄只响一次; 离座不响
- 网页: 声音开关(持久化 config/sound.json)+试听按钮; 启动时静音文件探测声卡(实测 ok=True)
- 单元策略测试通过 (SOUND_POLICY_OK)

### 前端补充
- 状态机暴露 pending_state/pending_progress → 页面显示"已检测到头部前倾, 还需Xs确认"
- 页头新增端到端延迟徽章(绿<600ms/黄<1200ms/红)
- 校准/流/声音 API 全部实测通过

### 阶段8-浸泡 (05:20:45 UTC 完成, 结果实测)
- 30分钟浸泡 (04:50:45→05:20:45, 10s采样): **samples=180 fails=0** ✅
- FPS 均值 14.7 / 最低 12.9; 端到端延迟 中位 68ms / 最大 151ms; NPU 中位 40ms
- 状态分布: NORMAL 178 / UNKNOWN 2 (2次为采样瞬间无人体, 符合设计)
- 浸泡期间内存无异常增长迹象, 服务未重启未断流; 05:26 复查 /api/state 健康 (PID 189035, fps 15.1, lat 68ms)
- 待小朋友放学后: 正式个人校准 + 儿童阈值微调 + 阶段8剩余项(长时CPU/内存曲线、断流自动恢复验证)
