# 儿童学习坐姿监控 (RK3588 NPU 本地实时版)

一年级小学生在家学习的坐姿实时监测系统：**树莓派类开发板 (Rockchip RK3588S) + USB 摄像头 + NPU 本地推理**，
检测低头、头前伸、趴桌、弯腰、侧倾、歪肩等不良坐姿，网页实时查看 + 声音提醒。

**隐私设计**：视频画面只在内存中实时处理，不录像、不存图、不做人脸识别、不上传云端。
唯一出板的数据是局域网内的网页预览流。

## 效果

- 15 FPS 实时检测，NPU 推理 ~40ms，端到端画面延迟 <150ms
- 6 类不良姿态判定，全部基于**相对个人校准基准**的偏移量，不用拍脑袋的绝对角度
- 时间平滑：异常须持续 4 秒才报警（翻书、拿笔不误报）；恢复须连续正常 6 秒才回落（不闪烁）
- 声音提醒：优先在**浏览器端**发声（板子不用接喇叭），板载声卡可同步输出
- 局域网 Web 界面：实时画面+骨架、状态大字、特征数值、15 秒个人校准流程

## 硬件环境

| 项目 | 本仓库验证于 |
|------|-------------|
| 开发板 | 香橙派类 RK3588S (8G)，Ubuntu 20.04, aarch64 |
| NPU 运行时 | librknnrt 2.5.0 / rknpu 驱动 0.9.8 / rknn-toolkit-lite2 2.3.2 |
| 摄像头 | 任意 UVC USB 摄像头 (/dev/video*, MJPG 1280x720@30) |
| 推理模型 | YOLOv8n-Pose → RKNN INT8 (640x640, COCO 17 关键点) |

## 快速开始

```bash
# 1. 依赖: python3 + rknn-toolkit-lite2 (Rockchip 官方 wheel)
#    pip3 install rknn_toolkit_lite2-*.whl  (板端, 见 requirements.txt 注释)
#    pip3 install -r requirements.txt

# 2. 生成 RKNN 模型 (在 x86 Linux 上转换一次即可; 见下"模型转换")
python3 models/convert_rknn.py            # 产出 models/yolov8n-pose-int8.rknn
# YOLOv8 基础权重(yolov8n-pose.pt/onnx)因 AGPL-3.0 许可不随本仓库分发, 需自行从上游获取

# 3. 自检
python3 tests/test_posture_logic.py       # 姿态判定单元测试
python3 camera/test_camera.py             # 摄像头检查

# 4. 启动 Web 界面 (局域网)
python3 web.py                            # http://<板子IP>:8100

# 5. 首次使用: 打开网页 → 让孩子端正坐好 → 点"开始校准" 15 秒
```

无显示器的纯命令行模式：`python3 main.py`（HUD 走 ffplay 管道，可加 `--no-display`）。

### 运行时配置文件

仓库只提交模板，真实文件由程序/校准流程生成并被 `.gitignore` 排除（含个人基准数据）：

| 文件 | 来源 | 说明 |
|------|------|------|
| `config/calibration.json` | 网页校准 或 `calibration/calibrate.py` | 个人基准；缺失时判"未校准"，保守不报警 |
| `config/sound.json` | `POST /api/sound` | `{"enabled": bool}`，仅持久化开关 |
| `config/alert.example.json` | 仓库模板 | 旧版 `ui/sound.py` 的参数留档，**当前代码不读取** |

### 开机自启 (systemd)

```bash
sudo cp deploy/posture-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now posture-web
systemctl status posture-web
```

单元里写了 `WorkingDirectory=/root/posture_monitor`，换路径要一并改；`User=root` 是因为要访问 `/dev/video*` 与 ALSA 设备，如需降权请自行加 `video`/`audio` 组。


## 架构

```
UVC摄像头 ──采集线程(只留最新帧)──┐
                                   ├─ pose/engine.py   YOLOv8n-Pose RKNN 推理(NPU)
                                   ├─ posture/features.py  17关键点 → 归一化特征
                                   ├─ posture/judge.py     个人基准偏移 + 6类规则
                                   ├─ posture/temporal.py  4s确认/6s滞回状态机
                                   ├─ posture/sound.py     提示音策略(板载喇叭)
                                   └─ web.py               Flask MJPEG流+状态JSON+浏览器WebAudio
校准:  calibration/calibrate.py   15s 采样取中位数 → config/calibration.json
```

### 姿态特征（全部以肩宽归一化，对拍摄距离/体型不敏感）

| 特征 | 含义 | 用于 |
|------|------|------|
| `head_down` | 耳-鼻垂直距/肩宽 | 低头 |
| `head_fwd` | 鼻-肩水平距/肩宽 | 头前伸 |
| `eye_hip_ratio` | 眼(鼻)中点—髋中心距/肩宽 | 颈部前伸(双证据之一) |
| `body_fwd` | 鼻—椅参考点距/肩宽 | 身体前倾 |
| `torso_tilt` | 肩中点—髋中点连线与垂线夹角 | 弯腰 |
| `shoulder_line` | 肩线倾角 | 歪肩 |
| `trunk_off` | 躯干水平倾角 | 侧歪 |

**头前伸双证据**：`eye_hip_ratio` 明显下降 **且** 头颈下沉量超阈才报 HEAD_FWD，
大幅降低"只是探身拿东西"的误报。（`eye_hip_ratio` 思路参考了社区项目 BNDSer/Sitting-posture-detection-system 的归一化比值法。）

### 判定规则

- 所有阈值都是**相对校准基准的偏移量**，不是绝对值 → 换孩子/换机位只需重新 15s 校准
- 未校准时用"当前帧临时基准 + 保守阈值"，宁显示"未知"不误判为正常
- 报警节奏：异常持续≥4s 才确认；确认后持续红色每 30s 重复提醒；坐正后播一次"恢复"鼓励音

## Web API

| 路由 | 说明 |
|------|------|
| `GET /` | 仪表盘 (实时流+校准+声音开关) |
| `GET /api/stream` | multipart MJPEG 流 (画面已画好骨架/HUD) |
| `GET /api/state` | 当前状态 JSON (state/level/fps/lat_ms/feats/sound_evt) |
| `POST /api/calibrate` | `{"action":"start"|"stop"}` 15s 个人校准 |
| `POST /api/sound` | `{"enabled":bool}` / `{"test":true,"kind":"red"}` |

## 已知边界

- 单人场景设计；多人时取最大人体框
- 侧背对摄像头超过 45°、或趴桌后关键点全部遮挡时，判"未知"而非猜测
- 摄像头固定机位效果最佳（书桌侧前方 45°~60°）；换机位后需重新校准
- 板载喇叭需自行接入 es8388 的 speaker 输出口；不接喇叭不影响浏览器发声
- 模型输出后处理为向量化实现，语义对齐官方 airockchip/rknn_model_zoo yolov8_pose demo

## 许可

代码 MIT；YOLOv8n-Pose 基础权重 AGPL-3.0 (上游 Ultralytics)，不随本仓库分发，请自行获取并遵守其许可。
