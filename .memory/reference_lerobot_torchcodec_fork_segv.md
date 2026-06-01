---
name: reference-lerobot-torchcodec-fork-segv
description: lerobot DataLoader worker SEGV (~step 24k) 根因是 torchcodec 解码器 cache 被 fork 继承，PID-aware cache 修复
metadata: 
  node_type: memory
  type: reference
  originSessionId: 7a08b045-ec5e-41e0-90e1-3c36231eee63
---

LeRobot v0.5.2 训练（ACT/DP/任意带 video 的数据集）开 `num_workers>0` 时，DataLoader worker 在 **第一个 epoch 边界（≈step 24k @ batch8/184k帧）reproducibly SIGSEGV**，报 `DataLoader worker (pid xxx) is killed by signal: Segmentation fault`。

**根因**：`lerobot/datasets/video_utils.py` 有 module 全局 `_default_decoder_cache`（`VideoDecoderCache`），存 `{video_path: (torchcodec.VideoDecoder, fsspec file_handle)}`。DataLoader fork worker 时，子进程继承这个 dict 的副本，里面的 decoder/handle 指向**父进程的 ffmpeg/fd 状态**，子进程复用即 UB → 段错误。`persistent_workers=False` 时每 epoch 重新 fork worker，所以崩在 epoch 边界。dataset_reader.py:234 docstring 自己警告过这点。

**踩过的弯路**（都没用）：
- `num_workers=2` + `persistent_workers=False`：仍崩（fork 继承问题与 worker 数无关）
- `video_backend=pyav`：不崩但 **56× 慢**（3.2s/step vs 0.057s）—— CPU 单帧 seek 解码，不可用
- `num_workers=0`：不崩（主进程单一 cache 无 fork）但单线程解码 data_s=0.21s >> updt_s=0.032s，**~6× 慢**

**正确修法** —— PID-aware decoder cache（`robocasa-training/scripts/pid_safe_decoder_patch.py`）：
子类 `VideoDecoderCache`，记 `_owner_pid=os.getpid()`；`get_decoder` 里若 `os.getpid() != _owner_pid` 说明在新 fork 的 worker，**丢弃继承的 cache（不 close handle，close 会破坏父进程共享 fd）+ 换新 Lock + 重建 decoder**。launcher 启动时 `vu._default_decoder_cache = PidSafeDecoderCache()` 替换全局即可（`decode_video_frames_torchcodec` 调用时读 module 全局，无需 patch 函数本身）。

**验证**：3-episode 小数据集 probe，num_workers=4 + persistent_workers=False 迭代 5000 batch 跨 57 次 worker 重生，**零 SEGV**，39ms/batch。真实训练 data_s 0.211→0.020（10×），step 4→18 step/s。

**结论**：torchcodec + fork 的坑，PID-aware cache 是干净修法，比 LeIsaac 的 mp4→.npy precache（要 108GB 盘）更省。相关 [[feedback-train-with-watcher]]（分 slice + watcher）。
