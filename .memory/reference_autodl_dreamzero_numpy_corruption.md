---
name: reference_autodl_dreamzero_numpy_corruption
description: AutoDL DreamZero env 的 numpy 1.x/2.x 残留损坏 → pandas/parquet 全崩 → 阻塞训练 loader;修=物理清 numpy 目录重装
metadata: 
  node_type: memory
  type: reference
  originSessionId: 8f5534ca-f3c2-49d7-a247-bf54c9209e6f
---

AutoDL 机器 `ssh -p 32660 root@connect.westd.seetacloud.com`（DreamZero 训练机，conda base env）出现的环境坑，2026-05-31 调通。属 [[project_dreamzero_robocasa_opencabinet]]。

**症状：** 任何 pandas 操作崩，连 `pd.DataFrame({'a':[1,2]})` 都报 `TypeError: Cannot convert numpy.ndarray to numpy.ndarray`（在 `pandas._libs.lib.maybe_convert_objects`）。`pd.read_parquet` 同样崩。**致命影响：DreamZero 训练 loader `groot/vla/data/dataset/lerobot_sharded.py` line 184/471 用 `pd.read_parquet`，所以这不只阻塞 GEAR 转换脚本，而是阻塞整个训练。**

**误判排除：** 不是 pyarrow 版本（24→21 无效）、不是 pandas 版本（2.3.3→2.2.3 无效）、不是数据（fixed_size_list 列正常）。`--force-reinstall numpy` 也无效。

**真根因：** site-packages 里 numpy 是 **2.x 和 1.26.4 的损坏混合**——之前装过 numpy 2.x，后装 1.26.4 时 pip uninstall 只删自己 RECORD 里的文件，残留了 numpy 2.x 的 `numpy/_core/_multiarray_umath.cpython-312*.so`（2.x 布局）。pandas 的 Cython 链到这个残留 2.x C 扩展 → ABI 崩。注意 numpy 1.26.4 **本身就合法带一个 `numpy/_core` 兼容垫片**，所以"存在 _core"不是判据；判据是里面有没有 2.x 的 stale `.so`。

**修法（关键）：物理删除所有 numpy 目录再干净重装**，pip force-reinstall 修不掉孤儿目录：
```
SP=/root/miniconda3/lib/python3.12/site-packages
rm -rf $SP/numpy $SP/numpy.libs $SP/numpy-*.dist-info
pip install --no-cache-dir --no-deps numpy==1.26.4
```
验证：`pd.DataFrame({'a':[1]})` 不报错 + `pd.read_parquet(任一 episode.parquet)` 出 (N,16) state。修复后最终版本：numpy 1.26.4 / pandas 2.2.3 / pyarrow 21.0.0 / datasets 4.8.5（datasets 要 pyarrow>=21，所以别降到 21 以下）。

**通用教训：** "Cannot convert numpy.ndarray to numpy.ndarray" + `np.allclose` 在普通 ndarray 上报 `__array_function__ no implementation` = numpy C 扩展与 Python 包版本错位的签名，几乎总是脏 site-packages 残留，物理清目录而非 pip reinstall。
