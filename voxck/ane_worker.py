"""Core ML(ANE) 推理子进程。

为什么要独立进程：实测 coremltools 的 predict 在预测期间约 75% 时间持有 GIL，
放在主进程线程里会让贴回从 1.4ms 变 47ms，也会卡住 asyncio 循环和 MLX 线程。
子进程用 spawn（fork 与 Metal/ObjC 运行时不兼容）。

协议（multiprocessing Pipe，pickle numpy）：
  父 → 子: ("run", xs (n,8,32,32) fp16, pes (n,50,384) fp16)   逐帧预测，每 DEC_BS 帧回传一批
  父 → 子: ("stop",)                                             中断当前 run（子进程每帧检查一次）
  父 → 子: None                                                  退出
  子 → 父: ("batch", ys (b,4,32,32) fp16) ... ("done", n_done, ane_ms)
"""
from __future__ import annotations

import multiprocessing as mp
import statistics
import time

import numpy as np


def _set_qos_user_interactive():
    """把子进程主线程提到 user-interactive QoS：父进程有 CPU 活时 ANE 推理会从 36ms 掉到 47ms，
    原因是 Core ML→aned→ANE 的多次进程间往返对调度延迟敏感。"""
    import ctypes
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.pthread_set_qos_class_self_np(ctypes.c_int(0x21), ctypes.c_int(0))
    except Exception:
        pass


def _serve(conn, pkg: str, batch: int, layout: str = "bsc", qos: bool = True):
    try:
        if qos:
            _set_qos_user_interactive()
        _serve_inner(conn, pkg, batch, layout)
    except BaseException as e:                 # 子进程出错必须告诉父进程，否则父进程会永远等
        try:
            conn.send(("error", repr(e)))
        except Exception:
            pass
        raise


def _serve_inner(conn, pkg: str, batch: int, layout: str):
    import coremltools as ct
    m = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    # layout: "bsc" = diffusers 原版 (B,50,384)；"bc1s" = Apple ANE 结构 (B,384,1,50)
    enc = (lambda p: p) if layout == "bsc" else (lambda p: np.ascontiguousarray(p.transpose(0, 2, 1)[:, :, None, :]))
    z = {"x": np.zeros((1, 8, 32, 32), np.float16), "enc": enc(np.zeros((1, 50, 384), np.float16))}
    for _ in range(3):
        m.predict(z)
    conn.send(("ready",))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if msg[0] != "run":
            continue
        _, xs, pes = msg
        buf, ts, done, stopped = [], [], 0, False
        for i in range(xs.shape[0]):
            if conn.poll():                      # 中断只可能来自父进程
                nxt = conn.recv()
                if nxt is None:
                    return
                if nxt[0] == "stop":
                    stopped = True
                    break
            t = time.perf_counter()
            buf.append(m.predict({"x": xs[i:i + 1], "enc": enc(pes[i:i + 1])})["y"])
            ts.append(time.perf_counter() - t)
            done += 1
            if len(buf) == batch or i == xs.shape[0] - 1:
                conn.send(("batch", np.concatenate(buf).astype(np.float16)))
                buf = []
        conn.send(("done", done, statistics.median(ts) * 1000 if ts else 0.0, stopped))


class AneUnet:
    """父进程句柄。start() 阻塞到模型加载完成。"""

    def __init__(self, pkg: str, batch: int = 4, layout: str = "bsc", qos: bool = True):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, pkg, batch, layout, qos), daemon=True, name="ane-unet")
        self.proc.start()
        child.close()
        msg = self.conn.recv()
        assert msg == ("ready",), msg

    def run(self, xs: np.ndarray, pes: np.ndarray):
        """生成器：逐批产出 (b,4,32,32) fp16；结束时 self.last = (n_done, ane_ms, stopped)。"""
        self.conn.send(("run", xs.astype(np.float16), pes.astype(np.float16)))
        while True:
            msg = self.conn.recv()
            if msg[0] == "batch":
                yield msg[1]
            elif msg[0] == "error":
                raise RuntimeError(f"ANE 子进程出错: {msg[1]}")
            else:
                self.last = msg[1:]
                return

    def stop(self):
        self.conn.send(("stop",))

    def close(self):
        try:
            self.conn.send(None)
        except Exception:
            pass
        self.proc.join(timeout=3)
