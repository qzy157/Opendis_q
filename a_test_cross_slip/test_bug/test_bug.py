#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bug.py — 验证 FCC 交滑移 build_screw_chains 的“弯臂漏检”bug 是否已修复。

本脚本是 bug.h 的 Python 版复刻。构型完全一致：一条 ~900 b 的位错臂，
前 5 段是沿 1/2[1 0 -1] 的理想螺型、共面于 (111)，随后在末端拐弯变成非螺型。

  - 旧实现  cross_slip_fcc_thermal.h / CrossSlipFCCThermal
      用整条臂“首尾弦向量”判螺型。弯臂使弦向偏离 b 约 38°(> 15° 容差) →
      整条臂被判为非螺型而整体丢弃 → build_screw_chains 返回 0 条链。  ← 这就是 bug

  - 新实现  cross_slip_fcc_test.h / CrossSlipFCCTest
      沿臂切出“最长连续螺型 + 同面 + 同向”的 run。前 5 段构成一条 run →
      build_screw_chains 返回 1 条链。  ← 修复后

由于该测试网络未施加外力，节点力为 0，不会真正“执行”交滑移（应力阈值通不过）；
我们检验的是上一层——“螺型链是否被正确构建”。新旧两个实现的 build_screw_chains
都在 stderr 打了调试行（[OLD-CS] / [NEW-CS]），脚本在 C 层捕获这些行并据此判定。

用法：
    cd Opendis_q/a_test_cross_slip/test_bug
    python3 test_bug.py
注意：CrossSlipFCCTest 是刚加进绑定的，必须先在 Linux 上重新编译 pyexadis 才能用：
    cd Opendis_q && cmake --build build -j8 && cmake --build build --target install
"""

import os
import sys
import contextlib
import tempfile
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, NodeConstraints, get_exadis_params, get_exadis_force, CalForce
except ImportError as e:
    raise ImportError('无法导入 pyexadis（请确认已编译并安装）: %s' % e)


# ---------------------------------------------------------------------------
#  在 C/C++ 层捕获 fprintf(stderr, ...) 的输出
#  （Python 的 sys.stderr 重定向抓不到 C 层直接写 fd=2 的内容，必须做 fd 级重定向）
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def capture_c_stderr(sink):
    """sink: 传入一个 list；with 块退出后捕获到的文本会 append 进去。"""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    tmp = tempfile.TemporaryFile(mode='w+b')
    try:
        sys.stderr.flush()
        os.dup2(tmp.fileno(), stderr_fd)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        tmp.seek(0)
        sink.append(tmp.read().decode('utf-8', errors='replace'))
        tmp.close()


def count_old_chains(log):
    """旧实现 [OLD-CS]：每条物理链一行，结尾 '-> keep' 表示保留，'-> REJECT(whole arm)' 表示整臂丢弃。"""
    return sum(1 for ln in log.splitlines() if '[OLD-CS]' in ln and '-> keep' in ln)


def count_new_chains(log):
    """新实现 [NEW-CS]：每个 run 一行，含 'KEEP-as-chain' 表示该 run 成链。"""
    return sum(1 for ln in log.splitlines() if '[NEW-CS]' in ln and 'KEEP-as-chain' in ln)


def count_ws_chains(log):
    """wansheng 实现 [WS-CS]：每个 run 一行，结尾 '-> keep' 表示该 run 成链。"""
    return sum(1 for ln in log.splitlines() if '[WS-CS]' in ln and '-> keep' in ln)


def build_bent_arm_network():
    """复刻 bug.h 的弯臂网络：8 个节点、7 段，b=1/2[1 0 -1]，plane=(111)。"""
    L = 1.0e5                                   # b 单位的大盒子，避免 ~900 b 的臂被 PBC 折叠
    cell = pyexadis.Cell(h=L * np.eye(3), is_periodic=[1, 1, 1])

    b = np.array([0.5, 0.0, -0.5])              # 1/2[1 0 -1]（未归一，模 ~0.707；建链内部会归一）
    plane = np.array([1.0, 1.0, 1.0])
    plane = plane / np.linalg.norm(plane)       # (111)

    P = np.array([
        [  0.000,   0.000,    0.000],
        [ 70.711,   0.000,  -70.711],
        [141.421,   0.000, -141.421],
        [212.132,   0.000, -212.132],
        [282.843,   0.000, -282.843],
        [353.553,   0.000, -353.553],           # 0..5：沿 b 的理想螺型（5 段）
        [311.018, 230.179, -541.196],           # 5..6：拐弯，非螺型
        [268.482, 460.357, -728.839],           # 6..7：拐弯，非螺型
    ])
    # 整体平移到盒子中心：平移不变，不改变任何相对几何/判链结果，但彻底避开周期边界处的折叠歧义
    P = P + np.array([L / 2, L / 2, L / 2])

    nodes = [np.concatenate((P[i], [NodeConstraints.UNCONSTRAINED])) for i in range(len(P))]
    segs = [np.concatenate(([i, i + 1], b, plane)) for i in range(len(P) - 1)]
    return ExaDisNet(cell, np.array(nodes), np.array(segs))


def run_handler(make_name, params, force, cs_params, G):
    """调用一次交滑移 handler，返回 (是否成功, 捕获到的 stderr 文本)。"""
    if not hasattr(pyexadis, make_name):
        return False, ''
    make_fn = getattr(pyexadis, make_name)
    cs = make_fn(params=params, force=force, fcc_params=cs_params)
    log = []
    with capture_c_stderr(log):
        pyexadis.handle_cross_slip(net=G.net, cross_slip=cs)
    return True, log[0]


def main():
    pyexadis.initialize()
    try:
        # --- 最小 FCC 系统：默认取向 R = 单位阵 → Rinv = 单位阵（与 bug.h 一致）---
        state = {
            "crystal": "fcc",
            "burgmag": 2.49e-10,        # Ni；本测试只建链，不依赖该值
            "mu": 76.0e9,               # 占位
            "nu": 0.31,                 # 占位
            "a": 2.0,
            "maxseg": 2000.0,
            "minseg": 10.0,
            "rann": 1.0,
            "rtol": 2.0,
            "nextdt": 1e-12,
            "maxdt": 1e-10,
            "use_glide_planes": True,   # handle() 要求 use_glide_planes=true，否则直接 fatal
        }

        params = get_exadis_params(state)
        calforce = CalForce(state=state, force_mode='LineTension')   # handler 不实际用力，仅为构造接口占位
        force, _ = get_exadis_force(calforce, state, params)

        G = build_bent_arm_network()
        print("构建测试网络：%d 节点, %d 段（前 5 段螺型 @ (111)，后 2 段弯折）\n"
              % (G.num_nodes(), G.num_segments()))

        # ============ 旧实现（对照，应复现 bug：0 条链）============
        print("=" * 70)
        print("[1] 旧实现 CrossSlipFCCThermal (cross_slip_fcc_thermal.h) —— 期望 0 条链")
        print("=" * 70)
        cs_p_old = pyexadis.CrossSlipFCCThermal_Params()
        cs_p_old.screwAngleTolerance = 15.0     # 旧版 Params 没有 minRunLength 字段
        ok_old, log_old = run_handler('make_cross_slip_fcc_thermal', params, force, cs_p_old, G)
        if ok_old:
            sys.stdout.write(log_old)
            n_old = count_old_chains(log_old)
            print(">>> 旧实现 build_screw_chains 得到 %d 条链\n" % n_old)
        else:
            n_old = None
            print(">>> 绑定里没有 make_cross_slip_fcc_thermal（旧 .so？）跳过\n")

        # ============ 新实现（被测，应修复：>=1 条链）============
        print("=" * 70)
        print("[2] 新实现 CrossSlipFCCTest (cross_slip_fcc_test.h) —— 期望 >=1 条链")
        print("=" * 70)
        if hasattr(pyexadis, 'make_cross_slip_fcc_test'):
            cs_p_new = pyexadis.CrossSlipFCCTest_Params()
            cs_p_new.screwAngleTolerance = 15.0
            cs_p_new.minRunLength = 0.0
            ok_new, log_new = run_handler('make_cross_slip_fcc_test', params, force, cs_p_new, G)
            sys.stdout.write(log_new)
            n_new = count_new_chains(log_new)
            print(">>> 新实现 build_screw_chains 得到 %d 条链\n" % n_new)
        else:
            n_new = None
            print(">>> 绑定里没有 make_cross_slip_fcc_test。")
            print(">>> 请先重新编译并安装 pyexadis（见文件顶部用法说明），再运行本测试。\n")

        # ============ wansheng 方案（被测，应修复：>=1 条链）============
        print("=" * 70)
        print("[3] wansheng 方案 CrossSlipFCCWansheng (cross_slip_fcc_wansheng_standalone.h) —— 期望 >=1 条链")
        print("=" * 70)
        if hasattr(pyexadis, 'make_cross_slip_fcc_wansheng'):
            cs_p_ws = pyexadis.CrossSlipFCCWansheng_Params()
            cs_p_ws.screwAngleTolerance = 15.0
            cs_p_ws.minChainSegments = 4          # 论文 p.4：最少 4 段
            ok_ws, log_ws = run_handler('make_cross_slip_fcc_wansheng', params, force, cs_p_ws, G)
            sys.stdout.write(log_ws)
            n_ws = count_ws_chains(log_ws)
            print(">>> wansheng build_screw_chains 得到 %d 条链\n" % n_ws)
        else:
            n_ws = None
            print(">>> 绑定里没有 make_cross_slip_fcc_wansheng。")
            print(">>> 请先把 cross_slip_fcc_wansheng_standalone.h 接入并重新编译 pyexadis（见文件顶部用法）。\n")

        # ============ 判定 ============
        print("=" * 70)
        print("结论")
        print("=" * 70)
        if n_old is not None:
            tag = "✓ 复现了 bug（整臂被丢弃）" if n_old == 0 else "✗ 未复现（构型或参数与预期不符）"
            print("  旧实现: %d 条链   %s" % (n_old, tag))
        if n_ws is None:
            print("  wansheng: 未测（pyexadis 未接入 wansheng 绑定）")
        elif n_ws >= 1:
            print("  wansheng: %d 条链   \u2713 螺型 run 被正确识别" % n_ws)
        else:
            print("  wansheng: %d 条链   \u2717 仍未识别出螺型 run" % n_ws)
        if n_new is None:
            print("  新实现: 未测（pyexadis 未重新编译）")
            print("\n  => 无法判定，请重新编译后再跑。")
            return
        if n_new >= 1:
            print("  新实现: %d 条链   ✓ 螺型 run 被正确识别" % n_new)
            print("\n  => PASS：bug 已修复（新实现在旧实现漏检的弯臂上找回了螺型链）。")
        else:
            print("  新实现: %d 条链   ✗ 仍未识别出螺型 run" % n_new)
            print("\n  => FAIL：bug 未修复。")
            if not log_new.strip():
                print("     （未捕获到 [NEW-CS] 调试行：可能 DEBUG 块已删除，"
                      "或未走到 build_screw_chains——请检查 crystal/use_glide_planes 设置）")
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
