#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bug_force.py — “加上力”版：用一个超临界交滑移力，测出 FCC 交滑移的“弯臂漏检”bug。

与同目录的 test_bug.py 的区别
-----------------------------
  - test_bug.py  ：节点力为 0，只检验上一层“螺型链是否被正确构建”(build_screw_chains)，
                   靠 stderr 的 [OLD-CS]/[NEW-CS] 调试行数链。它证明不了“交滑移到底有没有发生”。
  - 本脚本       ：给螺型 run 施加一个【足够大的交滑移驱动力】，让交滑移真正“执行”，
                   然后直接看网络有没有变——以此把 bug 测出来。

构型（与 bug.h / test_bug.py 完全一致）
--------------------------------------
一条 ~900 b 的位错臂：前 5 段是沿 1/2[1 0 -1] 的理想螺型、共面于 (111)，末端 2 段拐弯成非螺型。

bug 的本质
----------
  旧实现 cross_slip_fcc_thermal.h：用整条臂“首尾弦向量”判螺型。弯臂弦向偏离 b ~38°(>15°容差)
    → 整条臂被判为非螺型而整体丢弃 → build_screw_chains 返回 0 条链 → handle_bulk 永远不会被调用。
    后果：**无论施加多大的交滑移力，旧实现都不会执行交滑移**（力被白白浪费）。← 这就是 bug

  新实现 cross_slip_fcc_test.h：沿臂切出“最长连续螺型+同面+同向”run（前 5 段）→ 1 条链 →
    在本脚本的超临界力下 handle_bulk 通过应力阈值并触发交滑移 → 前 5 段的滑移面从 (111)
    翻到交滑移面 (1 -1 1)。← 修复后

  wansheng 实现 cross_slip_fcc_wansheng.h：思路与新实现一致——逐 segment 判螺型、切出极大
    连续螺型 run（前 5 段，>= minChainSegments=4）→ 1 条链 → handle_bulk 在同一条超临界力下
    触发交滑移 → 前 5 段滑移面翻到 (1 -1 1)。本脚本的第 [3] 段就是为覆盖 wansheng 的
    “执行”路径（应力阈值/热激活/翻面）而加，test_bug.py 只覆盖了它的建链层。

怎么“测出来”
------------
交滑移一旦执行，execute_crossslip 会把 run 上每个 segment 的滑移面改成交滑移面（update_seg_plane）。
所以我们在 handle 前后对比【每个 segment 的滑移面法向】：
  - 旧实现：没有链 → 没有交滑移 → 7 段滑移面全保持 (111)，**0 段改变**  → 复现 bug
  - 新实现：螺型 run 交滑移 → 第 0..4 段滑移面变成 (1 -1 1)，**5 段改变**  → bug 已修复
（注：这条 run 本来就是一条笔直的理想螺型，节点已落在螺型线上，所以投影后节点几乎不动——
  能观测到的唯一变化就是“滑移面翻面”，这正是判据。）

为什么这个力一定能触发交滑移
----------------------------
handle_bulk 的三道关卡（Hussein et al. 2015, Sec.2）：
  1. |tau_cs|  >= MU*b/(10*L)              —— 交滑移面 Schmid 应力够大
  2. |tau_cs|  >= 1.1*|tau_glide|          —— 比原滑移面大 10% 以上（防止来回横跳）
  3. 热激活概率 P = nu*dt*(L/Lref)*exp(-(Ea - Va*dSigma_E)/(kB T)) 抽样命中
我们把每个节点的力都设成 +MAG * cs_line（cs_line=交滑移面内的刃方向），于是
  tau_cs    = |F·cs_line|/L 很大；tau_glide = |F·glide_line|/L = 0.33*tau_cs（两个{111}刃方向夹角~70.5°）
  → 关卡 1、2 轻松通过。再把 Ea=0、Va=0 → exp(...)=1，P=nu*dt*(L/Lref)≈2.5e5 >> 1 → 关卡 3 必中。
这样就把“能不能触发交滑移”完全归结为“螺型链有没有被建出来”——也就单刀直入地测到了 bug。

用法
----
    cd Opendis_q/a_test_cross_slip/test_bug
    python3 test_bug_force.py
注意：CrossSlipFCCTest 是新加进绑定的，必须先在 Linux 上重新编译 pyexadis 才能用：
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
#  在 C/C++ 层捕获 fprintf(stderr, ...) 的输出（[OLD-CS]/[NEW-CS] 调试行）
#  Python 的 sys.stderr 重定向抓不到 C 层直接写 fd=2 的内容，必须做 fd 级重定向。
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def capture_c_stderr(sink):
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


# ---------------------------------------------------------------------------
#  几何：Burgers / 滑移面 / 交滑移面 / 力的方向，全部与 C++ 内部算法一致
# ---------------------------------------------------------------------------
B_VEC      = np.array([0.5, 0.0, -0.5])           # 1/2[1 0 -1]
GLIDE_PLANE = np.array([1.0, 1.0, 1.0])           # (111)


def get_crossslip_plane(glide_plane, burg):
    """复刻 C++ get_crossslip_plane：b 哪个分量为 0，就把面法向那个分量翻号。"""
    tol = 1e-6
    bn = burg / np.linalg.norm(burg)
    g = glide_plane / np.linalg.norm(glide_plane)
    if abs(np.dot(bn, g)) > tol:
        return np.zeros(3)
    if abs(bn[0]) < tol: return np.array([-g[0],  g[1],  g[2]])
    if abs(bn[1]) < tol: return np.array([ g[0], -g[1],  g[2]])
    if abs(bn[2]) < tol: return np.array([ g[0],  g[1], -g[2]])
    return np.zeros(3)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_bent_arm_network():
    """复刻 bug.h 的弯臂网络：8 节点、7 段，b=1/2[1 0 -1]，plane=(111)。"""
    L = 1.0e5                                       # b 单位的大盒子，避免 ~900 b 的臂被 PBC 折叠
    cell = pyexadis.Cell(h=L * np.eye(3), is_periodic=[1, 1, 1])

    b = B_VEC.copy()
    plane = unit(GLIDE_PLANE)

    P = np.array([
        [  0.000,   0.000,    0.000],
        [ 70.711,   0.000,  -70.711],
        [141.421,   0.000, -141.421],
        [212.132,   0.000, -212.132],
        [282.843,   0.000, -282.843],
        [353.553,   0.000, -353.553],               # 0..5：沿 b 的理想螺型（5 段，每段 ~100 b，共 ~500 b）
        [311.018, 230.179, -541.196],               # 5..6：拐弯，非螺型
        [268.482, 460.357, -728.839],               # 6..7：拐弯，非螺型
    ])
    P = P + np.array([L / 2, L / 2, L / 2])          # 平移到盒子中心，彻底避开周期边界折叠歧义（平移不变）

    nodes = [np.concatenate((P[i], [NodeConstraints.UNCONSTRAINED])) for i in range(len(P))]
    segs = [np.concatenate(([i, i + 1], b, plane)) for i in range(len(P) - 1)]
    return ExaDisNet(cell, np.array(nodes), np.array(segs))


def apply_crossslip_force(G, mag):
    """给所有节点施加 +mag * cs_line 的力（cs_line = 交滑移面内刃方向）。

    这样合力强烈投影在交滑移面刃方向上，使 |tau_cs| 远超阈值、且 > 1.1|tau_glide|。
    """
    cs_plane = get_crossslip_plane(GLIDE_PLANE, B_VEC)        # (1 -1 1)
    cs_line  = unit(np.cross(B_VEC, cs_plane))                # 交滑移面内的刃方向
    N = G.num_nodes()
    forces = np.tile(mag * cs_line, (N, 1))
    tags = G.get_tags()
    G.net.set_forces(forces, tags)
    return cs_plane, cs_line


def count_changed_planes(planes_before, planes_after):
    """统计滑移面法向方向发生改变的 segment 个数（对符号/归一化稳健）。"""
    changed = []
    for i in range(len(planes_before)):
        a = unit(planes_before[i])
        b = unit(planes_after[i])
        cosang = abs(float(np.dot(a, b)))
        if cosang < 0.99:                                    # 偏离原方向 > ~8° 即认为翻面了
            changed.append(i)
    return changed


def run_handler_with_force(make_name, params, force, cs_params, mag):
    """新建网络→施力→建 handler→执行 handle_cross_slip，返回 (ok, 改变的段, cs_plane, stderr日志)。"""
    if not hasattr(pyexadis, make_name):
        return False, None, None, ''

    G = build_bent_arm_network()
    planes_before = G.get_segs_data()["planes"].copy()       # handle 前：应全是 (111)
    cs_plane, _ = apply_crossslip_force(G, mag)

    make_fn = getattr(pyexadis, make_name)
    cs = make_fn(params=params, force=force, fcc_params=cs_params)

    log = []
    with capture_c_stderr(log):
        pyexadis.handle_cross_slip(net=G.net, cross_slip=cs)

    planes_after = G.get_segs_data()["planes"].copy()        # handle 后：变了的段=交滑移执行过的段
    changed = count_changed_planes(planes_before, planes_after)
    return True, (changed, planes_after), cs_plane, log[0]


def make_deterministic_params(cls_params):
    """把热激活参数调成“必发生”：Ea=0、Va=0 → exp=1，P=nu*dt*(L/Lref) >> 1。
    同时设 screwAngleTolerance=15（与 bug 构型一致），新实现额外设 minRunLength=0。
    """
    cls_params.screwAngleTolerance   = 15.0
    cls_params.bulkActivationEnergy  = 0.0       # 去掉势垒
    cls_params.bulkActivationVolume  = 0.0       # 去掉 Escaig 依赖：exp(...)=1
    cls_params.bulkAttemptFrequency  = 5.0e17    # 默认；nu*dt*(L/Lref) ≈ 2.5e5 >> 1
    cls_params.bulkReferenceLength   = 1000.0
    cls_params.evalFrequency         = 1
    cls_params.temperature           = 300.0
    return cls_params


def main():
    pyexadis.initialize()
    try:
        # --- 最小 FCC 系统：默认取向 R = 单位阵 → Rinv = 单位阵（与 bug.h 一致）---
        state = {
            "crystal": "fcc",
            "burgmag": 2.49e-10,
            "mu": 76.0e9,
            "nu": 0.31,
            "a": 2.0,
            "maxseg": 2000.0,
            "minseg": 10.0,
            "rann": 1.0,
            "rtol": 2.0,
            "nextdt": 1e-12,
            "maxdt": 1e-10,
            "use_glide_planes": True,                # handle() 要求 use_glide_planes=true，否则直接 fatal
        }

        params = get_exadis_params(state)
        calforce = CalForce(state=state, force_mode='LineTension')   # handle 内部不重算力，仅为构造接口占位
        force, _ = get_exadis_force(calforce, state, params)

        # --- 施加的力：每节点 MAG*cs_line。MAG 远大于触发阈值所需的最小值 ---
        MAG = 1.0e4                                  # 远超临界：tau_cs/threshold ~ 2.6e4（见下方打印）
        L_run = 500.0                                # 螺型 run 总长 ~500 b（5 段×100 b）
        threshold = state["mu"] * state["burgmag"] / (10.0 * L_run)
        tau_cs_pred = 5.0 * MAG / L_run              # 合力在 6 节点端点半权下 = 5*MAG；投影到 cs_line
        print("施力方案：每节点 F = %.1e * cs_line(交滑移面刃方向)" % MAG)
        print("  预估 |tau_cs|   ≈ %.4g" % tau_cs_pred)
        print("  交滑移阈值      = MU*b/(10*L) ≈ %.4g   →  余量 ~%.0f 倍" %
              (threshold, tau_cs_pred / threshold))
        print("  热激活：Ea=0, Va=0 → P = nu*dt*(L/Lref) ≈ %.2g  (>>1，必触发)\n"
              % (5.0e17 * state["nextdt"] * (L_run / 1000.0)))

        # ============ 旧实现（对照，应复现 bug：施力后仍 0 段交滑移）============
        print("=" * 72)
        print("[1] 旧实现 CrossSlipFCCThermal —— 期望：施了超临界力，仍然 0 段交滑移（bug）")
        print("=" * 72)
        cs_p_old = make_deterministic_params(pyexadis.CrossSlipFCCThermal_Params())
        ok_old, res_old, csp_old, log_old = run_handler_with_force(
            'make_cross_slip_fcc_thermal', params, force, cs_p_old, MAG)
        if ok_old:
            sys.stdout.write(log_old)
            changed_old = res_old[0]
            print(">>> 旧实现：滑移面发生改变的 segment 数 = %d  %s\n"
                  % (len(changed_old), changed_old if changed_old else "(无)"))
        else:
            changed_old = None
            print(">>> 绑定里没有 make_cross_slip_fcc_thermal（旧 .so？）跳过\n")

        # ============ 新实现（被测，应修复：施力后 5 段交滑移）============
        print("=" * 72)
        print("[2] 新实现 CrossSlipFCCTest —— 期望：螺型 run(第0..4段)交滑移，5 段翻面")
        print("=" * 72)
        if hasattr(pyexadis, 'make_cross_slip_fcc_test'):
            cs_p_new = make_deterministic_params(pyexadis.CrossSlipFCCTest_Params())
            cs_p_new.minRunLength = 0.0
            ok_new, res_new, csp_new, log_new = run_handler_with_force(
                'make_cross_slip_fcc_test', params, force, cs_p_new, MAG)
            sys.stdout.write(log_new)
            changed_new, planes_after_new = res_new
            print(">>> 新实现：滑移面发生改变的 segment 数 = %d  %s"
                  % (len(changed_new), changed_new if changed_new else "(无)"))
            if changed_new:
                cs_hat = unit(csp_new)
                print("    交滑移目标面 (1 -1 1)/√3 = %s" % np.round(cs_hat, 3))
                for i in changed_new:
                    print("      seg %d 新滑移面法向 = %s" % (i, np.round(unit(planes_after_new[i]), 3)))
            print()
        else:
            changed_new = None
            print(">>> 绑定里没有 make_cross_slip_fcc_test。")
            print(">>> 请先重新编译并安装 pyexadis（见文件顶部用法说明），再运行本测试。\n")

        # ============ wansheng 方案（被测，应修复：施力后 5 段交滑移）============
        print("=" * 72)
        print("[3] wansheng 方案 CrossSlipFCCWansheng —— 期望：螺型 run(第0..4段)交滑移，5 段翻面")
        print("=" * 72)
        if hasattr(pyexadis, 'make_cross_slip_fcc_wansheng'):
            cs_p_ws = make_deterministic_params(pyexadis.CrossSlipFCCWansheng_Params())
            cs_p_ws.minChainSegments = 4         # wansheng 用 minChainSegments(默认4)，无 minRunLength 字段
            ok_ws, res_ws, csp_ws, log_ws = run_handler_with_force(
                'make_cross_slip_fcc_wansheng', params, force, cs_p_ws, MAG)
            sys.stdout.write(log_ws)
            changed_ws, planes_after_ws = res_ws
            print(">>> wansheng：滑移面发生改变的 segment 数 = %d  %s"
                  % (len(changed_ws), changed_ws if changed_ws else "(无)"))
            if changed_ws:
                cs_hat = unit(csp_ws)
                print("    交滑移目标面 (1 -1 1)/√3 = %s" % np.round(cs_hat, 3))
                for i in changed_ws:
                    print("      seg %d 新滑移面法向 = %s" % (i, np.round(unit(planes_after_ws[i]), 3)))
            print()
        else:
            changed_ws = None
            log_ws = ''
            print(">>> 绑定里没有 make_cross_slip_fcc_wansheng。")
            print(">>> 请先接入 cross_slip_fcc_wansheng.h 并重新编译安装 pyexadis（见文件顶部用法说明），再运行本测试。\n")

        # ============ 判定 ============
        print("=" * 72)
        print("结论")
        print("=" * 72)
        if changed_old is not None:
            tag = "✓ 复现 bug（施了超临界力也没交滑移：链被整条丢弃）" if len(changed_old) == 0 \
                  else "✗ 未复现（旧实现竟发生了交滑移，构型/参数与预期不符）"
            print("  旧实现:   %d 段交滑移   %s" % (len(changed_old), tag))

        if changed_ws is None:
            print("  wansheng: 未测（pyexadis 未接入 wansheng 绑定）")
        elif len(changed_ws) >= 1:
            print("  wansheng: %d 段交滑移   ✓ 螺型 run 在力驱动下正确交滑移（滑移面翻到交滑移面）"
                  % len(changed_ws))
        else:
            print("  wansheng: %d 段交滑移   ✗ 仍未触发（力/参数未达阈值或链未建出）" % len(changed_ws))

        if changed_new is None:
            print("  新实现:   未测（pyexadis 未重新编译）")
        elif len(changed_new) >= 1:
            print("  新实现:   %d 段交滑移   ✓ 螺型 run 在力驱动下正确交滑移" % len(changed_new))
        else:
            print("  新实现:   %d 段交滑移   ✗ 仍未触发" % len(changed_new))

        # —— 总判定：以 wansheng 为本测试主对象 ——
        if changed_ws is None:
            print("\n  => wansheng 未测，无法判定；请重新编译并接入绑定后再跑。")
        elif len(changed_ws) >= 1:
            print("\n  => PASS：bug 已修复。")
            print("     旧实现把弯臂整条丢弃，超临界力也驱动不了交滑移；")
            print("     wansheng 切出螺型 run，同一条力就把它真正翻了面——力把 bug 测了出来。")
        else:
            print("\n  => FAIL：wansheng 未触发交滑移（bug 未修复，或力/参数未达阈值）。")
            if not log_ws.strip():
                print("     （未捕获到 [WS-CS] 调试行：可能 DEBUG 块已删除，"
                      "或未走到 build_screw_chains——请检查 crystal/use_glide_planes 设置）")
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
