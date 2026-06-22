#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_wansheng.py — 纯 Python 复刻，验证 wansheng 方案是否修复 FCC 交滑移“弯臂漏检”bug。

背景
----
test_bug.py / test_bug_force.py 走的是“编译好的 pyexadis 绑定”这条路，只能测
  - make_cross_slip_fcc_thermal  (旧 chord 版, [OLD-CS])
  - make_cross_slip_fcc_test      (test.h 方案, [NEW-CS] KEEP-as-chain)
而 wansheng.h 复用了类名 CrossSlipFCCThermal / 绑定 make_cross_slip_fcc_thermal，
无法和真正的旧版同时编进同一个 .so 做对照（见文末说明）。

所以这里不依赖 pyexadis，把三种 build_screw_chains 的“建链逻辑”原样移植到 Python，
喂同一条 bug 构型（与 test_bug.py 完全一致：一条 ~900 b 的位错臂，前 5 段沿 1/2[1 0 -1]
理想螺型且共面 (111)，末端 2 段拐弯成非螺型），直接对比三者各自建出几条链。

  期望:  旧 chord 版 -> 0 条 (复现 bug)    wansheng -> >=1 条    test -> >=1 条

局限: 这是“算法层”验证(R=I, 力=0, 只看建链)。C++ 真实集成测试仍建议在编译后用
      test_bug.py / test_bug_force.py 跑(见文末把 wansheng 接入的办法)。

用法:
    python3 test_wansheng.py
"""

import numpy as np

DEG = np.pi / 180.0
SCOS = np.cos(15.0 * DEG)          # 螺型角容差余弦, 与三份 .h 的 screwAngleTolerance=15 一致
TOL  = 1e-6


# ---------------------------------------------------------------------------
#  FCC 晶族判据 (与三份 .h 完全一致; 本测试 R=I 所以晶体系=实验室系)
# ---------------------------------------------------------------------------
def _u(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def is_110_family(v, tol=0.1):
    u = np.abs(_u(v)); ax, ay, az = u
    if ax < tol and abs(ay - az) < tol and ay > tol: return True
    if ay < tol and abs(ax - az) < tol and ax > tol: return True
    if az < tol and abs(ax - ay) < tol and ax > tol: return True
    return False

def is_111_family(v, tol=0.1):
    u = np.abs(_u(v)); ax, ay, az = u
    return abs(ax - ay) < tol and abs(ay - az) < tol and ax > tol


# ---------------------------------------------------------------------------
#  bug 构型 (复刻 test_bug.py 的 build_bent_arm_network, 但只用 numpy)
#  8 节点 / 7 段;  b = 1/2[1 0 -1];  plane = (111);  前 5 段螺型, 后 2 段拐弯
# ---------------------------------------------------------------------------
def build_bent_arm():
    P = np.array([
        [  0.000,   0.000,    0.000],
        [ 70.711,   0.000,  -70.711],
        [141.421,   0.000, -141.421],
        [212.132,   0.000, -212.132],
        [282.843,   0.000, -282.843],
        [353.553,   0.000, -353.553],   # 0..5: 沿 b 的理想螺型 (5 段)
        [311.018, 230.179, -541.196],   # 5..6: 拐弯, 非螺型
        [268.482, 460.357, -728.839],   # 6..7: 拐弯, 非螺型
    ])
    b     = np.array([0.5, 0.0, -0.5])          # 1/2[1 0 -1]
    plane = _u(np.array([1.0, 1.0, 1.0]))       # (111)
    snodes = list(range(len(P)))                # 物理臂: 节点 0..7
    ssegs  = list(range(len(P) - 1))            # 段 0..6, 段 k 连 (k, k+1)
    seg_b     = [b.copy()     for _ in ssegs]
    seg_plane = [plane.copy() for _ in ssegs]
    return P, snodes, ssegs, seg_b, seg_plane, b, plane


def seg_dir(P, snodes, k):
    """段 k 的方向向量 (无 PBC: 测试盒子极大, 不会折叠)。"""
    d = P[snodes[k + 1]] - P[snodes[k]]
    return d, np.linalg.norm(d)


# ===========================================================================
#  实现 1: 旧 chord 版  (cross_slip_fcc_thermal.h)
#  用整条臂首尾弦判螺型; 弯臂弦角 ~38° > 15° -> 整臂丢弃
# ===========================================================================
def build_old_chord(P, snodes, ssegs, b, plane, verbose=True):
    nseg = len(ssegs)
    bhat = _u(b)
    # 整臂弦
    chord = P[snodes[-1]] - P[snodes[0]]
    clen  = np.linalg.norm(chord)
    screw_alignment = abs(np.dot(chord / clen, bhat)) if clen > TOL else 0.0
    # 逐段统计非螺型数
    non_screw = 0
    amin, amax = 1e9, -1e9
    for k in range(nseg):
        d, dl = seg_dir(P, snodes, k)
        if dl < TOL:
            continue
        a = np.degrees(np.arccos(min(1.0, abs(np.dot(d / dl, bhat)))))
        amin, amax = min(amin, a), max(amax, a)
        if abs(np.dot(d / dl, bhat)) < SCOS:
            non_screw += 1
    chord_ang = np.degrees(np.arccos(min(1.0, screw_alignment)))
    reject = (screw_alignment < SCOS) or (non_screw > nseg // 2)
    if verbose:
        print("[OLD-CS] link=0 nseg=%d chord=%.1f seg_ang=[%.1f,%.1f] non_screw=%d/%d -> %s"
              % (nseg, chord_ang, amin, amax, non_screw, nseg,
                 "REJECT(whole arm)" if reject else "keep"))
    return 0 if reject else 1


# ===========================================================================
#  实现 2: wansheng 方案  (cross_slip_fcc_wansheng.h)
#  臂级 consistent_plane + FCC 过滤后, 逐段标 is_screw, 取极大连续螺型 run,
#  run 段数 >= minChainSegments(=4) 即成链。
# ===========================================================================
def build_wansheng(P, snodes, ssegs, seg_plane, b, min_chain_segments=4, verbose=True):
    nseg = len(ssegs)
    bhat = _u(b)

    # 臂级一致面检查 (wansheng 保留了原实现这一步)
    plane0 = _u(seg_plane[0])
    for k in range(nseg):
        p = _u(seg_plane[k])
        if np.linalg.norm(p - plane0) > TOL and np.linalg.norm(p + plane0) > TOL:
            if verbose: print("[WS-CS] link=0 inconsistent plane -> skip whole arm")
            return 0
    # FCC 过滤 (R=I)
    if not is_110_family(b) or not is_111_family(plane0):
        if verbose: print("[WS-CS] link=0 FCC filter fail -> skip")
        return 0

    # 逐段标 is_screw
    is_screw = []
    for k in range(nseg):
        d, dl = seg_dir(P, snodes, k)
        is_screw.append(dl > TOL and abs(np.dot(d / dl, bhat)) >= SCOS)

    # 取极大连续螺型 run
    kept = 0
    kpos = 0
    while kpos < nseg:
        if not is_screw[kpos]:
            kpos += 1
            continue
        kstart = kpos
        while kpos < nseg and is_screw[kpos]:
            kpos += 1
        kend = kpos                       # run = 段 [kstart, kend)
        sub_nseg = kend - kstart
        amin, amax = 1e9, -1e9
        for q in range(kstart, kend):
            d, dl = seg_dir(P, snodes, q)
            if dl < TOL: continue
            a = np.degrees(np.arccos(min(1.0, abs(np.dot(d / dl, bhat)))))
            amin, amax = min(amin, a), max(amax, a)
        keep = sub_nseg >= min_chain_segments
        if verbose:
            print("[WS-CS] link=0 run=[%d,%d) sub_nseg=%d seg_ang=[%.1f,%.1f] -> %s"
                  % (kstart, kend, sub_nseg, amin, amax,
                     "keep" if keep else "drop(<min)"))
        if keep:
            kept += 1
    return kept


# ===========================================================================
#  实现 3: test.h 方案  (cross_slip_fcc_test.h) —— 交叉对照
#  逐段螺型 + 同面 + 同向(sense) 的极大 run; 每 run 查 {111}; run_len >= minRunLength(=0)
# ===========================================================================
def build_test(P, snodes, ssegs, seg_plane, b, min_run_len=0.0, verbose=True):
    nseg = len(ssegs)
    bhat = _u(b)
    if not is_110_family(b):
        if verbose: print("[NEW-CS] link=0 b not <110> -> skip")
        return 0

    kept = 0
    k = 0
    while k < nseg:
        d, dl = seg_dir(P, snodes, k)
        pk = seg_plane[k]
        screw_k = dl > TOL and abs(np.dot(d / dl, bhat)) >= SCOS
        if not screw_k or np.linalg.norm(pk) < TOL:
            k += 1
            continue
        rstart, rend = k, k
        run_plane = _u(pk)
        run_sense = 1.0 if np.dot(d / dl, bhat) >= 0.0 else -1.0
        j = k + 1
        while j < nseg:
            dj, sj = seg_dir(P, snodes, j)
            if sj < TOL: break
            screw_j = abs(np.dot(dj / sj, bhat)) >= SCOS
            pj = seg_plane[j]
            coplanar_j = np.linalg.norm(pj) >= TOL and (
                np.linalg.norm(_u(pj) - run_plane) < TOL or
                np.linalg.norm(_u(pj) + run_plane) < TOL)
            sense_j = ((1.0 if np.dot(dj / sj, bhat) >= 0.0 else -1.0) == run_sense)
            if screw_j and coplanar_j and sense_j:
                rend = j; j += 1
            else:
                break
        run_len = sum(seg_dir(P, snodes, q)[1] for q in range(rstart, rend + 1))
        amin, amax = 1e9, -1e9
        for q in range(rstart, rend + 1):
            dq, sq = seg_dir(P, snodes, q)
            if sq < TOL: continue
            a = np.degrees(np.arccos(min(1.0, abs(np.dot(dq / sq, bhat)))))
            amin, amax = min(amin, a), max(amax, a)
        keep = is_111_family(run_plane) and run_len >= min_run_len
        if verbose:
            print("[NEW-CS] link=0 run[seg %d..%d] (%d segs) seg_ang=[%.1f,%.1f] "
                  "run_len=%.1f {111}=%d -> %s"
                  % (rstart, rend, rend - rstart + 1, amin, amax, run_len,
                     int(is_111_family(run_plane)),
                     "KEEP-as-chain" if keep else "drop"))
        if keep:
            kept += 1
        k = rend + 1
    return kept


def main():
    P, snodes, ssegs, seg_b, seg_plane, b, plane = build_bent_arm()
    print("构建测试网络: %d 节点, %d 段 (前 5 段螺型 @ (111), 后 2 段弯折)" % (len(P), len(ssegs)))
    print("b = 1/2[1 0 -1],  plane = (111),  R = I (晶体系=实验室系)\n")

    print("=" * 72)
    print("[1] 旧实现 (cross_slip_fcc_thermal.h, 整臂弦) —— 期望 0 条链")
    print("=" * 72)
    n_old = build_old_chord(P, snodes, ssegs, b, plane)
    print(">>> 旧实现建出 %d 条链\n" % n_old)

    print("=" * 72)
    print("[2] wansheng 方案 (cross_slip_fcc_wansheng.h, 极大螺型 run + 最少4段) —— 期望 >=1 条")
    print("=" * 72)
    n_ws = build_wansheng(P, snodes, ssegs, seg_plane, b, min_chain_segments=4)
    print(">>> wansheng 建出 %d 条链\n" % n_ws)

    print("=" * 72)
    print("[3] test.h 方案 (cross_slip_fcc_test.h, 螺型+同面+同向 run) —— 交叉对照, 期望 >=1 条")
    print("=" * 72)
    n_test = build_test(P, snodes, ssegs, seg_plane, b, min_run_len=0.0)
    print(">>> test.h 建出 %d 条链\n" % n_test)

    print("=" * 72)
    print("结论")
    print("=" * 72)
    print("  旧 chord 版 : %d 条   %s" % (n_old, "✓ 复现 bug(整臂被丢弃)" if n_old == 0 else "✗ 未复现"))
    print("  wansheng    : %d 条   %s" % (n_ws,  "✓ 螺型 run 被找回"      if n_ws  >= 1 else "✗ 仍漏检"))
    print("  test.h      : %d 条   %s" % (n_test,"✓ 螺型 run 被找回"      if n_test>= 1 else "✗ 仍漏检"))
    ok = (n_old == 0 and n_ws >= 1)
    print("\n  => %s" % ("PASS: wansheng 在旧版漏检的弯臂上找回了螺型链, bug 已修复(算法层)。"
                          if ok else "FAIL: 见上方逐链调试行。"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
