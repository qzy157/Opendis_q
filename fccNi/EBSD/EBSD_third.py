# -*- coding: utf-8 -*-
"""
由立方单晶长方体两个相邻面的 EBSD 欧拉角，计算第三个面的晶面指数和法向。

用法：修改下方“输入”部分的欧拉角，然后运行 python third_face_from_euler.py

约定：
  - 欧拉角为 Bunge 约定 (phi1, Phi, phi2)，单位为度。
  - 晶体坐标 = g * 样品坐标，因此 g 的第 1、2、3 列分别是
    样品 X、Y、Z 轴在晶体坐标中的方向，EBSD 面法向取样品 Z 轴（第 3 列）。
  - 三个面互相垂直，面 3 的法向 = 面 1 法向 x 面 2 法向（右手系）。

说明：
  两个面分别测量时，指数不一定在同一晶体坐标系下。程序保持面 1 不动，
  遍历面 2 的 24 种立方对称等价形式，选出与面 1 最接近垂直的一组。
  如果多组同样接近垂直，再选面 2 法向最接近面 1 样品 X/Y 轴的一组。
"""

import itertools
from functools import reduce
from math import gcd

import numpy as np

# ==================== 输入 ====================
EULER_1 = (209.49, 52.51,46.36)   # 面 1 的欧拉角 (phi1, Phi, phi2)，单位：度
EULER_2 = (322.4, 44.91, 89.56)    # 面 2 的欧拉角 (phi1, Phi, phi2)，单位：度

MAX_INDEX = 3            # 标准指数中 |h|、|k|、|l| 的最大值
SHOW_CANDIDATES = True   # 是否列出所有接近垂直的候选解
# ==============================================

TIE_TOLERANCE = 0.1      # 垂直度偏差相差小于此值（度）时视为并列
PERP_WARNING = 6.0       # 候选解的垂直度偏差上限（度）


def bunge_matrix(phi1, Phi, phi2):
    """返回 Bunge 取向矩阵 g（晶体坐标 = g * 样品坐标）。"""
    p1, P, p2 = np.radians([phi1, Phi, phi2])
    c1, s1 = np.cos(p1), np.sin(p1)
    c, s = np.cos(P), np.sin(P)
    c2, s2 = np.cos(p2), np.sin(p2)
    return np.array([
        [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
        [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
        [s1 * s, -c1 * s, c],
    ])


def cubic_symmetry_ops():
    """返回立方晶系的 24 个旋转操作。"""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            op = np.zeros((3, 3))
            for i in range(3):
                op[i, perm[i]] = signs[i]
            if np.linalg.det(op) > 0:
                ops.append(op)
    return ops


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def axis_angle_deg(a, b):
    """两条轴线的夹角（不区分正反方向），范围 0~90 度。"""
    return float(np.degrees(np.arccos(np.clip(abs(unit(a) @ unit(b)), 0.0, 1.0))))


def reduce_indices(v):
    v = [int(round(x)) for x in v]
    divisor = reduce(gcd, [abs(x) for x in v])
    return tuple(x // divisor for x in v) if divisor else tuple(v)


def nearest_indices(n, max_index):
    """返回最接近法向 n 的整数指数；相等时取低指数。"""
    candidates = {
        reduce_indices(hkl)
        for hkl in itertools.product(range(-max_index, max_index + 1), repeat=3)
        if any(hkl)
    }
    return min(candidates, key=lambda hkl: (round(angle_deg(n, hkl), 8), sum(x * x for x in hkl)))


def find_candidates(n1, n2, x1, y1):
    """返回面 2 的所有等价形式，按与面 1 的垂直度排序（去除重复）。"""
    candidates = []
    seen = set()
    for op in cubic_symmetry_ops():
        v = op @ n2
        key = tuple(np.round(v, 8) + 0.0)
        if key in seen:
            continue
        seen.add(key)
        if np.linalg.norm(np.cross(n1, v)) < 1e-8:
            continue  # 与面 1 平行，无法叉乘，也不可能是相邻面
        angle = angle_deg(n1, v)
        candidates.append({
            "n2": v,
            "n3": unit(np.cross(n1, v)),
            "angle": angle,
            "perp_dev": abs(angle - 90.0),
            "axis_dev": min(axis_angle_deg(v, x1), axis_angle_deg(v, y1)),
        })
    candidates.sort(key=lambda c: c["perp_dev"])
    return candidates


def choose_best(candidates):
    smallest = candidates[0]["perp_dev"]
    tied = [c for c in candidates if c["perp_dev"] - smallest <= TIE_TOLERANCE]
    return min(tied, key=lambda c: c["axis_dev"])


def solve(euler_1, euler_2, max_index=3):
    g1 = bunge_matrix(*euler_1)
    g2 = bunge_matrix(*euler_2)
    n1 = g1[:, 2]
    candidates = find_candidates(n1, g2[:, 2], g1[:, 0], g1[:, 1])
    best = choose_best(candidates)
    n2, n3 = best["n2"], best["n3"]

    hkl1 = nearest_indices(n1, max_index)
    hkl2 = nearest_indices(n2, max_index)
    warnings = []
    if np.dot(hkl1, hkl2) == 0:
        # 标准指数互相垂直时，由叉乘得到面 3，保证三者构成右手系
        hkl3 = reduce_indices(np.cross(hkl1, hkl2))
    else:
        hkl3 = nearest_indices(n3, max_index)
        warnings.append("面 1、面 2 的标准指数不垂直，面 3 的标准指数取最接近的低指数面。")

    if best["perp_dev"] > PERP_WARNING:
        warnings.append(
            f"面 1 与面 2 最接近垂直的一组仍偏差 {best['perp_dev']:.2f} 度，"
            "请检查输入数据或两个面是否相邻。"
        )

    near = [c for c in candidates if c["perp_dev"] <= PERP_WARNING]
    families = {tuple(sorted(abs(x) for x in nearest_indices(c["n3"], max_index))) for c in near}
    if len(families) > 1:
        warnings.append("候选解给出的面 3 不属于同一晶面族，建议在第三个面上补测 EBSD。")

    return {
        "faces": [(hkl1, n1), (hkl2, n2), (hkl3, n3)],
        "angle_12": best["angle"],
        "candidates": near,
        "best": best,
        "warnings": warnings,
    }


def format_hkl(hkl):
    return "(" + " ".join(str(x) for x in hkl) + ")"


def format_vector(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def main():
    result = solve(EULER_1, EULER_2, MAX_INDEX)
    line = "=" * 70

    print(line)
    print("输入")
    print(f"  面 1 欧拉角 : {EULER_1}")
    print(f"  面 2 欧拉角 : {EULER_2}")
    print(line)
    print("结果（三个面已统一到同一晶体坐标系）")
    print(f"  {'面':<6}{'标准指数':<14}{'实测法向(晶体坐标)':<32}与标准面的偏差")
    for i, (hkl, n) in enumerate(result["faces"], start=1):
        name = f"面 {i}"
        print(f"  {name:<6}{format_hkl(hkl):<16}{format_vector(n):<36}{angle_deg(n, hkl):.2f} 度")
    print()
    print(f"  面 1 与面 2 的夹角 : {result['angle_12']:.2f} 度")
    print("  右手关系         : 面 1 法向 x 面 2 法向 = 面 3 法向")

    if SHOW_CANDIDATES:
        print(line)
        print(f"接近垂直的候选解（垂直度偏差不超过 {PERP_WARNING:.0f} 度，* 为选用的一组）")
        print(f"  {'':<3}{'面1-面2夹角':<14}{'面 2 法向':<30}面 3 法向")
        for c in result["candidates"]:
            mark = "*" if c is result["best"] else " "
            angle_text = f"{c['angle']:.2f} 度"
            print(f"  {mark:<3}{angle_text:<13}{format_vector(c['n2']):<32}{format_vector(c['n3'])}")

    if result["warnings"]:
        print(line)
        print("注意")
        for text in result["warnings"]:
            print(f"  - {text}")
    print(line)


if __name__ == "__main__":
    main()