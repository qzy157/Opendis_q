
# -*- coding: utf-8 -*-
"""
镍单晶 EBSD 三个晶面偏差角计算

用户自行输入：
1. EBSD 三个 Bunge 欧拉角
2. X/Y/Z 三个面的目标标准晶面指数

程序输出：
1. 三个实际晶面的法向
2. 最近等价标准晶面
3. 各面独立匹配的偏差角
4. 三个面的联合正交匹配结果

适用于立方晶系（如 FCC 镍单晶）。

注意：
EBSD 样品 X/Y/Z 轴必须与实际加工面的
三个法向对应。
"""

import numpy as np
import itertools


# ==================================================
# 一、用户参数设置（只需修改这里）
# ==================================================

# EBSD 欧拉角，单位：度
PHI1 = 271.7
PHI = 1.58
PHI2 = 44.8

# 三个面的目标标准晶面指数
# 只需要输入晶面族的代表指数
# 程序自动考虑立方对称性及正负号
REF_X = (1, 1, 0)
REF_Y = (1, 0, 0)
REF_Z = (0, 1, 1)

# 是否进行三个面的联合正交匹配
JOINT_MATCH = True

# EBSD X/Y 与加工 X/Y 的面内旋转角（度）
# 两者对齐时设置为 0
INPLANE_ROTATION = 0.0


# ==================================================
# 二、基础函数
# ==================================================

def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < 1e-12:
        raise ValueError("晶面指数不能全为零")

    return v / norm


def angle_deg(a, b, plane=False):
    """
    计算两个方向夹角。

    plane=False：有向夹角，0~180°
    plane=True：晶面夹角，0~90°
    """
    dot = np.dot(unit(a), unit(b))

    if plane:
        dot = abs(dot)

    dot = np.clip(dot, -1.0, 1.0)

    return np.degrees(np.arccos(dot))


def fmt_idx(v):
    v = np.rint(v).astype(int)
    return "(" + " ".join(map(str, v)) + ")"


def fmt_vec(v):
    return "(" + ", ".join(
        f"{x:+.6f}" for x in v
    ) + ")"


# ==================================================
# 三、Bunge 欧拉角转换取向矩阵
# ==================================================

def orientation_matrix(phi1, Phi, phi2):

    p1, P, p2 = np.radians(
        [phi1, Phi, phi2]
    )

    def rz(t):
        c = np.cos(t)
        s = np.sin(t)

        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

    def rx(t):
        c = np.cos(t)
        s = np.sin(t)

        return np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])

    return rz(p2) @ rx(P) @ rz(p1)


def sample_normals(g, rotation=0.0):
    """
    将 EBSD 坐标转换为实际加工坐标。

    返回矩阵的三列分别为：
    X/Y/Z 实际晶面法向（晶体坐标）。
    """

    t = np.radians(rotation)

    c = np.cos(t)
    s = np.sin(t)

    transform = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    return g @ transform


# ==================================================
# 四、立方晶系等价晶面
# ==================================================

def cubic_symmetry_ops():

    ops = []

    for perm in itertools.permutations(range(3)):

        for signs in itertools.product(
            [1, -1], repeat=3
        ):

            m = np.zeros((3, 3))

            for i in range(3):
                m[i, perm[i]] = signs[i]

            if np.isclose(np.linalg.det(m), 1):
                ops.append(m)

    return ops


CUBIC_OPS = cubic_symmetry_ops()


def equivalent_planes(ref):
    """
    生成立方晶系所有等价晶面法向，
    包括正负方向。
    """

    ref = unit(ref)
    directions = []

    for op in CUBIC_OPS:

        v = op @ ref

        for sign in (1, -1):

            d = sign * v

            if not any(
                np.dot(d, e) > 1 - 1e-10
                for e in directions
            ):
                directions.append(d)

    return directions


def nearest_equivalent(n, ref):
    """
    寻找最接近实际晶面法向的
    等价标准晶面。
    """

    best = None
    best_angle = float("inf")

    for d in equivalent_planes(ref):

        angle = angle_deg(n, d)

        if angle < best_angle:
            best = d.copy()
            best_angle = angle

    return best, best_angle


def direction_to_index(direction, ref):
    """
    将等价标准晶面法向恢复为整数指数。
    """
    scale = np.linalg.norm(ref)

    return np.rint(
        direction * scale
    ).astype(int)


# ==================================================
# 五、联合正交匹配
# ==================================================

def joint_match(normals, refs):
    """
    搜索三个目标晶面族中相互正交的
    标准晶面组合，并使总偏差最小。
    """

    candidates = [
        equivalent_planes(ref)
        for ref in refs
    ]

    best_Q = None
    best_angles = None
    best_score = float("inf")

    for x, y, z in itertools.product(*candidates):

        Q = np.column_stack([x, y, z])

        # 三个标准晶面必须正交
        if not np.allclose(
            Q.T @ Q,
            np.eye(3),
            atol=1e-8
        ):
            continue

        # 必须为右手坐标系
        if np.linalg.det(Q) < 0.999:
            continue

        angles = np.array([
            angle_deg(
                normals[:, i],
                Q[:, i]
            )
            for i in range(3)
        ])

        score = np.sum(angles ** 2)

        if score < best_score:
            best_score = score
            best_Q = Q.copy()
            best_angles = angles.copy()

    return best_Q, best_angles, best_score


# ==================================================
# 六、主计算函数
# ==================================================

def analyze():

    # 生成 EBSD 取向矩阵
    g = orientation_matrix(
        PHI1, PHI, PHI2
    )

    # 三个实际加工面的法向
    normals = sample_normals(
        g, INPLANE_ROTATION
    )

    refs = [REF_X, REF_Y, REF_Z]
    names = ["X", "Y", "Z"]

    print("=" * 70)
    print("镍单晶 EBSD 三个晶面偏差角计算")
    print("=" * 70)

    print(
        f"欧拉角："
        f"{PHI1:.4f}°, "
        f"{PHI:.4f}°, "
        f"{PHI2:.4f}°"
    )

    print("\n取向矩阵：")
    print(np.round(g, 6))

    print("\n" + "=" * 70)
    print("一、三个晶面独立匹配结果")
    print("=" * 70)

    for i, name in enumerate(names):

        actual = normals[:, i]
        ref = refs[i]

        nearest, deviation = nearest_equivalent(
            actual, ref
        )

        idx = direction_to_index(
            nearest, ref
        )

        print(f"\n{name} 面")
        print("-" * 45)

        print(
            "实际晶面法向：",
            fmt_vec(actual)
        )

        print(
            "输入标准晶面族：",
            ref
        )

        print(
            "最近等价标准晶面：",
            fmt_idx(idx)
        )

        print(
            f"偏差角：{deviation:.4f}°"
        )

    # 三个面的联合匹配
    if JOINT_MATCH:

        print("\n" + "=" * 70)
        print("二、三个晶面联合正交匹配")
        print("=" * 70)

        Q, angles, score = joint_match(
            normals, refs
        )

        if Q is None:

            print(
                "输入的三个晶面族无法组成"
                "相互正交的标准坐标系。"
            )

        else:

            for i, name in enumerate(names):

                idx = direction_to_index(
                    Q[:, i],
                    refs[i]
                )

                print(
                    f"{name} 面："
                    f"标准晶面 {fmt_idx(idx)}，"
                    f"偏差 {angles[i]:.4f}°"
                )

            print(
                f"\n总偏差评分："
                f"{score:.6f} deg²"
            )

    print("\n" + "=" * 70)
    print("计算完成")
    print("=" * 70)


# ==================================================
# 七、程序入口
# ==================================================

if __name__ == "__main__":

    analyze()

    input("\n按回车键退出...")
