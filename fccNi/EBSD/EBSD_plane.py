# -*- coding: utf-8 -*-
"""
EBSD 欧拉角 -> 表面晶面指数 -> 与标准晶面的角度差（立方晶系，Bunge Z-X-Z 约定）

使用方法：修改下面“参数设置”区的数值，保存后直接运行本文件即可
（在 IDE 里点运行，或双击 / python ebsd_plane.py）。

功能：
  1. 由平均欧拉角计算实际晶面指数、与标准晶面的偏离角、倾转方向。
  2. 由平均/最大定向伸展给出像素偏离角的“理论范围”（三角不等式估计）。
  3. （可选）读取 AZtecCrystal 导出的像素数据，逐点计算偏离角，
     给出真实的平均值、标准差、分位数、最大值和直方图。
"""

# ======================= 参数设置（只需修改这里） =======================
PHI1 = 322.4          # 欧拉角 φ1（度）  AZtecCrystal：平均定向伸展1
PHI = 44.91            # 欧拉角 Φ （度）  AZtecCrystal：平均定向伸展2
PHI2 = 89.56          # 欧拉角 φ2（度）  AZtecCrystal：平均定向伸展3

MEAN_SPREAD = 0.31    # 平均定向伸展（度）
MAX_SPREAD = 4.13     # 最大定向伸展（度）

REF_HKL = (1, 1, 0)   # 标准晶面指数 (h, k, l)

# ---- 可选：逐像素统计 ----
# 填 AZtecCrystal“像素数据”导出的文件路径（csv 或 txt），不需要就保持 None
# 例：PIXEL_FILE = r"D:\EBSD\样品2区3_像素数据.csv"
PIXEL_FILE = None
SAVE_HISTOGRAM = True  # 是否保存偏离角直方图（需要 matplotlib）

MAX_INDEX = 100       # 整数化时允许的最大指数（一般不用改）
# ======================================================================

import itertools
import os
import warnings
from math import gcd

warnings.filterwarnings("ignore", message="Glyph")

import numpy as np


# ---------------- 基础函数 ----------------
def orientation_matrix(phi1, Phi, phi2):
    """Bunge 取向矩阵 g：样品坐标 -> 晶体坐标，g = Rz(φ2)·Rx(Φ)·Rz(φ1)"""
    p1, P, p2 = np.radians([phi1, Phi, phi2])#把三个欧拉角从度转换成弧度，再分别赋给三个变量

    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])#生成绕 Z 轴转动角度 t 的 3×3 旋转矩阵，t 的单位是弧度

    def rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])#生成绕 X 轴转动角度 t 的 3×3 旋转矩阵

    return rz(p2) @ rx(P) @ rz(p1)#三个旋转矩阵连乘，得到总的取向矩阵 g


def cubic_symmetry_ops():
    """立方晶系 24 个转动对称操作"""
    ops = []
    for perm in itertools.permutations(range(3)):## 枚举轴的排列
        for signs in itertools.product([1, -1], repeat=3):#枚举三个正负号
            m = np.zeros((3, 3))
            for i in range(3):
                m[i, perm[i]] = signs[i]# 第 i 行第 perm[i] 列填正负号
            if np.isclose(np.linalg.det(m), 1):
                ops.append(m)#行列式 +1： 纯转动，保持右手坐标系，共 24 个
    return ops


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)#向量归一化成单位向量


def angle_deg(a, b):
    """两个方向的夹角（度）"""
    c = np.clip(np.dot(unit(a), unit(b)), -1, 1)
    return np.degrees(np.arccos(c))#先由 cosθ 求出弧度，再转换成度


def equivalent_directions(ref):
    """标准晶面法向的全部立方等价方向（单位向量，去重，不区分 ±）"""
    out = []# 存放已找到的方向
    for op in cubic_symmetry_ops():# 遍历 24 个转动操作
        d = unit(op @ ref)# 作用到 ref 上，再归一化
        if not any(abs(np.dot(d, e)) > 1 - 1e-9 for e in out):# 判断是否重复
            out.append(d)
    return np.array(out)#把一个标准晶面展开成它的全部等价方向，并去掉重复


def nearest_equivalent(n, ref):
    """在 ref 的所有立方等价方向（含 ±）中找与 n 最接近的一个"""
    best, best_ang = None, 180.0#先设初始值：best 是目前最近的方向，best_ang 是目前最小的夹角。两个方向的夹角最大是 180°，所以从 180° 开始比较，后面算出的角度几乎都会比它小。
    for d in equivalent_directions(ref):
        for sgn in (1, -1):
            a = angle_deg(n, sgn * d)#外层遍历去重后的等价方向，内层把每个方向的 + 和 - 都试一遍。这是因为 equivalent_directions 去重时只保留了正反方向中的一个，而 angle_deg 又区分正负。比如 n 和 (-1 0 -1) 的夹角接近 180°，但和 (1 0 1) 只差 0.32°，所以两个都要算。
            if a < best_ang:
                best, best_ang = sgn * d * np.linalg.norm(ref), a#找到更小的夹角就更新记录。
    return best, best_ang


def to_miller(v, max_index=100, tol=0.02):
    """把方向 v 近似为整数指数：取能把角度误差降到 tol(度) 以下的最小指数"""
    v = unit(v)
    vmax = np.max(np.abs(v))#这里先归一化，再找出绝对值最大的分量。后面用 v / vmax 把最大分量缩放成 ±1
    best, best_err = None, 180.0
    for m in range(1, max_index + 1):
        cand = np.round(v / vmax * m).astype(int)#每轮把缩放后的向量乘以 m，再四舍五入成整数，所以最大指数不超过 max_index
        if not cand.any():
            continue#这是防止出现 (0 0 0)。不过最大分量始终是 ±m，这里其实不会触发，只是一层保险
        g = gcd(gcd(abs(cand[0]), abs(cand[1])), abs(cand[2]))
        cand = cand // g#三个数除以最大公约数，比如 (2 0 2) 约成 (1 0 1)
        err = angle_deg(v, cand)
        if err < best_err - 1e-9:#比当前最优值更好才更新，1e-9 用来忽略浮点误差
            best, best_err = cand, err
        if err < tol:#误差小于 tol（默认 0.02°）就停止，所以通常能得到较小的指数
            break#记录误差最小的结果
    return best, best_err


def fmt_idx(idx, br="()"):
    return f"{br[0]}{' '.join(str(int(i)) for i in idx)}{br[1]}"#格式化晶面或晶向指数


def fmt_vec(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"#格式化小数向量


# ---------------- 逐像素统计 ----------------
def read_pixel_euler(path):
    """读取 AZtecCrystal 导出的像素数据，返回 (N,3) 欧拉角数组（度）"""
    import pandas as pd

    df = None
    for enc in ("utf-8-sig", "gbk", "utf-16"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError("无法读取文件，请确认是 csv/txt 格式")

    def find_col(key):
        for c in df.columns:
            name = str(c).lower().replace(" ", "").replace("_", "")
            if key in name:
                return c
        return None

    cols = [find_col(f"euler{i}") for i in (1, 2, 3)]
    if None in cols:
        raise ValueError(f"找不到 Euler 1/2/3 列，文件中的列名为：{list(df.columns)}")

    eul = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    keep = ~np.isnan(eul).any(axis=1)

    # 去掉零解（未标定）像素
    phase_col = find_col("phase") or (df.columns[0] if "相" in str(df.columns[0]) else None)
    if phase_col is not None:
        ph = df[phase_col].astype(str).str.lower()
        keep &= ~ph.str.contains("zero") & ~ph.isin(["0", "nan", ""])
    keep &= ~(np.abs(eul) < 1e-12).all(axis=1)
    return eul[keep]


def pixel_deviation(eul_deg, ref):
    """逐像素计算表面法向偏离标准晶面（含所有等价面）的角度（度）"""
    p, P, p2 = np.radians(eul_deg).T
    n = np.stack([np.sin(P) * np.sin(p2), np.sin(P) * np.cos(p2), np.cos(P)], axis=1)
    dots = np.abs(n @ equivalent_directions(ref).T).max(axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1, 1)))


def pixel_statistics(path, ref, line):
    print(line)
    print(" 逐像素统计（真实分布）")
    print(line)
    try:
        eul = read_pixel_euler(path)
    except Exception as e:
        print(f" 读取像素数据失败：{e}")
        return
    dev = pixel_deviation(eul, ref)
    q = np.percentile(dev, [5, 25, 50, 75, 95, 99])
    print(f" 有效像素数             : {dev.size}")
    print(f" 平均值 ± 标准差        : {dev.mean():.3f}° ± {dev.std():.3f}°")
    print(f" 中位数                 : {q[2]:.3f}°")
    print(f" 25% ~ 75% 分位         : {q[1]:.3f}° ~ {q[3]:.3f}°")
    print(f" 5% ~ 95% 分位          : {q[0]:.3f}° ~ {q[4]:.3f}°")
    print(f" 99% 分位               : {q[5]:.3f}°")
    print(f" 最小 / 最大            : {dev.min():.3f}° / {dev.max():.3f}°")
    for th in (1, 2, 3, 5):
        print(f" 偏离角 ≤ {th}° 的比例     : {np.mean(dev <= th) * 100:.2f}%")

    if SAVE_HISTOGRAM:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            upper = max(q[5] * 1.2, 1)
            ax.hist(dev[dev <= upper], bins=60, color="#3b6ea5", edgecolor="white", linewidth=0.3)
            ax.axvline(dev.mean(), color="#c0392b", ls="--", lw=1,
                       label=f"平均 {dev.mean():.2f}°")
            ax.set_xlabel(f"偏离 {fmt_idx(REF_HKL)} 的角度 (°)")
            ax.set_ylabel("像素数")
            ax.legend()
            fig.tight_layout()
            out = os.path.splitext(path)[0] + "_偏离角直方图.png"
            fig.savefig(out)
            plt.close(fig)
            print(f" 直方图已保存           : {out}")
        except ImportError:
            print(" （未安装 matplotlib，跳过直方图）")


# ---------------- 主计算 ----------------
def analyze(phi1, Phi, phi2, mean_spread, max_spread, ref_hkl, max_index=100):
    g = orientation_matrix(phi1, Phi, phi2)
    ref = np.array(ref_hkl, dtype=float)#先由欧拉角得到取向矩阵 g，再把标准晶面 (1, 1, 0) 转成浮点数组

    # 1. 表面法向（样品 Z）在晶体坐标系中的方向 = g 的第三列
    n = g[:, 2]#[:, 2] 表示取所有行的第 3 列，也就是样品 Z 方向在晶体坐标中的方向。你的结果是 (0.7060, 0.0054, 0.7082)。

    # 2. 实际晶面指数
    hkl, hkl_err = to_miller(n, max_index=max_index)#这里得到 (100 1 100)，误差 0.131°。

    # 3. 与标准晶面（考虑立方对称等价）的角度差
    ref_eq, dev_exact = nearest_equivalent(n, ref)#用精确的 n 算偏离角，得到 ref_eq = (1, 0, 1)，dev_exact = 0.324°。这是应该采用的结果。
    _, dev_hkl = nearest_equivalent(hkl.astype(float), ref)#用整数指数再算一次，得到 0.405°。_ 表示最近方向这个返回值不需要，直接丢弃。
    ref_eq_int = np.round(ref_eq).astype(int)#把 ref_eq 的浮点值取整，便于显示。

    # 4. 倾转轴与倾转方向（晶体坐标系）
    axis = np.cross(ref_eq, n)#倾转轴 axis： 叉乘结果同时垂直于 ref_eq 和 n，表示从标准方向转到 n 时绕的轴
    tilt_dir = n - np.dot(n, unit(ref_eq)) * unit(ref_eq)#倾转方向 tilt_dir： 从 n 中减去它沿 ref_eq 的分量，剩下的就是 n 垂直于 ref_eq 的分量，表示 n 朝哪边偏。
    has_tilt = np.linalg.norm(axis) > 1e-9#这里先判断是否真的存在倾斜。如果 n 和 ref_eq 完全重合，这两个向量都是零向量，归一化时会除以 0。
    if has_tilt:
        axis_idx = to_miller(axis, max_index=20, tol=1.0)[0]
        tilt_idx = to_miller(tilt_dir, max_index=20, tol=1.0)[0]#把倾转轴和倾转方向这两个小数向量化成整数晶向指数

    # 5. 该标准方向在样品坐标系中的指向
    ref_in_sample = g.T @ unit(ref_eq)#从晶体坐标换到样品坐标
    azimuth = np.degrees(np.arctan2(ref_in_sample[1], ref_in_sample[0])) % 360#计算面内方位角

    # ---------------- 输出 ----------------
    line = "=" * 64
    print(line)
    print(" 输入")
    print(line)
    print(f" 欧拉角 (φ1, Φ, φ2)     : {phi1:.2f}°, {Phi:.2f}°, {phi2:.2f}°")
    print(f" 平均定向伸展 / 最大    : {mean_spread:.2f}° / {max_spread:.2f}°")
    print(f" 标准晶面               : {fmt_idx(ref_hkl)}")

    print(line)
    print(" 平均取向的计算结果")
    print(line)
    print(f" 表面法向(晶体坐标)     : {fmt_vec(n)}")
    print(f" 实际晶面指数(近似)     : {fmt_idx(hkl)}   (整数化误差 {hkl_err:.3f}°)")
    print(f" 最接近的等价标准面     : {fmt_idx(ref_eq_int)}")
    print(f" 偏离角(由欧拉角精确算) : {dev_exact:.3f}°")
    print(f" 偏离角(由整数晶面算)   : {dev_hkl:.3f}°")
    if has_tilt:
        print(f" 倾转方向(晶体坐标)     : 由 {fmt_idx(ref_eq_int, '[]')} 朝 {fmt_idx(tilt_idx, '[]')} 方向倾斜")
        print(f" 倾转轴(晶体坐标)       : 约 {fmt_idx(axis_idx, '[]')}")
    print(f" 标准方向在样品坐标中   : {fmt_vec(ref_in_sample)}")
    print(f"   -> 偏离样品 Z {angle_deg(ref_in_sample, [0, 0, 1]):.3f}°，"
          f"倒向样品面内方位角 {azimuth:.1f}°（从 +X 逆时针量）")
    if dev_exact < 2.0:
        print(" 注意：偏离角 < 2°，与 EBSD 绝对取向精度（约 ±1–2°）相当，")
        print("       如需精确定向，建议用劳厄背散射/X 射线定向仪复核。")

    # 理论范围：三角不等式 |θ - ω| ≤ θp ≤ θ + ω
    print(line)
    print(" 像素偏离角的理论范围（三角不等式估计，仅供参考）")
    print(line)
    lo = max(dev_exact - mean_spread, 0)
    print(f" 典型像素(ω=平均伸展)   : {lo:.2f}° ~ {dev_exact + mean_spread:.2f}°")
    print(f" 理论上限(ω=最大伸展)   : ≤ {dev_exact + max_spread:.2f}°")
    print(" 说明：这只是边界估计。平均/最大定向伸展包含绕表面法向的面内转动，")
    print("       该部分不改变偏离角，因此真实范围通常更窄、最大值远小于上限。")
    print("       真实分布请设置 PIXEL_FILE 进行逐像素统计。")

    if PIXEL_FILE:
        pixel_statistics(PIXEL_FILE, ref, line)
    print(line)


if __name__ == "__main__":
    analyze(PHI1, PHI, PHI2, MEAN_SPREAD, MAX_SPREAD, REF_HKL, max_index=MAX_INDEX)
    input("\n按回车键退出...")