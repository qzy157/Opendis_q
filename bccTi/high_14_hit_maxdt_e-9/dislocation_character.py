#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计各类位错（螺/混合/刃）占总位错的百分比随应变的变化曲线。

位错类型 (CELL_DATA -> SCALARS DislocationCharacter):
    0 = 螺位错 (screw)
    1 = 混合位错 (mixed)
    2 = 刃位错 (edge)

占比按【位错段长度】加权（即位错密度口径），而非简单段数。
每个 vtk 的第一个 cell 是仿真盒子 (CELL_TYPES==12)，其 DislocationCharacter
恰好为 0，必须排除，只统计 CELL_TYPES==3 的真实位错线段。

vtk 文件 config.N.vtk (N=0..1194)：
    config.0   = 初始构型，应变=0（应力应变文件中无对应行）
    config.N>=1 对应 stress_strain_dens.dat 第 N 行 (跳过表头)。
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---------------- 路径配置 ----------------
BASE_DIR   = "/data/home/dg000246d/Opendis_q/bccTi/high_14_hit_maxdt_e-9"
VTK_DIR    = os.path.join(BASE_DIR, "vtk")
SS_FILE    = os.path.join(BASE_DIR, "output", "stress_strain_dens.dat")

# 输出到本脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE   = os.path.join(SCRIPT_DIR, "dislocation_fraction_vs_strain.png")
CSV_FILE   = os.path.join(SCRIPT_DIR, "dislocation_fraction_vs_strain.csv")

# vtk 文件统一匹配 *.vtk（兼容 config_1.vtk / config.1.vtk / config1.vtk 等命名）
VTK_PATTERN = "*.vtk"

CHAR_NAMES = {0: "Screw", 1: "Mixed", 2: "Edge"}

# ---------------- vtk 解析 ----------------
def parse_vtk(path):
    """返回 (screw_len, mixed_len, edge_len)：按段长度加权的各类位错总长。"""
    with open(path, "r") as f:
        lines = f.read().splitlines()

    n = len(lines)
    i = 0
    points = None
    cells = []          # 每个元素是该 cell 的点索引列表
    cell_types = None
    char = None         # DislocationCharacter 数组

    while i < n:
        line = lines[i].strip()

        if line.startswith("POINTS"):
            npts = int(line.split()[1])
            coords = []
            i += 1
            # 读取 npts 个点（每行可能有多个数，按需累积）
            vals = []
            while len(vals) < npts * 3:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            points = np.array(vals[:npts * 3]).reshape(npts, 3)
            continue

        if line.startswith("CELLS"):
            ncells = int(line.split()[1])
            i += 1
            for _ in range(ncells):
                parts = [int(x) for x in lines[i].split()]
                cells.append(parts[1:])   # parts[0] 是点数，跳过
                i += 1
            continue

        if line.startswith("CELL_TYPES"):
            nct = int(line.split()[1])
            i += 1
            ct = []
            while len(ct) < nct:
                ct.extend(int(x) for x in lines[i].split())
                i += 1
            cell_types = np.array(ct[:nct])
            continue

        if line.startswith("SCALARS") and "DislocationCharacter" in line:
            # 下一行是 LOOKUP_TABLE，再往后是数据
            i += 2
            vals = []
            # 数据个数 == cell 数量
            ncells = len(cells)
            while len(vals) < ncells:
                row = lines[i].split()
                if not row:          # 遇到空行或新段落停止
                    break
                # 若遇到新的关键字段则停止
                if any(k in lines[i] for k in ("SCALARS", "VECTORS", "LOOKUP_TABLE")):
                    break
                vals.extend(int(x) for x in row)
                i += 1
            char = np.array(vals[:len(cells)])
            continue

        i += 1

    # 长度加权统计，仅保留 CELL_TYPES == 3 (line) 的真实位错段
    lengths = {0: 0.0, 1: 0.0, 2: 0.0}
    for idx, pts in enumerate(cells):
        if cell_types[idx] != 3:        # 排除仿真盒子等非线段 cell
            continue
        if len(pts) < 2:
            continue
        p0, p1 = points[pts[0]], points[pts[1]]
        L = np.linalg.norm(p1 - p0)
        c = int(char[idx])
        if c in lengths:
            lengths[c] += L
    return lengths[0], lengths[1], lengths[2]


# ---------------- 主流程 ----------------
def frame_index(path):
    """从文件名提取帧号，取文件名中最后一段数字。
    config_12.vtk -> 12, config.12.vtk -> 12"""
    nums = re.findall(r"(\d+)", os.path.basename(path))
    return int(nums[-1]) if nums else -1

def main():
    # 读取应变列 (第二列，索引1)，跳过表头
    ss = np.loadtxt(SS_FILE, comments="#")
    strain_all = ss[:, 1]

    vtk_files = sorted(glob.glob(os.path.join(VTK_DIR, VTK_PATTERN)),
                       key=frame_index)
    if not vtk_files:
        # 列出目录实际内容辅助排查
        try:
            sample = os.listdir(VTK_DIR)[:10]
        except FileNotFoundError:
            sample = "(目录不存在)"
        raise FileNotFoundError(
            f"在 {VTK_DIR} 未找到匹配 {VTK_PATTERN} 的文件。"
            f"目录内容示例: {sample}")

    print(f"找到 {len(vtk_files)} 个 vtk 文件")

    # 是否把 config.0（初始构型，应变=0）画进曲线。改为 False 则只画 config.1~。
    INCLUDE_FRAME0 = True

    strains, frac_screw, frac_mixed, frac_edge = [], [], [], []

    for path in vtk_files:
        N = frame_index(path)            # vtk 从 0 开始 (config.0 ... config.1194)

        if N == 0:
            # config.0 是初始构型，应变=0，应力应变文件中无对应行
            if not INCLUDE_FRAME0:
                continue
            strain_val = 0.0
        else:
            # config.N (N>=1) 对应应变数据第 N 行 (1-based) -> 索引 N-1
            si = N - 1
            if si < 0 or si >= len(strain_all):
                print(f"  跳过 {os.path.basename(path)}: 应变数据无第 {N} 行")
                continue
            strain_val = strain_all[si]

        s_len, m_len, e_len = parse_vtk(path)
        total = s_len + m_len + e_len
        if total <= 0:
            continue

        strains.append(strain_val)
        frac_screw.append(100.0 * s_len / total)
        frac_mixed.append(100.0 * m_len / total)
        frac_edge.append(100.0 * e_len / total)

    strains    = np.array(strains)
    frac_screw = np.array(frac_screw)
    frac_mixed = np.array(frac_mixed)
    frac_edge  = np.array(frac_edge)

    # 按应变排序，确保曲线单调推进
    order = np.argsort(strains)
    strains    = strains[order]
    frac_screw = frac_screw[order]
    frac_mixed = frac_mixed[order]
    frac_edge  = frac_edge[order]

    # ---- 保存数据为 csv ----
    header = "strain,screw_percent,mixed_percent,edge_percent"
    np.savetxt(CSV_FILE,
               np.column_stack([strains, frac_screw, frac_mixed, frac_edge]),
               delimiter=",", header=header, comments="")
    print(f"数据已保存到: {CSV_FILE}")

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(strains, frac_screw, "-",  color="#d62728", linewidth=1.6, label="Screw (0)")
    ax.plot(strains, frac_mixed, "-",  color="#2ca02c", linewidth=1.6, label="Mixed (1)")
    ax.plot(strains, frac_edge,  "-",  color="#1f6feb", linewidth=1.6, label="Edge (2)")

    ax.set_xlabel("Strain")
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Dislocation Character Fraction vs Strain")
    ax.set_ylim(0, 100)
    ax.ticklabel_format(axis="x", style="plain")
    # 应变保留固定小数位，避免标签过长重叠
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=300)
    print(f"图片已保存到: {OUT_FILE}")


if __name__ == "__main__":
    main()