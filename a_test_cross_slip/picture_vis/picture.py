#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比四种交滑移设置下的应力-应变曲线与位错密度-应变曲线。

数据文件四列: Step  Strain  Stress(Pa)  Density(1/m^2)

  no            : 未开启交滑移
  yes           : thermal 版本交滑移
  yes_wansheng  : wansheng 版本交滑移
  yes_test      : test 版本交滑移

生成的两张图片保存在本 .py 文件所在目录下。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示环境(HPC)也能出图
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. 四个数据文件(绝对路径)及其图例标签
# ---------------------------------------------------------------------------
CASES = [
    ("/data/home/dg000246d/Opendis_q/a_test_cross_slip/no/output_no_cross_slip/stress_strain_dens.dat",
     "No cross-slip",        "#1f77b4"),
    ("/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_vis/output_with_cross_slip/stress_strain_dens.dat",
     "Cross-slip (thermal)", "#d62728"),
    ("/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_wansheng_vis/output_cross_slip_wansheng/stress_strain_dens.dat",
     "Cross-slip (wansheng)", "#2ca02c"),
    ("/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_test_vis/output_cross_slip_test/stress_strain_dens.dat",
     "Cross-slip (test)",    "#9467bd"),
]

# 图片输出目录 = 本脚本所在目录
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(path):
    """读取数据文件, 返回 strain, stress(MPa), density(1/m^2)。"""
    # 跳过以 # 开头的表头, 自动忽略行尾多余空白
    data = np.loadtxt(path, comments="#")
    strain  = data[:, 1]
    stress  = data[:, 2] / 1.0e6     # Pa -> MPa
    density = data[:, 3]             # 1/m^2
    return strain, stress, density


# ---------------------------------------------------------------------------
# 2. 读取所有数据
# ---------------------------------------------------------------------------
results = []
for path, label, color in CASES:
    if not os.path.isfile(path):
        print(f"[警告] 文件不存在, 已跳过: {path}")
        continue
    strain, stress, density = load_data(path)
    results.append((label, color, strain, stress, density))
    print(f"[已读取] {label:24s}  点数={len(strain):5d}  "
          f"max_strain={strain.max():.3e}  "
          f"max_stress={stress.max():.2f} MPa  "
          f"max_dens={density.max():.3e} /m^2")

if not results:
    raise SystemExit("没有读取到任何有效数据文件, 请检查路径。")

# ---------------------------------------------------------------------------
# 3. 图一: 应力-应变曲线
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))
for label, color, strain, stress, density in results:
    ax1.plot(strain, stress, label=label, color=color, linewidth=1.6)

ax1.set_xlabel("Strain", fontsize=13)
ax1.set_ylabel("Stress (MPa)", fontsize=13)
ax1.set_title("Stress-Strain Curves", fontsize=14)
ax1.legend(fontsize=11, frameon=True)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.tick_params(labelsize=11)
fig1.tight_layout()

ss_png = os.path.join(OUT_DIR, "stress_strain.png")
fig1.savefig(ss_png, dpi=300)
print(f"[已保存] {ss_png}")

# ---------------------------------------------------------------------------
# 4. 图二: 位错密度-应变曲线
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6))
for label, color, strain, stress, density in results:
    ax2.plot(strain, density, label=label, color=color, linewidth=1.6)

ax2.set_xlabel("Strain", fontsize=13)
ax2.set_ylabel(r"Dislocation density (m$^{-2}$)", fontsize=13)
ax2.set_title("Dislocation Density vs Strain", fontsize=14)
ax2.legend(fontsize=11, frameon=True)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.tick_params(labelsize=11)
fig2.tight_layout()

dens_png = os.path.join(OUT_DIR, "density_strain.png")
fig2.savefig(dens_png, dpi=300)
print(f"[已保存] {dens_png}")

print("完成。")