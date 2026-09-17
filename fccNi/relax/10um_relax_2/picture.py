#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制位错密度-步数曲线
数据文件 (列: Step Strain Stress Density)
图片输出到指定绝对路径。
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- 配置 ----
# 数据文件绝对路径 (SimulateNetwork 的 write_dir='output')
DATA_FILE = "/data/home/dg000246d/Opendis_q/fccNi/relax/10um_relax_2/output/stress_strain_dens.dat"
# 图片输出绝对路径
OUT_FILE = "/data/home/dg000246d/Opendis_q/fccNi/relax/10um_relax_2/density_step_curve.png"
DENS_SCALE = 1e10                                # m^-2 -> 10^10 m^-2
DENS_UNIT = r"$10^{10}\ \mathrm{m^{-2}}$"

# ---- 读取数据 ----
# 跳过以 # 开头的注释行；列: step, strain, stress, density
data = np.loadtxt(DATA_FILE, comments="#")
step = data[:, 0]
dens = data[:, 3] / DENS_SCALE   # 换算为 10^10 m^-2

# ---- 绘图 ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(step, dens, "-", color="#1f6feb", linewidth=1.2)

ax.set_xlabel("Step")
ax.set_ylabel(f"Dislocation density ({DENS_UNIT})")
ax.set_title("Dislocation Density vs. Step")
ax.grid(True, linestyle="--", alpha=0.4)

# 横坐标直接显示步数（不使用科学计数法）
ax.ticklabel_format(axis="x", style="plain")

fig.tight_layout()

# ---- 保存到指定绝对路径 ----
fig.savefig(OUT_FILE, dpi=300)
print(f"图片已保存到: {OUT_FILE}")
