#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制应力-应变曲线 和 位错密度-应变曲线
数据文件 (列: Step Strain Stress Density)
图片输出到数据文件所在目录，只需修改 DATA_FILE。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                            # 服务器无显示器
import matplotlib.pyplot as plt

# ---- 配置 (只需修改这一行) ----
DATA_FILE = "/data/home/dg000246d/Opendis_q/fccNi/strain_rate/3000_011/40um_3000_011_1/40um_3000_011_1_stress_strain_dens.dat"

# ---- 输出路径 (由 DATA_FILE 自动生成) ----
OUT_DIR = os.path.dirname(DATA_FILE)
CASE = os.path.basename(DATA_FILE).replace("_stress_strain_dens.dat", "")
OUT_STRESS = os.path.join(OUT_DIR, f"{CASE}_stress_strain_curve.png")
OUT_DENS = os.path.join(OUT_DIR, f"{CASE}_density_strain_curve.png")

STRAIN_SCALE = 1e-2                              # 应变 -> %
STRESS_SCALE = 1e6                               # Pa -> MPa
DENS_SCALE = 1e10                                # m^-2 -> 10^10 m^-2
DENS_UNIT = r"$10^{10}\ \mathrm{m^{-2}}$"

# ---- 读取数据 ----
# 跳过以 # 开头的注释行；列: step, strain, stress, density
data = np.loadtxt(DATA_FILE, comments="#")
strain = data[:, 1] / STRAIN_SCALE   # 换算为 %
stress = data[:, 2] / STRESS_SCALE   # 换算为 MPa
dens = data[:, 3] / DENS_SCALE       # 换算为 10^10 m^-2

# ---- 应力-应变曲线 ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(strain, stress, "-", color="#1f6feb", linewidth=1.2)

ax.set_xlabel("Strain (%)")
ax.set_ylabel("Stress (MPa)")
ax.set_title(f"Stress vs. Strain ({CASE})")
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_STRESS, dpi=300)
plt.close(fig)
print(f"图片已保存到: {OUT_STRESS}")

# ---- 位错密度-应变曲线 ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(strain, dens, "-", color="#1f6feb", linewidth=1.2)

ax.set_xlabel("Strain (%)")
ax.set_ylabel(f"Dislocation density ({DENS_UNIT})")
ax.set_title(f"Dislocation Density vs. Strain ({CASE})")
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_DENS, dpi=300)
plt.close(fig)
print(f"图片已保存到: {OUT_DENS}")
