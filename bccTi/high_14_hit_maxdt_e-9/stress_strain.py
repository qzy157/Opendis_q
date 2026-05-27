#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制应力-应变曲线
数据文件 (列: Step Strain Stress Density)
图片输出到指定绝对路径。
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- 配置 ----
# 数据文件绝对路径
DATA_FILE = "/data/home/dg000246d/Opendis_q/bccTi/high_14_hit_maxdt_e-9/output/stress_strain_dens.dat"
# 图片输出绝对路径
OUT_FILE = "/data/home/dg000246d/Opendis_q/bccTi/high_14_hit_maxdt_e-9/stress_strain_curve.png"
STRESS_UNIT = "MPa"                     # 原始数据单位为 Pa，绘图换算为 MPa
STRESS_SCALE = 1e6                      # Pa -> MPa

# ---- 读取数据 ----
# 跳过以 # 开头的注释行；列: step, strain, stress, density
data = np.loadtxt(DATA_FILE, comments="#")
strain = data[:, 1]
stress = data[:, 2] / STRESS_SCALE   # 换算为 MPa

# ---- 绘图 ----
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(strain, stress, "-", color="#1f6feb", linewidth=1.5)

ax.set_xlabel("Strain")
ax.set_ylabel(f"Stress ({STRESS_UNIT})")
ax.set_title("Stress-Strain Curve")
ax.set_ylim(0, 120)
ax.grid(True, linestyle="--", alpha=0.4)

# 横坐标直接显示原始应变数值（不使用科学计数法）
ax.ticklabel_format(axis="x", style="plain")

fig.tight_layout()

# ---- 保存到指定绝对路径 ----
fig.savefig(OUT_FILE, dpi=300)
print(f"图片已保存到: {OUT_FILE}")