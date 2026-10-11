# -*- coding: utf-8 -*-
"""
读取 DDD 输出的 stress_strain_dens.dat，绘制：
  图1：应变 - 应力 曲线
  图2：应变 - 位错密度 曲线
数据文件列：Step  Strain  Stress(Pa)  Density(m^-2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ================== 路径设置（绝对路径，请按实际修改） ==================
DATA_FILE = r"/data/home/dg000246d/Opendis_q/fccNi/strain_rate/3000_001/10um_3000_001_1/stress_strain_dens.data"          # 输入数据
OUT_STRESS = r"/data/home/dg000246d/Opendis_q/fccNi/strain_rate/3000_001/10um_3000_001_1/strain_stress.png"       # 输出图1
OUT_DENSITY = r"/data/home/dg000246d/Opendis_q/fccNi/strain_rate/3000_001/10um_3000_001_1/strain_density.png"     # 输出图2
# =====================================================================


def main():
    # 读取数据，跳过以 # 开头的表头
    data = np.loadtxt(DATA_FILE, comments="#")
    strain = data[:, 1]
    stress = data[:, 2] / 1e6      # Pa -> MPa
    density = data[:, 3]           # m^-2

    # 确保输出目录存在
    for p in (OUT_STRESS, OUT_DENSITY):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    # ---------- 图1：应变 - 应力 ----------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(strain, stress, color="tab:blue", linewidth=1.5)
    ax1.set_xlabel("Strain")
    ax1.set_ylabel("Stress (MPa)")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.grid(alpha=0.3, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(OUT_STRESS, dpi=300)
    plt.close(fig1)

    # ---------- 图2：应变 - 位错密度 ----------
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(strain, density, color="tab:red", linewidth=1.5)
    ax2.set_xlabel("Strain")
    ax2.set_ylabel(r"Dislocation density (m$^{-2}$)")
    ax2.set_xlim(left=0)
    ax2.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    # 位错密度跨越两个数量级，如需对数坐标可取消下一行注释
    # ax2.set_yscale("log")
    ax2.grid(alpha=0.3, linestyle="--")
    fig2.tight_layout()
    fig2.savefig(OUT_DENSITY, dpi=300)
    plt.close(fig2)

    print(f"已保存: {OUT_STRESS}")
    print(f"已保存: {OUT_DENSITY}")


if __name__ == "__main__":
    main()