
# -*- coding: utf-8 -*-
"""
读取多个 DDD 输出的 stress_strain_dens.dat 文件，绘制：

图1：应变 - 应力 对比曲线
图2：应变 - 位错密度 对比曲线

数据文件列：
Step  Strain  Stress(Pa)  Density(m^-2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 输入数据配置（可以添加更多组）
# ============================================================

DATA_FILES = [
    {
        "path": r"/data/home/dg000246d/Opendis_q/fccNi/picture/3um/rgroups_output/stress_strain_dens.dat",
        "label": "3um_yes",
        "color": "tab:blue",
        "linestyle": "-",
    },
    {
        "path": r"/data/home/dg000246d/Opendis_q/fccNi/picture/3um/no_output/no_stress_strain_dens.dat",
        "label": "3um_no",
        "color": "tab:red",
        "linestyle": "-",
    },
]

# ============================================================
# 2. 输出图片路径
# ============================================================

OUT_STRESS = r"/data/home/dg000246d/Opendis_q/fccNi/picture/no_yes/3um/strain_stress_compare.png"

OUT_DENSITY = r"/data/home/dg000246d/Opendis_q/fccNi/picture/no_yes/3um/strain_density_compare.png"

# ============================================================
# 3. 绘图参数
# ============================================================

FIGSIZE = (7, 5)
DPI = 300
LINEWIDTH = 1.5

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": False,
    "savefig.dpi": DPI,
})


# ============================================================
# 4. 读取数据函数
# ============================================================

def load_ddd_data(filepath):
    """
    读取 DDD 数据文件。

    返回：
        strain  : 应变
        stress  : 应力 (MPa)
        density : 位错密度 (m^-2)
    """

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"找不到数据文件：{filepath}"
        )

    data = np.loadtxt(
        filepath,
        comments="#",
        ndmin=2
    )

    if data.shape[1] < 4:
        raise ValueError(
            f"数据列数不足4列：{filepath}"
        )

    strain = data[:, 1]
    stress = data[:, 2] / 1e6
    density = data[:, 3]

    # 去除 NaN 和 Inf
    valid = (
        np.isfinite(strain)
        & np.isfinite(stress)
        & np.isfinite(density)
    )

    strain = strain[valid]
    stress = stress[valid]
    density = density[valid]

    if len(strain) == 0:
        raise ValueError(
            f"没有有效数据：{filepath}"
        )

    return strain, stress, density


# ============================================================
# 5. 主程序
# ============================================================

def main():

    # 创建输出目录
    for output_path in (OUT_STRESS, OUT_DENSITY):
        os.makedirs(
            os.path.dirname(os.path.abspath(output_path)),
            exist_ok=True
        )

    # 创建两张图
    fig1, ax1 = plt.subplots(figsize=FIGSIZE)
    fig2, ax2 = plt.subplots(figsize=FIGSIZE)

    # 遍历所有输入文件
    for config in DATA_FILES:

        filepath = config["path"]
        label = config["label"]
        color = config["color"]
        linestyle = config["linestyle"]

        print(f"正在读取：{filepath}")

        strain, stress, density = load_ddd_data(filepath)

        # -------------------------------
        # 图1：应变 - 应力
        # -------------------------------
        ax1.plot(
            strain,
            stress,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=LINEWIDTH
        )

        # -------------------------------
        # 图2：应变 - 位错密度
        # -------------------------------
        ax2.plot(
            strain,
            density,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=LINEWIDTH
        )

        print(
            f"{label}: "
            f"数据点数 = {len(strain)}, "
            f"最大应变 = {np.max(strain):.6e}, "
            f"最大应力 = {np.max(stress):.3f} MPa, "
            f"最大位错密度 = {np.max(density):.3e} m^-2"
        )

    # ========================================================
    # 6. 图1设置
    # ========================================================

    ax1.set_xlabel("Strain")
    ax1.set_ylabel("Stress (MPa)")

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax1.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(0, 0)
    )

    ax1.minorticks_on()
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(loc="best")

    fig1.tight_layout()
    fig1.savefig(OUT_STRESS, dpi=DPI, bbox_inches="tight")

    # ========================================================
    # 7. 图2设置
    # ========================================================

    ax2.set_xlabel("Strain")
    ax2.set_ylabel(r"Dislocation density (m$^{-2}$)")

    ax2.set_xlim(left=0)

    ax2.ticklabel_format(
        axis="both",
        style="sci",
        scilimits=(0, 0)
    )

    # 如需对数坐标，取消下面一行注释
    # ax2.set_yscale("log")

    ax2.minorticks_on()
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.legend(loc="best")

    fig2.tight_layout()
    fig2.savefig(OUT_DENSITY, dpi=DPI, bbox_inches="tight")

    # 关闭图片
    plt.close(fig1)
    plt.close(fig2)

    print("\n绘图完成！")
    print(f"应力应变图：{OUT_STRESS}")
    print(f"位错密度图：{OUT_DENSITY}")


if __name__ == "__main__":
    main()
