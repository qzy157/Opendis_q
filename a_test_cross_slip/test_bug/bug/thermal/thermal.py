#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thermal.py —— 用"符合伯氏守恒的弯臂"跑一个真实 ExaDiS 仿真，复现 FCC 交滑移的
"弯臂漏检"bug（旧实现）。引用的交滑移源文件：
    core/exadis/src/cross_slip_types/cross_slip_fcc_thermal.h

产物：把每一步位错构型写成 output/config.<i>.data（ExaDiS 标准输出）。
你可自行把 .data 转成 .vtk，用 ParaView 看时间序列。

=============================================================================
一、相对参考脚本 test_bug_force.py 的两处修正
-----------------------------------------------------------------------------
1) 伯氏守恒：参考脚本 8 个节点全 UNCONSTRAINED，是两端悬空的开线——自由端点带净
   Burgers 矢量、违反节点处 sum b = 0，非法。本文件按 ExaDiS 规范
   （pyexadis_utils.insert_frank_read_src）把弯臂两端设 PINNED_NODE（锚定），
   中间 UNCONSTRAINED。钉扎端点是"锚"免于闭合约束、内部逐点守恒 —— 合法构型。
   拓扑上（network.cpp）physical node = 连接度!=2 或 PINNED_NODE，所以"两端钉扎、
   内部 2 连通"让 physical_links() 把整条弯臂返回为【一条臂】，正好落进 thermal
   旧实现"整臂判螺型"的失效路径 —— bug 语义不变。

2) 从"手动注力单步测试"改为"真实仿真 + 应力驱动"：本文件用 SimulateNetwork 跑真实
   时间步，力由 ExaDiS 从施加应力算出，用【应力加载】提供交滑移驱动。

二、应力设计（与 test.h / wansheng.h 完全一致，保证对照公平）
-----------------------------------------------------------------------------
施加常应力  sigma = tau0 * (s⊗m + m⊗s)， s=b̂， m=[1 -1 0]̂：
  * 原滑移面 (111) 上 RSS = 0   -> 交滑移前不弯；对旧实现更意味着"全程原面不动"。
  * 交滑移面 (1 -1 1) 上 RSS = 0.816*tau0 = 40.8 MPa（tau0=5e7）。
Voigt[xx,yy,zz,yz,xz,xy] = tau0*[1,0,0,0.5,-0.5,-0.5]。

三、bug 构型（伯氏守恒版）
-----------------------------------------------------------------------------
一条 ~900 b 的钉扎臂，b=1/2[1 0 -1]，滑移面 (111)：
  节点 0..5：沿 b 的理想螺型 5 段（每段 ~100 b，共 ~500 b），共面 (111)
  节点 5..7：末端拐弯 2 段，非螺型
  节点 0、7：PINNED_NODE；节点 1..6：UNCONSTRAINED

四、预期（thermal.h，旧实现 = bug）
-----------------------------------------------------------------------------
build_screw_chains 用整臂首尾弦向量判螺型：弦向与 b 夹角 ~38.6°(>15°) -> 整条臂被判
非螺型而【整条丢弃】-> 0 条链 -> handle_bulk 从不被调用 -> 无论多大交滑移应力都不
交滑移。又因原面 (111) 上 RSS=0，位错在原面也无滑移力。
  => 全程 100 帧位错【完全静止、滑移面始终 (111)、无一段翻面】—— 复现 bug。
对照 test.h / wansheng.h：同一构型、同一应力下前 5 段会交滑移到 (1 -1 1) 并弯出。

用法
-----------------------------------------------------------------------------
    cd Opendis_q/a_test_cross_slip/test_bug/bug/thermal
    python3 thermal.py
    # 生成 output/config.0.data ... config.100.data
（thermal 绑定通常已在 .so 中；若缺失请重编译安装 pyexadis。）
"""

import sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if pyexadis_path not in sys.path:
    sys.path.append(pyexadis_path)
np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    from pyexadis_base import (ExaDisNet, DisNetManager, NodeConstraints, SimulateNetwork,
                               CalForce, MobilityLaw, TimeIntegration, Remesh, CrossSlip,
                               VisualizeNetwork, get_exadis_params, get_exadis_force)
except ImportError as e:
    raise ImportError('无法导入 pyexadis（请确认已编译并安装）: %s' % e)

# ---------------------------------------------------------------------------
#  本文件专属：只引用 thermal.h 对应的绑定
# ---------------------------------------------------------------------------
MAKE_FN_NAME = 'make_cross_slip_fcc_thermal'
PARAMS_CLS   = 'CrossSlipFCCThermal_Params'
CS_MODE      = 'FCCThermal'

# ---------------------------------------------------------------------------
#  几何：Burgers / 滑移面
# ---------------------------------------------------------------------------
B_VEC       = np.array([0.5, 0.0, -0.5])     # 1/2[1 0 -1]
GLIDE_PLANE = np.array([1.0, 1.0, 1.0])      # (111)
TAU0        = 5.0e7                           # 剪应力标度 [Pa]（RSS_cs=0.816*tau0=40.8 MPa）


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_applied_stress(tau0):
    """sigma = tau0 (s⊗m + m⊗s)，s=b̂，m=[1 -1 0]̂：原面 (111) RSS=0，交滑移面 (1 -1 1)
    RSS=0.816*tau0。返回 Voigt[xx,yy,zz,yz,xz,xy]。"""
    s = unit(B_VEC)
    m = unit(np.array([1.0, -1.0, 0.0]))
    S = tau0 * (np.outer(s, m) + np.outer(m, s))
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1]])


# ---------------------------------------------------------------------------
#  伯氏守恒的弯臂网络：8 节点、7 段；两端 PINNED_NODE，内部 UNCONSTRAINED
# ---------------------------------------------------------------------------
def build_bent_arm_network(L_box=4000.0):
    cell = pyexadis.Cell(h=L_box * np.eye(3), is_periodic=[1, 1, 1])
    b = B_VEC.copy()
    plane = unit(GLIDE_PLANE)

    P = np.array([
        [  0.000,   0.000,    0.000],
        [ 70.711,   0.000,  -70.711],
        [141.421,   0.000, -141.421],
        [212.132,   0.000, -212.132],
        [282.843,   0.000, -282.843],
        [353.553,   0.000, -353.553],            # 0..5：沿 b 的理想螺型（5 段 × ~100 b）
        [311.018, 230.179, -541.196],            # 5..6：拐弯，非螺型
        [268.482, 460.357, -728.839],            # 6..7：拐弯，非螺型
    ])
    P = P - P.mean(axis=0) + cell.center()        # 居中到盒心

    n = len(P)
    constraints = [NodeConstraints.PINNED_NODE if (i == 0 or i == n - 1)
                   else NodeConstraints.UNCONSTRAINED for i in range(n)]
    nodes = np.array([np.concatenate((P[i], [constraints[i]])) for i in range(n)])
    segs  = np.array([np.concatenate(([i, i + 1], b, plane)) for i in range(n - 1)])
    return DisNetManager(ExaDisNet(cell, nodes, segs))


def make_cross_slip(state, force_module):
    """构建 thermal.h（旧实现）的交滑移对象，热激活势垒清零（Ea=0,Va=0 -> 去掉概率
    因素）。thermal.h 无 minRunLength/minChainSegments 字段 —— 即使去掉概率因素，旧实现
    仍在"整臂判螺型"上游就把链丢掉，这些参数根本轮不到起作用。"""
    params = get_exadis_params(state)
    force_obj, force_python = get_exadis_force(force_module, state, params)

    p = getattr(pyexadis, PARAMS_CLS)()
    p.temperature            = 300.0
    p.evalFrequency          = 1
    p.screwAngleTolerance    = 15.0
    p.bulkActivationEnergy   = 0.0
    p.bulkActivationVolume   = 0.0
    p.bulkAttemptFrequency   = 5.0e17
    p.bulkReferenceLength    = 1000.0

    xs = CrossSlip.__new__(CrossSlip)
    xs.cross_slip_mode = CS_MODE
    xs.force_python = force_python
    xs.cross_slip = getattr(pyexadis, MAKE_FN_NAME)(params=params, force=force_obj, fcc_params=p)
    return xs


def main():
    pyexadis.initialize()
    try:
        if not hasattr(pyexadis, MAKE_FN_NAME):
            print('>>> 绑定里没有 %s。请先重新编译并安装 pyexadis 后再运行。' % MAKE_FN_NAME)
            return

        state = {
            "crystal": "fcc", "burgmag": 2.55e-10, "mu": 54.6e9, "nu": 0.324, "a": 6.0,
            "maxseg": 300.0, "minseg": 50.0, "rann": 10.0, "rtol": 10.0,
            "nextdt": 1e-12, "maxdt": 1e-10,
            "use_glide_planes": True,        # handle() 要求 use_glide_planes=true
        }

        net = build_bent_arm_network(L_box=4000.0)

        calforce  = CalForce(force_mode='LineTension', state=state)
        mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
        timeint   = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
        remesh    = Remesh(remesh_rule='LengthBased', state=state)
        cross_slip = make_cross_slip(state, calforce)
        vis       = VisualizeNetwork()

        applied_stress = build_applied_stress(TAU0)
        print('[ CrossSlipFCCThermal (thermal.h, 旧实现/有 bug) | 应力驱动真实仿真 ]')
        print('  applied_stress Voigt =', applied_stress)
        print('  RSS(111)=0，RSS(1 -1 1)=%.1f MPa（可交滑移面）' % (0.816 * TAU0 / 1e6))
        print('  输出 -> output/config.<i>.data\n')

        sim = SimulateNetwork(calforce=calforce, mobility=mobility, timeint=timeint,
                              collision=None, topology=None, remesh=remesh,
                              cross_slip=cross_slip, vis=vis,
                              state=state, max_step=100,
                              loading_mode='stress', applied_stress=applied_stress,
                              burgmag=state["burgmag"],
                              print_freq=1, plot_freq=1, plot_pause_seconds=0.0001,
                              write_freq=1, write_dir='output')
        sim.run(net, state)
        print('\n完成。构型序列已写入 output/config.<i>.data（可转 vtk 后用 ParaView 查看）。')
        print('预期（复现 bug）：整条弯臂被丢弃、全程静止在 (111)、无一段交滑移；')
        print('           与 test.h / wansheng.h 的"前 5 段翻面弯出"形成对照。')
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
