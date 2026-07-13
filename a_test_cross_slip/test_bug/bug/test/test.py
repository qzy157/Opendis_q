#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py —— 用"符合伯氏守恒的弯臂"跑一个真实 ExaDiS 仿真，检验 FCC 交滑移的
"弯臂漏检"bug（修复版）。引用的交滑移源文件：
    core/exadis/src/cross_slip_types/cross_slip_fcc_test.h

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
   内部 2 连通"让 physical_links() 把整条弯臂返回为【一条臂】，bug 语义不变。

2) 从"手动注力单步测试"改为"真实仿真 + 应力驱动"：参考脚本直接往节点写力、只调一次
   handle_cross_slip。本文件用 SimulateNetwork 跑真实时间步——力由 ExaDiS 自己从
   施加应力算出（不能再手动注入），因此改用【应力加载】把交滑移驱动出来。

二、应力设计（保证交滑移必触发、且不会误触发原面滑移）
-----------------------------------------------------------------------------
施加常应力  sigma = tau0 * (s⊗m + m⊗s)， s = b/|b|， m = [1 -1 0]/√2。
  * s·(111)=0 且 m⊥(111)  ->  原滑移面 (111) 上分解剪应力 RSS = 0
       => 交滑移前位错在原面上不受滑移力、螺型 run 保持笔直，不会提前弯掉。
  * 交滑移面 (1 -1 1) 上 RSS = 0.816*tau0 (取 tau0=5e7 -> 40.8 MPa)
       => 远超交滑移阈值 mu*b/(10L) ≈ 10.9 MPa（约 3.7 倍），必触发；
          又低于 Frank-Read 发射临界，弯出而不失控。
Voigt[xx,yy,zz,yz,xz,xy] = tau0*[1,0,0,0.5,-0.5,-0.5]。

三、bug 构型（伯氏守恒版）
-----------------------------------------------------------------------------
一条 ~900 b 的钉扎臂，b=1/2[1 0 -1]，滑移面 (111)：
  节点 0..5：沿 b 的理想螺型 5 段（每段 ~100 b，共 ~500 b），共面 (111)
  节点 5..7：末端拐弯 2 段，非螺型
  节点 0、7：PINNED_NODE；节点 1..6：UNCONSTRAINED

四、预期（test.h，修复版）
-----------------------------------------------------------------------------
逐 segment 判螺型，切出前 5 段连续螺型 run -> 1 条链 -> 头一两步就在 (1 -1 1) 面
上过 Schmid 阈值触发交滑移（Ea=0,Va=0 去掉概率因素）-> 前 5 段滑移面翻到 (1 -1 1)，
随后在该面上弯出。对照 thermal.h：整条臂被丢弃、全程静止（原面 RSS=0，不动）。

用法
-----------------------------------------------------------------------------
    cd Opendis_q/a_test_cross_slip/test_bug/bug/test
    python3 test.py
    # 生成 output/config.0.data ... config.100.data
注意：CrossSlipFCCTest 为新增绑定，需先在 Linux 重编译安装 pyexadis：
    cd Opendis_q && cmake --build build -j8 && cmake --build build --target install
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
#  本文件专属：只引用 test.h 对应的绑定
# ---------------------------------------------------------------------------
MAKE_FN_NAME = 'make_cross_slip_fcc_test'
PARAMS_CLS   = 'CrossSlipFCCTest_Params'
CS_MODE      = 'FCCTest'

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
    """sigma = tau0 (s⊗m + m⊗s)，s=b̂，m=[1 -1 0]̂：
    原面 (111) RSS=0（交滑移前不弯），交滑移面 (1 -1 1) RSS=0.816*tau0（驱动交滑移）。
    返回 Voigt[xx,yy,zz,yz,xz,xy]。"""
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
    """构建 test.h 的交滑移对象，并把热激活势垒清零（Ea=0,Va=0 -> 去掉概率因素，
    能交滑移的必然交滑移），minRunLength=0（不按长度过滤该螺型 run）。"""
    params = get_exadis_params(state)
    force_obj, force_python = get_exadis_force(force_module, state, params)

    p = getattr(pyexadis, PARAMS_CLS)()
    p.temperature            = 300.0
    p.evalFrequency          = 1
    p.screwAngleTolerance    = 15.0
    p.bulkActivationEnergy   = 0.0            # 去势垒
    p.bulkActivationVolume   = 0.0            # 去 Escaig 依赖 -> exp()=1
    p.bulkAttemptFrequency   = 5.0e17
    p.bulkReferenceLength    = 1000.0
    p.minRunLength           = 0.0            # test.h 专属字段

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

        calforce  = CalForce(force_mode='LineTension', state=state)       # 局部力，含施加应力 PK；小系统稳定
        mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
        timeint   = TimeIntegration(integrator='Trapezoid', state=state, force=calforce, mobility=mobility)
        remesh    = Remesh(remesh_rule='LengthBased', state=state)
        cross_slip = make_cross_slip(state, calforce)
        vis       = VisualizeNetwork()

        applied_stress = build_applied_stress(TAU0)
        print('[ CrossSlipFCCTest (test.h, 修复版) | 应力驱动真实仿真 ]')
        print('  applied_stress Voigt =', applied_stress)
        print('  RSS(111)=0（交滑移前不弯），RSS(1 -1 1)=%.1f MPa（驱动交滑移）'
              % (0.816 * TAU0 / 1e6))
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
        print('预期：前 5 段螺型 run 头一两步交滑移到 (1 -1 1) 并在该面弯出。')
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
