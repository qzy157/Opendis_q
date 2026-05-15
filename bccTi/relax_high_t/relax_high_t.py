import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')


def tau_f_cs_from_T(T):
    """阶梯取值：tau_f_cs 随温度的变化 [Pa]（BCC Ti 实验数据）。"""
    if   T <= 298.0:  return 230.0e6
    elif T <= 473.0:  return 185.3e6
    elif T <= 673.0:  return 148.3e6
    elif T <= 873.0:  return 126.4e6
    else:             return 119.7e6

def bcc_Ti_3um_relax():

    pyexadis.initialize()

    state = {
        "crystal": 'bcc',
        "burgmag": 2.7886e-10,
        "mu": 45.7e9,#剪切模量，单位为帕斯卡（Pa），表示材料的刚度。
        "nu": 0.336,#泊松比，表示材料在一个方向受拉伸时，在垂直方向上的收缩程度。
        "a": 1.0,#晶格常数，单位为米（m），表示晶体结构中原子之间的距离。
        "maxseg": 480.0,
        "minseg": 120.0,
        "rtol": 0.25,#相对误差容限，控制数值计算的精度。
        "rann": 0.5,#邻近距离，单位为米（m），用于判断位错线段之间是否相互作用。
        "nextdt": 1e-12,
        "maxdt": 1e-7,
        "split3node": 0,#是否允许在三节点处进行分割，0 表示不允许，1 表示允许。
        "use_glide_planes": 1,#是否使用滑移平面，1 表示使用，0 表示不使用。这表示你主动启用了 BCC 晶体的滑移面约束，覆盖了 BCC 类型的默认行为。
        "num_bcc_plane_families": 1,#BCC晶体的滑移面族数量。仅仅启动{110}滑移面族，覆盖了BCC类型的默认行为。
    }
    G = ExaDisNet()
    G.read_data('../init/3um_3e12_frank_read.data')
    net = DisNetManager(G)

    vis = None

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=3, cell=net.cell)
    # 注意：Opendis_q 中温度相关 BCC mobility 注册名为 'BCC_0B_TEMP'（HIT fork 中为 'BCC_0B_temp'）
    mobility  = MobilityLaw(mobility_law='BCC_0B_TEMP', state=state, Mclimb=1e-6, kT=0.0, bT=1073.0, vmax=3400.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    cross_slip = CrossSlip(
        cross_slip_mode='ForceBasedSerial',
        state=state,
        force=calforce,
        kT          = 0.0,   # 升温斜率 [K/strain]，与 mobility 的 kT 相同
        bT          = 1073.0,   # 初始温度 [K]，与 mobility 的 bT 相同（800°C）
        delta_H_cs  = 0.257,     # 零应力激活焓 [eV]，BCC Ti 估算值
        tau_P_cs    = 814.91e6,    # 交滑移面 Peierls 应力 [Pa]，BCC Ti 估算值
        p_shape     = 0.5614,   # Peierls 势形状参数 p（来自 mobility_bcc0b_temp.h）
        q_shape     = 0.5987,   # Peierls 势形状参数 q（来自 mobility_bcc0b_temp.h）
        delta_S_cs  = 0.0,      # 激活熵（暂取 0）[eV/K]
        omega_D     = 1e13,     # Debye 频率 [s^-1]
        eps_dot_sim = 3000.0,   # 模拟应变率 [s^-1]，与 erate 一致
        eps_dot_exp = 3000.0,     # 实验参考应变率 [s^-1]
        L0_ref      = 1e-6,     # 参考位错长度 [m]
        tau_f_cs    = tau_f_cs_from_T(1073.0),  # 交滑移面摩擦应力 [Pa]，按初始温度阶梯取值
    )

    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint,
                              collision=collision, topology=topology, remesh=remesh, cross_slip=cross_slip,
                              vis=vis, loading_mode="stress", applied_stress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),#不施加外部应力，进行自由弛豫
                              max_step=100000, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=2, plot_pause_seconds=0.0001,
                              write_freq=100, write_dir='output')#记录一次的时间间隔为100步
    sim.run(net, state)

    pyexadis.finalize()


if __name__ == "__main__":
    bcc_Ti_3um_relax()
