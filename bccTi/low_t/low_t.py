import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart, VisualizeNetwork
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')

def init_from_paradis_data_file(datafile):
    G = ExaDisNet()
    G.read_paradis(datafile)
    net = DisNetManager(G)
    restart = None
    return net, restart

def tau_f_cs_from_T(T):
    """阶梯取值：tau_f_cs 随温度的变化 [Pa]（BCC Ti 实验数据）。"""
    if   T <= 298.0:  return 230.0e6
    elif T <= 473.0:  return 185.3e6
    elif T <= 673.0:  return 148.3e6
    elif T <= 873.0:  return 126.4e6
    else:             return 119.7e6


def bcc_Ti_3um_3e3():

    pyexadis.initialize()

    state = {
        "crystal": 'bcc',
        "burgmag": 2.7886e-10,
        "mu": 45.7e9,
        "nu": 0.336,
        "a": 1.0,
        "maxseg": 720.0,
        "minseg": 240.0,
        "rtol": 0.05,   # 原 0.5 太宽,外层 dt 每步 ×1.2 暴涨,Subcycling 内循环数 ~ vmax·dt/minseg 失控;低温位错速度低,容差再激进一档
        "rann": 0.5,
        "nextdt": 1e-12,
        "maxdt": 5e-10,  # 原 1e-7 配合 vmax=3400 m/s 子循环数爆炸,单步 ~22h;压到 5e-10,每外层步应变增量 1.5e-6 已足够粗
        "split3node": 0,
        "use_glide_planes": 1,
        "num_bcc_plane_families": 1,
    }
    erate = 3000.0
    output_dir = 'output'

    restart_id = sys.argv[1] if len(sys.argv) > 1 else None
    if restart_id is None:
        # Initial configuration from file
        data_filename = '../relax_low_t/output/config.90000.data'#看图要在120µs以后，采取config3800
        print(f"init from {data_filename}")
        net, restart = init_from_paradis_data_file(data_filename)
    else:
        # Restart configuration
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"restart from {restart_filename}")
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, restart_filename))

    vis = VisualizeNetwork()

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    # 注意：Opendis_q 中温度相关 BCC mobility 注册名为 'BCC_0B_TEMP'（HIT fork 中为 'BCC_0B_temp'）
    mobility  = MobilityLaw(mobility_law='BCC_0B_TEMP', state=state, Mclimb=1e-6, kT=1534.16, bT=298.0, vmax=3400.0)#温度kT=1534.16K，bT=298.0K(起始温度为25℃)，vmax=3400.0m/s(声速)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    # BCC Ti 热激活交滑移（无非施密特修正）
    # 温度随塑性应变升高：T = kT * pstrain + bT，与 mobility 完全一致
    cross_slip = CrossSlip(
        cross_slip_mode='ForceBasedSerial',
        state=state,
        force=calforce,
        kT          = 1534.16,  # 升温斜率 [K/strain]，与 mobility 的 kT 相同
        bT          = 298.0,    # 初始温度 [K]，与 mobility 的 bT 相同（25°C）
        delta_H_cs  = 0.257,     # 零应力激活焓 [eV]，BCC Ti 估算值
        tau_P_cs    = 814.91e6,    # 交滑移面 Peierls 应力 [Pa]，BCC Ti 估算值
        p_shape     = 0.5614,   # Peierls 势形状参数 p（来自 mobility_bcc0b_temp.h）
        q_shape     = 0.5987,   # Peierls 势形状参数 q（来自 mobility_bcc0b_temp.h）
        delta_S_cs  = 0.0,      # 激活熵（暂取 0）[eV/K]
        omega_D     = 1e13,     # Debye 频率 [s^-1]
        eps_dot_sim = 3000.0,   # 模拟应变率 [s^-1]，与 erate 一致
        eps_dot_exp = 3000.0,     # 实验参考应变率 [s^-1]
        L0_ref      = 1e-6,     # 参考位错长度 [m]
        tau_f_cs    = tau_f_cs_from_T(298.0),   # 交滑移面摩擦应力 [Pa]，BCC Ti 估算值
    )

    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint,
                              collision=collision, topology=topology, remesh=remesh, cross_slip=cross_slip,
                              vis=vis, loading_mode='strain_rate', erate=erate, edir=np.array([0.,0.,1.]),
                              max_strain=0.1, burgmag=state["burgmag"], state=state,
                              print_freq=10, plot_freq=10, plot_pause_seconds=0.0001,
                              write_freq=10, write_dir=output_dir, restart=restart)
    sim.run(net, state)

    pyexadis.finalize()


if __name__ == "__main__":
    bcc_Ti_3um_3e3()
