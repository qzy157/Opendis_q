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

def init_from_paradis_data_file(datafile):#定义函数 init_from_paradis_data_file，作用是从 paradis 格式的文件（位错模拟常用的初始配置文件）加载位错网络；
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

    pyexadis.initialize()#初始化 pyexadis 模块，准备进行位错动力学模拟。

    state = {
        "crystal": 'bcc',
        "burgmag": 2.7886e-10,
        "mu": 45.7e9,#剪切模量，单位为帕斯卡（Pa），表示材料的刚度。
        "nu": 0.336,#泊松比，表示材料在一个方向受拉伸时，在垂直方向上的收缩程度。
        "a": 1.0,#晶格常数，单位为米（m），表示晶体结构中原子之间的距离。
        "maxseg": 720.0,
        "minseg": 240.0,
        "rtol": 0.5,#相对误差容限，控制数值计算的精度。
        "rann": 0.5,#邻近距离，单位为米（m），用于判断位错线段之间是否相互作用。
        "nextdt": 1e-12,
        "maxdt": 1e-7,
        "split3node": 0,#是否允许在三节点处进行分割，0 表示不允许，1 表示允许。
        "use_glide_planes": 1,#1表示位错线段的运动约束到滑移面上。如果设置为0位错就可以在任意方向上运动，交滑移模块就无法进行。
        "num_bcc_plane_families": 1,#BCC晶体的滑移面族数量。仅仅启动{110}滑移面族，覆盖了BCC类型的默认行为。
    }
    erate = 3000.0#应变速率，单位为每秒（s^-1），表示材料在模拟过程中受到的拉伸或压缩速率。
    output_dir = 'output'

    restart_id = sys.argv[1] if len(sys.argv) > 1 else None#从命令行参数中获取 restart_id，如果没有提供，则默认为 None。这用于决定是从初始配置文件加载位错网络，还是从之前的模拟结果中恢复。
    if restart_id is None:#如果 restart_id 为 None，表示没有提供重启标识符，那么程序将从一个预定义的 paradis 数据文件中加载位错网络作为初始配置。否则，如果提供了 restart_id，程序将尝试从指定的重启文件中恢复之前的模拟状态。
        # Initial configuration from file
        data_filename = '../relax_high_t/output/config.90000.data'#(看图要在200 µs以后，采取config在2100之后的)
        print(f"init from {data_filename}")
        net, restart = init_from_paradis_data_file(data_filename)
    else:#如果提供了 restart_id，程序将尝试从指定的重启文件中恢复之前的模拟状态。重启文件通常包含了之前模拟的位错网络配置和相关状态信息，使得模拟可以从中断的地方继续进行。
        # Restart configuration
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"restart from {restart_filename}")
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, restart_filename))

    vis = VisualizeNetwork()#可视化工具，当前设置为 None，表示不使用可视化功能。如果需要进行模拟过程的可视化，可以将 vis 设置为一个适当的可视化对象。

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)#3*3*3 的网格用于计算位错之间的相互作用力。
    # 注意：Opendis_q 中温度相关 BCC mobility 注册名为 'BCC_0B_TEMP'（HIT fork 中为 'BCC_0B_temp'）
    mobility  = MobilityLaw(mobility_law='BCC_0B_TEMP', state=state, Mclimb=1e-6, kT=873.19, bT=1073.0, vmax=3400.0)#温度KT=873.19K，bT=1073.0K（起始温度为800℃）
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)#时间积分器，使用子循环方法（Subcycling）来处理不同时间尺度的位错运动。rgroups 定义了不同时间段的分组，用于调整时间步长以适应不同阶段的模拟需求。
    collision = Collision(collision_mode='Retroactive', state=state)#碰撞处理，使用回溯方法（Retroactive）来处理位错之间的碰撞事件。这种方法可以更准确地模拟位错在碰撞时的行为和结果。
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)#拓扑处理，使用并行方法（TopologyParallel）来处理位错网络的拓扑变化，如位错线段的连接和断开。这种方法可以提高模拟的效率和准确性，特别是在处理复杂的位错网络时。
    remesh    = Remesh(remesh_rule='LengthBased', state=state)#重网格化处理，使用基于长度的规则（LengthBased）来调整位错线段的网格。这种方法可以确保位错线段的长度在一定范围内，从而提高模拟的稳定性和准确性。
    # BCC Ti 热激活交滑移（无非施密特修正）
    # 温度随塑性应变升高：T = kT * pstrain + bT，与 mobility 完全一致
    cross_slip = CrossSlip(
        cross_slip_mode='ForceBasedSerial',
        state=state,
        force=calforce,
        kT          = 873.19,   # 升温斜率 [K/strain]，与 mobility 的 kT 相同
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
                              vis=vis, loading_mode='strain_rate', erate=erate, edir=np.array([0.,0.,1.]),# edir 定义了应变的方向，这里是沿着 z 轴施加拉伸。
                              max_strain=0.1, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=1, plot_pause_seconds=0.0001,#print_freq 定义了每隔多少步打印一次模拟信息，plot_freq 定义了每隔多少步更新一次可视化，plot_pause_seconds 定义了每次更新可视化时的暂停时间，以确保动画的流畅性。
                              write_freq=1, write_dir=output_dir, restart=restart)#每 100 步保存一次数据
    sim.run(net, state)

    pyexadis.finalize()


if __name__ == "__main__":
    bcc_Ti_3um_3e3()
