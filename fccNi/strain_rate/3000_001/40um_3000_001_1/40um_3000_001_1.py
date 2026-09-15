import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')


def init_from_paradis_data_file(datafile):
    G = ExaDisNet()
    G.read_paradis(datafile)
    net = DisNetManager(G)
    restart = None
    return net, restart


def fcc_Ni_40um_3e3_001():

    pyexadis.initialize()

    state = {
        "crystal": 'fcc',
        "burgmag": 2.49e-10,
        "mu": 76.0e9,
        "nu": 0.31,
        "a": 1.0,
        "maxseg": 500.0,
        "minseg": 125.0,
        "rtol": 0.25,
        "rann": 0.5,
        "nextdt": 1e-12,
        "maxdt": 1e-7,
        "use_glide_planes": 1,
    }
    # 与 bccTi 保持一致取正值；按 driver.cpp 中 dstrain = erate*dt，erate > 0 实际为沿 edir 单轴拉伸
    erate = 3000.0
    edir = np.array([0., 0., 1.])  # [001] 加载方向
    output_dir = 'output'

    restart_id = sys.argv[1] if len(sys.argv) > 1 else None
    if restart_id is None:
        # 初始构型：relax_1 松弛后的位错网络
        data_filename = '/data/home/dg000246d/Opendis_q/fccNi/relax/relax_1/output/config.data'
        print(f"init from {data_filename}")
        net, restart = init_from_paradis_data_file(data_filename)
    else:
        # 从 output/restart.<id>.exadis 续跑
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"restart from {restart_filename}")
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, restart_filename))

    vis = None

    # 模块设置与 relax_1.py 保持一致
    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)  # 迁移率暂用 Cu 示例值
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)
    # wansheng 热激活交滑移（cross_slip_fcc_wansheng.h）；未给出的参数用默认值（如 bulk Ea=0.8 eV）
    cross_slip = CrossSlip(cross_slip_mode='FCCWansheng', state=state, force=calforce,
                           temperature=300.0, minChainSegments=4)

    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint,
                              collision=collision, topology=topology, remesh=remesh, cross_slip=cross_slip,
                              vis=vis, loading_mode='strain_rate', erate=erate, edir=edir,
                              max_strain=0.1, burgmag=state["burgmag"], state=state,  # |strain| 达到 0.1 停止
                              print_freq=1, plot_freq=2, plot_pause_seconds=0.0001,
                              write_freq=100, write_dir=output_dir, restart=restart)
    sim.run(net, state)

    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_40um_3e3_001()
