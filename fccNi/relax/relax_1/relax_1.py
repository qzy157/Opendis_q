import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, get_exadis_params
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
except ImportError:
    raise ImportError('Cannot import pyexadis')


def fcc_Ni_10um_relax():

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
        "split3node": 0,
        "use_glide_planes": 1,
        "num_bcc_plane_families": 1,
    }
    G = ExaDisNet()
    G.read_data('/data/home/dg000246d/Opendis_q/fccNi/init/fr_1/fcc_Ni_10um_1e10_frank_read_1.data')
    net = DisNetManager(G)

    vis = None

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=3, cell=net.cell)
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
                              vis=vis, loading_mode="stress", applied_stress=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                              max_step=100000, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=2, plot_pause_seconds=0.0001,
                              write_freq=100, write_dir='output')
    sim.run(net, state)

    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_10um_relax()
