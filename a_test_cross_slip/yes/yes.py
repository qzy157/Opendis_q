import os, sys
import numpy as np

# Import pyexadis
pyexadis_paths = ['/data/home/dg000246d/Opendis_q/core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]
np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart
    from pyexadis_base import CalForce, MobilityLaw, TimeIntegration, Collision, Topology, Remesh, CrossSlip
    from pyexadis_base import get_exadis_params, get_exadis_force
except ImportError:
    raise ImportError('Cannot import pyexadis')


def init_from_paradis_data_file(datafile):
    G = ExaDisNet()
    G.read_paradis(datafile)
    net = DisNetManager(G)
    restart = None
    return net, restart


def make_cross_slip_fcc_thermal(state, force_module, temperature=300.0):
    """Build a CrossSlip object using the thermally-activated FCC cross-slip
    algorithm described in Hussein et al., Acta Materialia 85 (2015): 180-190.
    Parameters follow the default values in CrossSlipFCCThermal::Params.
    """
    params = get_exadis_params(state)
    force_obj, force_python = get_exadis_force(force_module, state, params)

    # Use default parameters from CrossSlipFCCThermal::Params
    # 注意：绑定的类名带下划线（CrossSlipFCCThermal_Params），见 exadis_pybind.cpp:1299
    cs_p = pyexadis.CrossSlipFCCThermal_Params()
    cs_p.temperature = temperature

    # CrossSlipFCCThermal（旧版、带"弯臂整条丢弃"物理 bug）由专用函数构造，
    # 不能走 make_cross_slip（后者只支持 ForceBasedParallel/ForceBasedSerial/None）。
    xs = CrossSlip.__new__(CrossSlip)
    xs.cross_slip_mode = 'FCCThermal'
    xs.force_python = force_python
    xs.cross_slip = pyexadis.make_cross_slip_fcc_thermal(
        params=params,
        force=force_obj,
        fcc_params=cs_p,
    )
    return xs


def example_fcc_Cu_15um_1e3_with_cross_slip():
    pyexadis.initialize()

    state = {
        "crystal": 'fcc',
        "burgmag": 2.55e-10,
        "mu": 54.6e9,
        "nu": 0.324,
        "a": 6.0,
        "maxseg": 2000.0,
        "minseg": 300.0,
        "rtol": 10.0,
        "rann": 10.0,
        "nextdt": 1e-10,
        "maxdt": 1e-9,
        "use_glide_planes": True,   # CrossSlipFCCThermal::handle() 要求 use_glide_planes=true（与 yes_test 配置对齐）
    }

    output_dir = 'output_with_cross_slip'

    restart_id = sys.argv[1] if len(sys.argv) > 1 else None

    if restart_id is None:
        data_filename = '/data/home/dg000246d/Opendis_q/examples/10_strain_hardening/180chains_16.10e.data'
        print(f"init from {data_filename}")
        net, restart = init_from_paradis_data_file(data_filename)
    else:
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"restart from {restart_filename}")
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, restart_filename))

    vis = None

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    cross_slip = make_cross_slip_fcc_thermal(state, calforce, temperature=300.0)

    sim = SimulateNetworkPerf(calforce=calforce, mobility=mobility, timeint=timeint,
                              collision=collision, topology=topology, remesh=remesh,
                              cross_slip=cross_slip, vis=vis,
                              loading_mode='strain_rate', erate=1e3, edir=np.array([0., 0., 1.]),
                              max_strain=0.01, burgmag=state["burgmag"], state=state,
                              print_freq=1, plot_freq=10, plot_pause_seconds=0.0001,
                              write_freq=1, write_dir=output_dir, restart=restart)
    sim.run(net, state)

    pyexadis.finalize()


if __name__ == "__main__":
    example_fcc_Cu_15um_1e3_with_cross_slip()