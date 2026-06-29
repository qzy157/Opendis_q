import os, sys
import numpy as np

# Import pyexadis
pyexadis_paths = ['/data/home/dg000246d/Opendis_q/core/exadis/python/']
[sys.path.append(os.path.abspath(path)) for path in pyexadis_paths if not path in sys.path]
np.set_printoptions(threshold=20, edgeitems=5)

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager, SimulateNetworkPerf, read_restart, VisualizeNetwork
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


def make_cross_slip_fcc_test(state, force_module, temperature=300.0):
    """构建使用 cross_slip_fcc_test.h（CrossSlipFCCTest）的交滑移对象。

    对应 C++ 类 CrossSlipFCCTest：沿物理臂切出"最长连续螺型+同面+同向"run，
    每个 run 单独成链（修复了旧 CrossSlipFCCThermal 用整臂弦向判螺型、把弯臂
    整条丢弃的 bug）。算法参考 Hussein et al., Acta Materialia 85 (2015) 180-190。

    注意（与模板 yes.py 的写法不同，这里用的是真正存在的绑定）：
      * 绑定函数是 make_cross_slip_fcc_test(params, force, fcc_params)，
        不是 make_cross_slip(..., cs_thermal_params=...)（后者只支持
        ForceBasedParallel/ForceBasedSerial/None 这几种模式）。
      * 参数类名是 CrossSlipFCCTest_Params（带下划线）。
    """
    params = get_exadis_params(state)
    force_obj, force_python = get_exadis_force(force_module, state, params)

    # CrossSlipFCCTest::Params 的默认值即物理合理值（Ea=0.8eV, Va=20 b^3, nu=5e17 ...）
    cs_p = pyexadis.CrossSlipFCCTest_Params()
    cs_p.temperature = temperature
    # minRunLength: 螺型 run 最小长度（b）。0 = 不设下限会在网格尺度触发伪交滑移；
    # 取 4×minseg(=1200 b ≈ 306 nm)，过滤强度与 wansheng 的 minChainSegments=4 可比
    cs_p.minRunLength = 1200.0

    # 绕过 CrossSlip.__init__，直接拼出 SimulateNetworkPerf 需要的 CrossSlip 包装对象：
    #   - 必须是 CrossSlip 实例（SimulateNetworkPerf 用 isinstance 校验）
    #   - .cross_slip 存 CrossSlipBind（会被交给 native driver 的 set_modules）
    xs = CrossSlip.__new__(CrossSlip)
    xs.cross_slip_mode = 'FCCTest'
    xs.force_python = force_python
    xs.cross_slip = pyexadis.make_cross_slip_fcc_test(
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
        "use_glide_planes": True,   # CrossSlipFCCTest::handle() 要求 use_glide_planes=true（FCC 默认也已是 1）
    }

    output_dir = 'output_cross_slip_test'

    restart_id = sys.argv[1] if len(sys.argv) > 1 else None

    if restart_id is None:
        data_filename = '/data/home/dg000246d/Opendis_q/examples/10_strain_hardening/180chains_16.10e.data'
        print(f"init from {data_filename}")
        net, restart = init_from_paradis_data_file(data_filename)
    else:
        restart_filename = f'restart.{restart_id}.exadis'
        print(f"restart from {restart_filename}")
        net, restart = read_restart(state=state, restart_file=os.path.join(output_dir, restart_filename))

    vis = VisualizeNetwork()

    calforce  = CalForce(force_mode='SUBCYCLING_MODEL', state=state, Ngrid=64, cell=net.cell)
    mobility  = MobilityLaw(mobility_law='FCC_0', state=state, Medge=64103.0, Mscrew=64103.0, vmax=4000.0)
    timeint   = TimeIntegration(integrator='Subcycling', rgroups=[0.0, 100.0, 600.0, 1600.0], state=state, force=calforce, mobility=mobility)
    collision = Collision(collision_mode='Retroactive', state=state)
    topology  = Topology(topology_mode='TopologyParallel', state=state, force=calforce, mobility=mobility)
    remesh    = Remesh(remesh_rule='LengthBased', state=state)

    cross_slip = make_cross_slip_fcc_test(state, calforce, temperature=300.0)

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
