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


def force_100_percent(cs_p):
    """把所有热激活势垒清零，让"能交滑移的位错"必然（100%）交滑移。

    交滑移概率（handle_bulk / handle_intersection 内）：
        prob = nu * dt * (L / L_ref) * exp(-(E_a - V_a*dSigma_E) / (kB*T))
    随机门槛是 `if (prob < 1.0 && uniform01(rng) > prob) return;`。

    令所有 *ActivationEnergy = 0、*ActivationVolume = 0：
        exponent = 0 -> exp = 1
        prob = nu * dt * (L/L_ref) ~= 5e17 * 1e-10 * (L/1000) >> 1
    于是 `prob < 1.0` 恒为假，随机掷骰被完全跳过 -> 100% 触发。

    注意：这里只去掉"概率因素"，并未动确定性的 Schmid 门槛
        |tau_cs| >= mu*b/(10L)  且  |tau_cs| >= 1.1|tau_glide|
    这两条仍然保留，它们才是"这条位错能不能交滑移"的物理判据。
    这样三种模块（test/thermal/wansheng）唯一的差异就落在"链构型"上，
    可以干净地检验 thermal 的链构型 bug。
    """
    # Bulk
    cs_p.bulkActivationEnergy      = 0.0
    cs_p.bulkActivationVolume      = 0.0
    # Hirth lock (intersection, attractive)
    cs_p.hirthActivationEnergy     = 0.0
    cs_p.hirthActivationVolume     = 0.0
    # Glide lock (intersection, attractive)
    cs_p.glideLockActivationEnergy = 0.0
    cs_p.glideLockActivationVolume = 0.0
    # LC lock (intersection, attractive)
    cs_p.lcLockActivationEnergy    = 0.0
    cs_p.lcLockActivationVolume    = 0.0
    return cs_p


def make_cross_slip_fcc_wansheng(state, force_module, temperature=300.0):
    """构建使用 cross_slip_fcc_wansheng.h（CrossSlipFCCWansheng）的交滑移对象。

    对应 C++ 类 CrossSlipFCCWansheng：先做全臂的 FCC 过滤（b∈<110>），再逐 segment
    判 15° 螺型角，把臂切成若干"极大连续螺型 run"，面的 {111}/共面判据下放到 run 内部
    逐段核对，每条 run 至少 minChainSegments 段才成链——同样修复了旧 CrossSlipFCCThermal
    把弯臂整条丢弃的 bug。参考 Hussein et al., Acta Materialia 85 (2015) 180-190。

    本文件是 100% 版本：在参考 wansheng.py 的基础上把热激活势垒清零，
    使得能交滑移的位错必然发生交滑移（见 force_100_percent）。
    """
    params = get_exadis_params(state)
    force_obj, force_python = get_exadis_force(force_module, state, params)

    # CrossSlipFCCWansheng::Params 的默认值即物理合理值（Ea=0.8eV, Va=20 b^3, nu=5e17 ...）
    cs_p = pyexadis.CrossSlipFCCWansheng_Params()
    cs_p.temperature = temperature
    # minChainSegments: 一条螺型子链允许的最少 segment 数。与参考 wansheng.py 保持一致（1）。
    cs_p.minChainSegments = 1
    # 去掉概率因素：所有势垒清零 -> prob >> 1 -> 100% 交滑移
    force_100_percent(cs_p)

    # 绕过 CrossSlip.__init__，直接拼出 SimulateNetworkPerf 需要的 CrossSlip 包装对象：
    #   - 必须是 CrossSlip 实例（SimulateNetworkPerf 用 isinstance 校验）
    #   - .cross_slip 存 CrossSlipBind（会被交给 native driver 的 set_modules）
    xs = CrossSlip.__new__(CrossSlip)
    xs.cross_slip_mode = 'FCCWansheng'
    xs.force_python = force_python
    xs.cross_slip = pyexadis.make_cross_slip_fcc_wansheng(
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
        "use_glide_planes": True,   # CrossSlipFCCWansheng::handle() 要求 use_glide_planes=true（FCC 默认也已是 1）
    }

    output_dir = 'output_cross_slip_wansheng_100'

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

    cross_slip = make_cross_slip_fcc_wansheng(state, calforce, temperature=300.0)

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
