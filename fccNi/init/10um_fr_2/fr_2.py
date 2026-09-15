import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, NodeConstraints, DisNetManager
    from pyexadis_utils import write_vtk, write_data
except ImportError:
    raise ImportError('Cannot import pyexadis')


# FCC 1/2<110>{111} 的 12 个滑移系：(柏氏矢量 b, 滑移面法向 n)
FCC_SLIP_SYSTEMS = [
    (np.array([0.,  1., -1.]), np.array([ 1.,  1.,  1.])),
    (np.array([1.,  0., -1.]), np.array([ 1.,  1.,  1.])),
    (np.array([1., -1.,  0.]), np.array([ 1.,  1.,  1.])),
    (np.array([0.,  1., -1.]), np.array([-1.,  1.,  1.])),
    (np.array([1.,  0.,  1.]), np.array([-1.,  1.,  1.])),
    (np.array([1.,  1.,  0.]), np.array([-1.,  1.,  1.])),
    (np.array([0.,  1.,  1.]), np.array([ 1., -1.,  1.])),
    (np.array([1.,  0., -1.]), np.array([ 1., -1.,  1.])),
    (np.array([1.,  1.,  0.]), np.array([ 1., -1.,  1.])),
    (np.array([0.,  1.,  1.]), np.array([ 1.,  1., -1.])),
    (np.array([1.,  0.,  1.]), np.array([ 1.,  1., -1.])),
    (np.array([1., -1.,  0.]), np.array([ 1.,  1., -1.])),
]


def make_frank_read_source(center, b_vec, n_vec, arm_length):
    """
    生成单个纯刃型弗兰克-里德位错源（5节点开放线段）：
        PINNED(0) -- FREE(1) -- FREE(2) -- FREE(3) -- PINNED(4)
    两端钉扎，中间三节点自由，在应力下向外弓出并增殖。
    模拟中 Remesh 会把长于 maxseg 的段自动细分。
    """
    b_unit = b_vec / np.linalg.norm(b_vec)
    pn = n_vec / np.linalg.norm(n_vec)

    # 臂方向：纯刃型，线向 = n × b（与 b 垂直，且在 {111} 滑移面内）
    arm_dir = np.cross(pn, b_unit)
    arm_dir /= np.linalg.norm(arm_dir)

    # 沿臂方向均匀分布 5 个节点，间距 arm_length/4
    pts = [center + (i - 2) * (arm_length / 4.0) * arm_dir for i in range(5)]

    rn = np.array([
        [pts[0][0], pts[0][1], pts[0][2], NodeConstraints.PINNED_NODE],
        [pts[1][0], pts[1][1], pts[1][2], NodeConstraints.UNCONSTRAINED],
        [pts[2][0], pts[2][1], pts[2][2], NodeConstraints.UNCONSTRAINED],
        [pts[3][0], pts[3][1], pts[3][2], NodeConstraints.UNCONSTRAINED],
        [pts[4][0], pts[4][1], pts[4][2], NodeConstraints.PINNED_NODE],
    ])

    links = np.array([
        [0, 1, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
        [1, 2, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
        [2, 3, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
        [3, 4, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
    ])

    return rn, links


def fcc_Ni_10um_frank_read():

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

    Lbox = 10.0e-6 / state["burgmag"]  # 模拟盒子边长 (b)，10 um
    rho = 1.0e10                       # 位错密度 (m^-2)
    Ldis_tot = rho * (Lbox * state["burgmag"])**3 / state["burgmag"]
    print(f"Lbox = {Lbox:.1f} b, total dislocation length: {Ldis_tot:.1f} b")

    Ldis_min = 2400.0  # 臂长下限 (b)
    Ldis_max = 4300.0  # 臂长上限 (b)
    N_dis = round(Ldis_tot / ((Ldis_min + Ldis_max) / 2))
    print(f"Generating {N_dis} Frank-Read sources")

    gap = 2000.0  # 源之间的最小间距 (b)
    rng = np.random.default_rng(seed=43)
    # margin 保证两端钉扎节点不超出盒子，且跨周期边界的镜像间距也 >= gap
    margin = Ldis_max / 2.0 + gap / 2.0
    # 12 个滑移系轮流分配后打乱，N_dis=12 时每个滑移系恰好一个源
    sys_ids = rng.permutation(np.arange(N_dis) % len(FCC_SLIP_SYSTEMS))

    all_rn = []
    all_links = []
    centers = []
    node_offset = 0
    L_sum = 0.0

    for i in range(N_dis):
        arm_length = rng.uniform(Ldis_min, Ldis_max)

        # 中心距 >= Ldis_max + gap，保证两个源的线段之间距离 >= gap
        for _ in range(10000):
            center = rng.uniform(margin, Lbox - margin, size=3)
            if all(np.linalg.norm(center - c) >= Ldis_max + gap for c in centers):
                break
        else:
            raise RuntimeError('Cannot place Frank-Read source: enlarge the box, or reduce Ldis_max / gap')
        centers.append(center)

        b_vec, n_vec = FCC_SLIP_SYSTEMS[sys_ids[i]]
        b_vec = rng.choice([-1.0, 1.0]) * b_vec  # +b / -b 随机

        rn, links = make_frank_read_source(center, b_vec, n_vec, arm_length)
        links[:, 0] += node_offset
        links[:, 1] += node_offset
        all_rn.append(rn)
        all_links.append(links)
        node_offset += 5  # 每个 FRS 含 5 个节点
        L_sum += arm_length
        print(f"  source {i:2d}: slip system {sys_ids[i]:2d}, L = {arm_length:.1f} b")

    print(f"Actual dislocation density: {L_sum / Lbox**3 / state['burgmag']**2:.3e} m^-2")

    all_rn = np.vstack(all_rn)
    all_links = np.vstack(all_links)

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[1, 1, 1])
    G = ExaDisNet(cell, all_rn, all_links)
    net = DisNetManager(G)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    write_vtk(net, os.path.join(script_dir, 'fcc_Ni_10um_1e10_frank_read_2.vtk'), crystal='FCC')
    write_data(net, os.path.join(script_dir, 'fcc_Ni_10um_1e10_frank_read_2.data'))
    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_10um_frank_read()
