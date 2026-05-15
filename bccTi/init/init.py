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


# BCC ½<111> 柏格斯矢量方向（4个独立方向）
BCC_BURGERS = [
    np.array([ 1.,  1.,  1.]),
    np.array([-1.,  1.,  1.]),
    np.array([ 1., -1.,  1.]),
    np.array([ 1.,  1., -1.]),
]


def make_frank_read_source(center, b_vec, arm_length):
    """
    生成单个弗兰克-里德位错源（5节点开放线段）：
        PINNED(0) -- FREE(1) -- FREE(2) -- FREE(3) -- PINNED(4)
    两端钉扎，中间三节点自由，在应力下向外弓出并增殖。
    """
    b_unit = b_vec / np.linalg.norm(b_vec)

    # 臂方向：与 b 垂直（在滑移面内）
    ref = np.array([0., 0., 1.])
    if abs(np.dot(b_unit, ref)) > 0.9:
        ref = np.array([1., 0., 0.])
    arm_dir = np.cross(b_unit, ref)
    arm_dir /= np.linalg.norm(arm_dir)

    # 滑移面法向量：n = b × ξ
    pn = np.cross(b_unit, arm_dir)
    pn /= np.linalg.norm(pn)

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


def bcc_Ti_3um_frank_read():

    pyexadis.initialize()

    state = {
        "crystal": 'bcc',
        "burgmag": 2.7886e-10,
        "mu": 45.7e9,
        "nu": 0.336,
        "a": 1.0,
        "maxseg": 480.0,
        "minseg": 120.0,
        "rtol": 0.25,
        "rann": 0.5,
        "nextdt": 1e-12,
        "maxdt": 1e-7,
        "split3node": 0,
        "use_glide_planes": 1,
        "num_bcc_plane_families": 1,
    }

    Lbox = 12000.0  # 模拟盒子边长 (nm)
    rho = 2.0e12    # 位错密度 (m^-2)
    Ldis_tot = rho * (Lbox * state["burgmag"])**3 / state["burgmag"]
    print(f"Total dislocation length: {Ldis_tot:.1f} nm")

    Ldis_min = 2000.0  # 臂长下限 (nm)
    Ldis_max = 4000.0  # 臂长上限 (nm)
    # 每个源活动臂长度约为 arm_length
    N_dis = round(Ldis_tot / ((Ldis_min + Ldis_max) / 2))
    print(f"Generating {N_dis} Frank-Read sources")

    rng = np.random.default_rng(seed=42)
    # margin 保证两端钉扎节点（距中心 arm/2）不超出盒子
    margin = Ldis_max / 2.0

    all_rn = []
    all_links = []
    node_offset = 0

    for _ in range(N_dis):
        arm_length = rng.uniform(Ldis_min, Ldis_max)
        center = rng.uniform(margin, Lbox - margin, size=3)
        b_vec = BCC_BURGERS[rng.integers(0, 4)]

        rn, links = make_frank_read_source(center, b_vec, arm_length)
        links[:, 0] += node_offset
        links[:, 1] += node_offset
        all_rn.append(rn)
        all_links.append(links)
        node_offset += 5  # 每个 FRS 含 5 个节点

    all_rn = np.vstack(all_rn)
    all_links = np.vstack(all_links)

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[1, 1, 1])
    G = ExaDisNet(cell, all_rn, all_links)
    net = DisNetManager(G)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    write_vtk(net, os.path.join(script_dir, '3um_3e12_frank_read.vtk'))
    write_data(net, os.path.join(script_dir, '3um_3e12_frank_read.data'))
    pyexadis.finalize()


if __name__ == "__main__":
    bcc_Ti_3um_frank_read()
