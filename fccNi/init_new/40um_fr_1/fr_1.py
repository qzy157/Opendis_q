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


def edge_line_direction(b_vec, n_vec):
    """纯刃型线向 = n × b（与 b 垂直，且在 {111} 滑移面内），单位向量"""
    arm_dir = np.cross(n_vec / np.linalg.norm(n_vec), b_vec / np.linalg.norm(b_vec))
    return arm_dir / np.linalg.norm(arm_dir)


def make_frank_read_source(center, b_vec, n_vec, arm_length):
    """
    生成单个纯刃型弗兰克-里德位错源（5节点开放线段）：
        PINNED(0) -- FREE(1) -- FREE(2) -- FREE(3) -- PINNED(4)
    两端钉扎，中间三节点自由，在应力下向外弓出并增殖。
    模拟中 Remesh 会把长于 maxseg 的段自动细分。
    """
    b_unit = b_vec / np.linalg.norm(b_vec)
    pn = n_vec / np.linalg.norm(n_vec)
    arm_dir = edge_line_direction(b_vec, n_vec)

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


def segment_distance(p1, q1, p2, q2):
    """线段 p1-q1 与 p2-q2 的最短距离（Ericson, Real-Time Collision Detection, 5.1.9）"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = np.dot(d1, d1), np.dot(d2, d2)
    b, c, f = np.dot(d1, d2), np.dot(d1, r), np.dot(d2, r)
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if denom > 1e-12 * a * e else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return np.linalg.norm((p1 + s * d1) - (p2 + t * d2))


def far_from_placed(center, arm_dir, arm_length, centers, dirs, lengths, Lbox, gap):
    """
    判断候选源与所有已放置源（取周期最小镜像）的线段最短距离是否都 >= gap。
    臂长 < Lbox/2 - gap 时，非最小镜像的中心距必然超过 (L_i+L_j)/2 + gap，只需检查最小镜像。
    """
    if len(centers) == 0:
        return True
    d = centers - center
    d -= Lbox * np.rint(d / Lbox)  # 周期最小镜像
    # 粗筛：中心距 >= 半臂长之和 + gap 的源不可能比 gap 更近
    near = np.linalg.norm(d, axis=1) < (lengths + arm_length) / 2.0 + gap
    half = 0.5 * arm_length * arm_dir
    for j in np.nonzero(near)[0]:
        cj = center + d[j]
        hj = 0.5 * lengths[j] * dirs[j]
        if segment_distance(center - half, center + half, cj - hj, cj + hj) < gap:
            return False
    return True


def fcc_Ni_40um_frank_read():

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

    Lbox = 40.0e-6 / state["burgmag"]  # 模拟盒子边长 (b)，40 um
    rho = 1.0e10                       # 位错密度 (m^-2)，与 10 um 保持一致
    Ldis_tot = rho * (Lbox * state["burgmag"])**3 / state["burgmag"]
    print(f"Lbox = {Lbox:.1f} b, total dislocation length: {Ldis_tot:.1f} b")

    # 臂长参考 bccTi 3um 构型的比例，取 10 um 盒子的 Lbox/6 ~ Lbox/3（约 1.7~3.3 um）
    # 所有盒子尺寸共用同一臂长：FR 开动应力 ~ mu*b/L 只由臂长决定，保证不同盒子尺寸可比
    Ldis_min = 6700.0   # 臂长下限 (b)
    Ldis_max = 13400.0  # 臂长上限 (b)
    N_dis = round(Ldis_tot / ((Ldis_min + Ldis_max) / 2))
    print(f"Generating {N_dis} Frank-Read sources")

    gap = 2000.0  # 源之间（含周期镜像）线段最短距离下限 (b)
    assert Ldis_max < Lbox / 2.0 - gap, 'Ldis_max too long for minimum-image distance check'
    rng = np.random.default_rng(seed=42)
    # 12 个滑移系的随机排列首尾拼接后取前 N_dis 个：
    # N_dis < 12 时随机选取互不相同的滑移系，N_dis >= 12 时各滑移系源数相差不超过 1
    n_sys = len(FCC_SLIP_SYSTEMS)
    sys_ids = np.concatenate([rng.permutation(n_sys) for _ in range(-(-N_dis // n_sys))])[:N_dis]

    all_rn = []
    all_links = []
    centers = np.empty((N_dis, 3))
    dirs = np.empty((N_dis, 3))
    lengths = np.empty(N_dis)
    node_offset = 0

    for i in range(N_dis):
        arm_length = rng.uniform(Ldis_min, Ldis_max)
        b_vec, n_vec = FCC_SLIP_SYSTEMS[sys_ids[i]]
        b_vec = rng.choice([-1.0, 1.0]) * b_vec  # +b / -b 随机
        arm_dir = edge_line_direction(b_vec, n_vec)

        # 周期盒子内中心任意取，按周期最小镜像保证与已放置源的线段距离 >= gap
        for _ in range(10000):
            center = rng.uniform(0.0, Lbox, size=3)
            if far_from_placed(center, arm_dir, arm_length, centers[:i], dirs[:i], lengths[:i], Lbox, gap):
                break
        else:
            raise RuntimeError('Cannot place Frank-Read source: enlarge the box, or reduce Ldis_max / gap')
        centers[i], dirs[i], lengths[i] = center, arm_dir, arm_length

        rn, links = make_frank_read_source(center, b_vec, n_vec, arm_length)
        rn[:, :3] = np.mod(rn[:, :3], Lbox)  # 节点折回盒内，跨边界线段由 ExaDiS 按最小镜像处理
        links[:, 0] += node_offset
        links[:, 1] += node_offset
        all_rn.append(rn)
        all_links.append(links)
        node_offset += 5  # 每个 FRS 含 5 个节点
        print(f"  source {i:3d}: slip system {sys_ids[i]:2d}, L = {arm_length:.1f} b")

    print(f"Actual dislocation density: {lengths.sum() / Lbox**3 / state['burgmag']**2:.3e} m^-2")

    all_rn = np.vstack(all_rn)
    all_links = np.vstack(all_links)

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[1, 1, 1])
    G = ExaDisNet(cell, all_rn, all_links)
    net = DisNetManager(G)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    write_vtk(net, os.path.join(script_dir, 'fcc_Ni_40um_1e10_frank_read_1.vtk'), crystal='FCC')
    write_data(net, os.path.join(script_dir, 'fcc_Ni_40um_1e10_frank_read_1.data'))
    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_40um_frank_read()
