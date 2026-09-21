import os, sys
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)
try:
    import pyexadis
    from pyexadis_base import ExaDisNet, DisNetManager
    from pyexadis_utils import insert_frank_read_src, write_vtk, write_data
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


def fcc_Ni_4um_frank_read():

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

    Lbox = 4.0e-6 / state["burgmag"]   # 模拟盒子边长 (b)，4 um
    rho = 3.0e12                       # 目标位错密度 (m^-2)
    Ldis_tot = rho * (Lbox * state["burgmag"])**3 / state["burgmag"]  # 目标位错总长度 (b)
    print(f"Lbox = {Lbox:.1f} b, total dislocation length: {Ldis_tot:.1f} b")

    # 臂长服从高斯分布，截断在 +/- 2 sigma 内（重采样而非截断到边界，避免在边界堆积）
    L_mean = 3000.0   # 臂长均值 (b)，约 747 nm ~ Lbox/4
    L_std = 600.0     # 臂长标准差 (b)，20% 的相对涨落
    L_min = L_mean - 2.0 * L_std   # 1800 b
    L_max = L_mean + 2.0 * L_std   # 4200 b
    # 最长臂 < Lbox/2：保证源跨周期边界后，段矢量的最小镜像判定无歧义
    assert L_max < 0.5 * Lbox, 'Arm length too long for the box: reduce L_mean / L_std'

    n_sys = len(FCC_SLIP_SYSTEMS)  # 滑移系数量 12
    # 先按平均臂长估源数，再向 12 取整：12 个滑移系严格等量分布
    N_dis = int(round(Ldis_tot / L_mean / n_sys)) * n_sys
    print(f"Generating {N_dis} Frank-Read sources ({N_dis // n_sys} per slip system)")

    min_center_dist = 600.0  # 源中心之间的最小间距 (b)，按周期最小镜像判定
    rng = np.random.default_rng(seed=42)

    # 每个滑移系出现 N_dis/12 次，再整体打乱 -> 各滑移系源数严格相等
    sys_ids = rng.permutation(np.tile(np.arange(n_sys), N_dis // n_sys))
    # +b / -b 各占一半后打乱，使净伯氏矢量尽量中性
    signs = rng.permutation(np.concatenate([np.ones(N_dis // 2),
                                            -np.ones(N_dis - N_dis // 2)]))

    # 周期盒。排斥判定要用 cell.closest_image，所以先建 cell
    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[1, 1, 1])

    nodes, segs = [], []
    centers = np.empty((N_dis, 3))  # 已放置源的中心坐标
    lengths = np.empty(N_dis)       # 已放置源的臂长

    for i in range(N_dis):
        # 高斯臂长，落在 [L_min, L_max] 之外就重采样
        while True:
            arm_length = rng.normal(L_mean, L_std)
            if L_min <= arm_length <= L_max:
                break

        # 源中心在盒内均匀随机；不设 margin，源可以跨越盒面
        for _ in range(10000):
            center = rng.uniform(0.0, Lbox, size=3)
            if i == 0:
                break
            # closest_image 给出已放置中心相对 center 的最近周期镜像，再取模得到最小镜像距离
            img = np.array(cell.closest_image(Rref=center, R=centers[:i]))
            if np.all(np.linalg.norm(img - center, axis=1) >= min_center_dist):
                break
        else:
            raise RuntimeError('Cannot place Frank-Read source: reduce min_center_dist or N_dis')
        centers[i], lengths[i] = center, arm_length

        b_vec, n_vec = FCC_SLIP_SYSTEMS[sys_ids[i]]
        # insert_frank_read_src 把 burg 原样写入 segs（只归一化 plane），必须传单位化的 b
        b_unit = signs[i] * b_vec / np.linalg.norm(b_vec)
        # 特征角：0-360 度随机。线向 = cos(theta)*b_hat + sin(theta)*(n_hat x b_hat)，恒在滑移面内
        theta = rng.uniform(0.0, 360.0)

        # 5 节点开放线段：PINNED(0) -- FREE(1) -- FREE(2) -- FREE(3) -- PINNED(4)
        # 两端钉扎，中间三节点自由；长于 maxseg 的段由 Remesh 自动细分
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b_unit, n_vec,
                                            arm_length, center, theta=theta, numnodes=5)
        print(f"  source {i:3d}: slip system {sys_ids[i]:2d}, sign {signs[i]:+.0f}, "
              f"L = {arm_length:7.1f} b, theta = {theta:6.1f} deg")

    print(f"Actual dislocation density: {lengths.sum() / Lbox**3 / state['burgmag']**2:.3e} m^-2")

    # 越界节点按周期边界折回盒内（write_data 本身也会折，这里显式做一次便于自检）
    nodes = np.array(nodes)
    nodes = np.hstack((np.array(cell.pbc_fold(nodes[:, :3])), nodes[:, 3:]))
    segs = np.vstack(segs)

    G = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    write_vtk(net, os.path.join(script_dir, 'fcc_Ni_4um_3e12_frank_read_1.vtk'), crystal='FCC')
    write_data(net, os.path.join(script_dir, 'fcc_Ni_4um_3e12_frank_read_1.data'))
    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_4um_frank_read()
