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


def fcc_Ni_3um_frank_read():

    pyexadis.initialize()

    state = {
        "crystal": 'fcc',
        "burgmag": 2.49e-10,
        "mu": 76.0e9,
        "nu": 0.31,
        "a": 1.0,
        # !!! maxseg/minseg 与本构型的臂长（~271 b）强耦合，下游 relax / strain_rate
        # 脚本必须照抄这两个值。.data 文件不携带它们（write_data 只写网络本身），
        # 若沿用 10/40/65/100 um 系列的 500/125，段长（~50 b）会低于 minseg，
        # remesh 会把每个源的自由节点并掉、只剩 PINNED--PINNED 段，源永久失效。
        "maxseg": 80.0,
        "minseg": 20.0,
        "rtol": 0.25,
        "rann": 0.5,
        "nextdt": 1e-12,
        "maxdt": 1e-7,
        "use_glide_planes": 1,
    }

    Lbox = 3.0e-6 / state["burgmag"]   # 模拟盒子边长 (b)，3 um
    rho = 1.0e10                       # 目标位错密度 (m^-2)，与 init_same 其它盒子尺寸一致
    Ldis_tot = rho * (Lbox * state["burgmag"])**3 / state["burgmag"]  # 目标位错总长度 (b)
    print(f"Lbox = {Lbox:.1f} b, total dislocation length: {Ldis_tot:.1f} b")

    rng = np.random.default_rng(seed=42)

    # 密度固定时 源数 x 臂长 = Ldis_tot，所以增源数只能靠缩臂长，代价是 FR 开动应力
    # tau ~ mu*b/L 同比例升高（tau[MPa] ~ 76000 / arm[b]）。L_target = 300 b 对应
    # 3/4/5 um 盒子的 4/9/17 个源、tau ~ 257-280 MPa，离 mu/100 = 760 MPa 还有余量。
    L_target = 300.0           # 目标臂长 (b)，约 75 nm
    L_std = 0.1 * L_target     # 10% 相对涨落：保留源强度的随机性，又不让 -2sigma 尾巴逼近 minseg
    L_lo, L_hi = L_target - 2.0 * L_std, L_target + 2.0 * L_std

    N_dis = max(1, round(Ldis_tot / L_target))
    print(f"Generating {N_dis} Frank-Read sources")

    # 臂长服从高斯分布，截断在 +/- 2 sigma 内（重采样而非截断到边界，避免在边界堆积）
    lengths = np.empty(N_dis)   # 每个源的臂长 (b)
    for i in range(N_dis):
        while True:
            L = rng.normal(L_target, L_std)
            if L_lo <= L <= L_hi:
                lengths[i] = L
                break
    # 整体缩放使总长严格等于 Ldis_tot -> 实际位错密度精确命中 rho
    lengths *= Ldis_tot / lengths.sum()

    # 每个源的段数：让初始段长落在 [minseg, maxseg] 的中点附近。
    # 最少 2 段 = 3 节点（PINNED--FREE--PINNED），保证至少有一个自由节点，源才能弓出
    seg_target = 0.5 * (state["minseg"] + state["maxseg"])
    nsegs = np.maximum(2, np.round(lengths / seg_target).astype(int))
    seglen = lengths / nsegs

    # 最长臂 < Lbox/2：保证源跨周期边界后，段矢量的最小镜像判定无歧义
    assert lengths.max() < 0.5 * Lbox, 'Arm length too long for the box: reduce L_target'
    # 段长必须高于 minseg，否则 remesh 会把自由节点并掉；3 节点源的自由节点一旦
    # 被并掉就只剩 PINNED--PINNED 段，而细化又跳过这种段，源永久失效
    assert seglen.min() >= 1.3 * state["minseg"], 'Segment too close to minseg: lower minseg or raise L_target'
    assert seglen.max() <= state["maxseg"], 'Segment above maxseg: remesh would refine at step 0'

    n_sys = len(FCC_SLIP_SYSTEMS)  # 滑移系数量 12
    # 12 个滑移系的随机排列首尾拼接后取前 N_dis 个：
    # N_dis < 12 时随机选取互不相同的滑移系，N_dis >= 12 时各滑移系源数相差不超过 1
    sys_ids = np.concatenate([rng.permutation(n_sys) for _ in range(-(-N_dis // n_sys))])[:N_dis]
    # +b / -b 各占一半后打乱，使净伯氏矢量尽量中性
    signs = rng.permutation(np.concatenate([np.ones(N_dis // 2),
                                            -np.ones(N_dis - N_dis // 2)]))

    min_center_dist = 600.0  # 源中心之间的最小间距 (b)，按周期最小镜像判定

    # 周期盒。排斥判定要用 cell.closest_image，所以先建 cell
    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[1, 1, 1])

    nodes, segs = [], []
    centers = np.empty((N_dis, 3))  # 已放置源的中心坐标

    for i in range(N_dis):
        arm_length = lengths[i]

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
        centers[i] = center

        b_vec, n_vec = FCC_SLIP_SYSTEMS[sys_ids[i]]
        # insert_frank_read_src 把 burg 原样写入 segs（只归一化 plane），必须传单位化的 b
        b_unit = signs[i] * b_vec / np.linalg.norm(b_vec)
        # 特征角：0-360 度随机。线向 = cos(theta)*b_hat + sin(theta)*(n_hat x b_hat)，恒在滑移面内
        theta = rng.uniform(0.0, 360.0)

        # 开放线段，两端 PINNED、中间节点自由；段数由 nsegs[i] 定，段长已核到 [minseg, maxseg] 内
        nodes, segs = insert_frank_read_src(cell, nodes, segs, b_unit, n_vec,
                                            arm_length, center, theta=theta,
                                            numnodes=int(nsegs[i]) + 1)
        print(f"  source {i:3d}: slip system {sys_ids[i]:2d}, sign {signs[i]:+.0f}, "
              f"L = {arm_length:7.1f} b, {nsegs[i]:2d} segs x {seglen[i]:5.1f} b, "
              f"theta = {theta:6.1f} deg")

    print(f"Actual dislocation density: {lengths.sum() / Lbox**3 / state['burgmag']**2:.3e} m^-2")

    # 越界节点按周期边界折回盒内（write_data 本身也会折，这里显式做一次便于自检）
    nodes = np.array(nodes)
    nodes = np.hstack((np.array(cell.pbc_fold(nodes[:, :3])), nodes[:, 3:]))
    segs = np.vstack(segs)

    G = ExaDisNet(cell, nodes, segs)
    net = DisNetManager(G)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    write_vtk(net, os.path.join(script_dir, 'fcc_Ni_3um_1e10_frank_read_1.vtk'), crystal='FCC')
    write_data(net, os.path.join(script_dir, 'fcc_Ni_3um_1e10_frank_read_1.data'))
    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_3um_frank_read()
