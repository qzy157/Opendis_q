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
    arm_dir = np.cross(n_vec / np.linalg.norm(n_vec), b_vec / np.linalg.norm(b_vec))#先把滑移面法向 n 和柏氏矢量 b 分别除以自身模长，变成单位向量 n̂、b̂，做叉积 ξ = n̂ × b̂。
    return arm_dir / np.linalg.norm(arm_dir)#得到纯刃型线向的单位向量 ξ̂ = ξ / |ξ|


def make_frank_read_source(center, b_vec, n_vec, arm_length):
    """
    生成单个纯刃型弗兰克-里德位错源（5节点开放线段）：
        PINNED(0) -- FREE(1) -- FREE(2) -- FREE(3) -- PINNED(4)
    两端钉扎，中间三节点自由，在应力下向外弓出并增殖。
    模拟中 Remesh 会把长于 maxseg 的段自动细分。
    """
    b_unit = b_vec / np.linalg.norm(b_vec)#柏氏矢量的单位向量
    pn = n_vec / np.linalg.norm(n_vec)#滑移面法向的单位向量
    arm_dir = edge_line_direction(b_vec, n_vec)#纯刃型线向的单位向量

    # 沿臂方向均匀分布 5 个节点，间距 arm_length/4
    pts = [center + (i - 2) * (arm_length / 4.0) * arm_dir for i in range(5)]#range(5) 让 i 取 0,1,2,3,4 (i - 2) 依次是 −2, −1, 0, +1, +2，以中间节点为 0 左右对称

    rn = np.array([
        [pts[0][0], pts[0][1], pts[0][2], NodeConstraints.PINNED_NODE],
        [pts[1][0], pts[1][1], pts[1][2], NodeConstraints.UNCONSTRAINED],
        [pts[2][0], pts[2][1], pts[2][2], NodeConstraints.UNCONSTRAINED],
        [pts[3][0], pts[3][1], pts[3][2], NodeConstraints.UNCONSTRAINED],
        [pts[4][0], pts[4][1], pts[4][2], NodeConstraints.PINNED_NODE],
    ])# 前三列：坐标 pts[i][0], pts[i][1], pts[i][2] pts[0][0] 是第 0 个节点的 x  pts[0][1] 是第 0 个节点的 y pts[0][2] 是第 0 个节点的 z

    links = np.array([
        [0, 1, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
        [1, 2, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
        [2, 3, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
        [3, 4, b_unit[0], b_unit[1], b_unit[2], pn[0], pn[1], pn[2]],
    ])#规定 5 个节点怎么连成位错线，以及每段位错的柏氏矢量和滑移面。

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
    Ldis_tot = rho * (Lbox * state["burgmag"])**3 / state["burgmag"]# 总位错长度 (b)
    print(f"Lbox = {Lbox:.1f} b, total dislocation length: {Ldis_tot:.1f} b")#输出盒子边长和目标位错总长度

    # 臂长参考 bccTi 3um 构型的比例，取 10 um 盒子的 Lbox/6 ~ Lbox/3（约 1.7~3.3 um）
    # 所有盒子尺寸共用同一臂长：FR 开动应力 ~ mu*b/L 只由臂长决定，保证不同盒子尺寸可比
    Ldis_min = 6700.0   # 臂长下限 (b)
    Ldis_max = 13400.0  # 臂长上限 (b)
    N_dis = round(Ldis_tot / ((Ldis_min + Ldis_max) / 2))#决定要生成几个 Frank-Read 源：用目标位错总长度除以平均臂长，再取整。
    print(f"Generating {N_dis} Frank-Read sources")

    gap = 2000.0  # 源之间的最小间距 (b)
    rng = np.random.default_rng(seed=42)#创建一个随机数生成器 rng，并把种子固定为 42
    # margin 保证两端钉扎节点不超出盒子（源不跨周期边界），且跨周期边界的镜像间距也 >= gap
    margin = Ldis_max / 2.0 + gap / 2.0
    assert Lbox > 2.0 * margin, 'Box too small: reduce Ldis_max / gap'#assert 条件, '报错信息'  条件为 True：什么都不做，继续往下执行 条件为 False：抛出 AssertionError: Box too small: reduce Ldis_max / gap
    # 12 个滑移系的随机排列首尾拼接后取前 N_dis 个：
    # N_dis < 12 时随机选取互不相同的滑移系，N_dis >= 12 时各滑移系源数相差不超过 1
    n_sys = len(FCC_SLIP_SYSTEMS) # 滑移系数量12
    sys_ids = np.concatenate([rng.permutation(n_sys) for _ in range(-(-N_dis // n_sys))])[:N_dis]#-(-N_dis // n_sys) 向上取整除法 ⌈N_dis / 12⌉// 是向下取整，对负数取整再取负，就得到向上取整，不用 import math； rng.permutation(12)  0–11 的一个随机排列，比如 [7, 2, 11, 0, ...]； [... for _ in range(k)] 把 k 个排列首尾拼成一个长度 12k 的数组；[:N_dis] 只取前 N_dis 个，作为每个源的滑移系编号
    #源较多时（比如 N_dis = 26）：⌈26/12⌉ = 3，拼出 36 个编号，取前 26 个。前两个完整排列让每个滑移系各出现 2 次，第三个排列只取前 2 个，所以有 2 个滑移系各出现 3 次。各滑移系的源数最多相差 1。
    all_rn = []#收集每个源的 rn（5×4）
    all_links = []#收集每个源的 links（4×8）
    centers = np.empty((N_dis, 3))#记录已放置源的中心坐标。
    lengths = np.empty(N_dis)#记录每个源的臂长。
    node_offset = 0#记录已放置源的节点数，用于后续源的节点编号偏移。第 i 个源的局部编号 0–4加上 5i变成全局编号，每放一个源加 5

    for i in range(N_dis):
        arm_length = rng.uniform(Ldis_min, Ldis_max)#臂长 L ∈ [6700, 13400] b，均匀分布
        b_vec, n_vec = FCC_SLIP_SYSTEMS[sys_ids[i]]#按 111 行分好的编号取滑移系
        b_vec = rng.choice([-1.0, 1.0]) * b_vec  # +b / -b 随机

        # 中心距 >= Ldis_max + gap，保证两个源的线段之间距离 >= gap
        for _ in range(10000):
            center = rng.uniform(margin, Lbox - margin, size=3)#随机生成一个中心坐标 center，范围在 [margin, Lbox - margin]，保证源不跨周期边界
            if np.all(np.linalg.norm(centers[:i] - center, axis=1) >= Ldis_max + gap):#计算已放置源的中心坐标 centers[:i] 与新生成的中心坐标 center 的距离，np.linalg.norm(..., axis=1) 计算每个源的距离，np.all(...) 判断是否所有距离都 >= Ldis_max + gap，如果是就 break 跳出循环，接受这个中心坐标；否则继续循环尝试新的随机中心坐标。
                break#找到一个合适的中心坐标 离所有已放置源的中心都足够远就接受
        else:
            raise RuntimeError('Cannot place Frank-Read source: enlarge the box, or reduce Ldis_max / gap')#如果循环 10000 次都找不到合适的中心坐标，就抛出 RuntimeError 异常，提示用户增大盒子尺寸或减小 Ldis_max / gap。
        centers[i], lengths[i] = center, arm_length#记录已放置源的中心坐标和臂长。

        rn, links = make_frank_read_source(center, b_vec, n_vec, arm_length)#用这个源的中心、柏氏矢量、滑移面法向和臂长，生成 5 个节点（rn，5×4）和 4 条线段（links，4×8）。节点编号是局部的 0–4。
        links[:, 0] += node_offset# 起点编号 n1 links[:, 0] 取所有行的第 0 列，+= 对这一整列原地加上偏移量
        links[:, 1] += node_offset# 终点编号 n2
        all_rn.append(rn)#收集每个源的 rn（5×4）
        all_links.append(links)#收集每个源的 links（4×8）
        node_offset += 5  # 每个 FRS 含 5 个节点
        print(f"  source {i:2d}: slip system {sys_ids[i]:2d}, L = {arm_length:.1f} b")#输出每个源的编号、滑移系编号和臂长。

    print(f"Actual dislocation density: {lengths.sum() / Lbox**3 / state['burgmag']**2:.3e} m^-2")#输出实际生成的位错密度，计算方法是所有源的臂长之和除以盒子体积再除以柏氏矢量模长平方。

    all_rn = np.vstack(all_rn)#把所有源的 rn（5×4）按行堆叠成一个大数组，形状是 (N_dis*5, 4)，每行是一个节点的坐标和约束条件。
    all_links = np.vstack(all_links)#把所有源的 links（4×8）按行堆叠成一个大数组，形状是 (N_dis*4, 8)，每行是一个线段的起点编号、终点编号、柏氏矢量和滑移面法向。

    cell = pyexadis.Cell(h=Lbox * np.eye(3), is_periodic=[1, 1, 1])#创建一个周期性立方体单元格，边长为 Lbox，h 是一个 3×3 的对角矩阵，表示单元格的三个边向量，is_periodic=[1, 1, 1] 表示在 x、y、z 三个方向上都是周期性的。
    G = ExaDisNet(cell, all_rn, all_links)#创建一个 ExaDisNet 对象 G，包含单元格 cell、所有节点的坐标和约束条件 all_rn，以及所有线段的起点编号、终点编号、柏氏矢量和滑移面法向 all_links。
    net = DisNetManager(G)#创建一个 DisNetManager 对象 net，用于管理位错网络 G。

    script_dir = os.path.dirname(os.path.abspath(__file__))#获取当前脚本的绝对路径，并取其目录部分，作为输出文件的保存路径。
    write_vtk(net, os.path.join(script_dir, 'fcc_Ni_10um_1e10_frank_read_1.vtk'), crystal='FCC')
    write_data(net, os.path.join(script_dir, 'fcc_Ni_10um_1e10_frank_read_1.data'))
    pyexadis.finalize()


if __name__ == "__main__":
    fcc_Ni_10um_frank_read()
