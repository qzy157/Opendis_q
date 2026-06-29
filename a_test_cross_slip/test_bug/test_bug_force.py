#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_bug_force.py — “加上力”版：用一个超临界交滑移力，测出 FCC 交滑移的“弯臂漏检”bug。

与同目录的 test_bug.py 的区别
-----------------------------
  - test_bug.py  ：节点力为 0，只检验上一层“螺型链是否被正确构建”(build_screw_chains)，
                   靠 stderr 的 [OLD-CS]/[NEW-CS] 调试行数链。它证明不了“交滑移到底有没有发生”。
  - 本脚本       ：给螺型 run 施加一个【足够大的交滑移驱动力】，让交滑移真正“执行”，
                   然后直接看网络有没有变——以此把 bug 测出来。

构型（与 bug.h / test_bug.py 完全一致）
--------------------------------------
一条 ~900 b 的位错臂：前 5 段是沿 1/2[1 0 -1] 的理想螺型、共面于 (111)，末端 2 段拐弯成非螺型。

bug 的本质
----------
  旧实现 cross_slip_fcc_thermal.h：用整条臂“首尾弦向量”判螺型。弯臂弦向偏离 b ~38°(>15°容差)
    → 整条臂被判为非螺型而整体丢弃 → build_screw_chains 返回 0 条链 → handle_bulk 永远不会被调用。
    后果：**无论施加多大的交滑移力，旧实现都不会执行交滑移**（力被白白浪费）。← 这就是 bug

  新实现 cross_slip_fcc_test.h：沿臂切出“最长连续螺型+同面+同向”run（前 5 段）→ 1 条链 →
    在本脚本的超临界力下 handle_bulk 通过应力阈值并触发交滑移 → 前 5 段的滑移面从 (111)
    翻到交滑移面 (1 -1 1)。← 修复后

  wansheng 实现 cross_slip_fcc_wansheng.h：思路与新实现一致——逐 segment 判螺型、切出极大
    连续螺型 run（前 5 段，>= minChainSegments=4）→ 1 条链 → handle_bulk 在同一条超临界力下
    触发交滑移 → 前 5 段滑移面翻到 (1 -1 1)。本脚本的第 [3] 段就是为覆盖 wansheng 的
    “执行”路径（应力阈值/热激活/翻面）而加，test_bug.py 只覆盖了它的建链层。

怎么“测出来”
------------
交滑移一旦执行，execute_crossslip 会把 run 上每个 segment 的滑移面改成交滑移面（update_seg_plane）。
所以我们在 handle 前后对比【每个 segment 的滑移面法向】：
  - 旧实现：没有链 → 没有交滑移 → 7 段滑移面全保持 (111)，**0 段改变**  → 复现 bug
  - 新实现：螺型 run 交滑移 → 第 0..4 段滑移面变成 (1 -1 1)，**5 段改变**  → bug 已修复
（注：这条 run 本来就是一条笔直的理想螺型，节点已落在螺型线上，所以投影后节点几乎不动——
  能观测到的唯一变化就是“滑移面翻面”，这正是判据。）

为什么这个力一定能触发交滑移
----------------------------
handle_bulk 的三道关卡（Hussein et al. 2015, Sec.2）：
  1. |tau_cs|  >= MU*b/(10*L)              —— 交滑移面 Schmid 应力够大
  2. |tau_cs|  >= 1.1*|tau_glide|          —— 比原滑移面大 10% 以上（防止来回横跳）
  3. 热激活概率 P = nu*dt*(L/Lref)*exp(-(Ea - Va*dSigma_E)/(kB T)) 抽样命中
我们把每个节点的力都设成 +MAG * cs_line（cs_line=交滑移面内的刃方向），于是
  tau_cs    = |F·cs_line|/L 很大；tau_glide = |F·glide_line|/L = 0.33*tau_cs（两个{111}刃方向夹角~70.5°）
  → 关卡 1、2 轻松通过。再把 Ea=0、Va=0 → exp(...)=1，P=nu*dt*(L/Lref)≈2.5e5 >> 1 → 关卡 3 必中。
这样就把“能不能触发交滑移”完全归结为“螺型链有没有被建出来”——也就单刀直入地测到了 bug。

用法
----
    cd Opendis_q/a_test_cross_slip/test_bug
    python3 test_bug_force.py
注意：CrossSlipFCCTest 是新加进绑定的，必须先在 Linux 上重新编译 pyexadis 才能用：
    cd Opendis_q && cmake --build build -j8 && cmake --build build --target install
"""

import os
import sys
import contextlib
import tempfile
import numpy as np

# Import pyexadis
pyexadis_path = '/data/home/dg000246d/Opendis_q/core/exadis/python/'
if not pyexadis_path in sys.path: sys.path.append(pyexadis_path)

try:
    import pyexadis
    from pyexadis_base import ExaDisNet, NodeConstraints, get_exadis_params, get_exadis_force, CalForce
except ImportError as e:
    raise ImportError('无法导入 pyexadis（请确认已编译并安装）: %s' % e)


# ---------------------------------------------------------------------------
#  在 C/C++ 层捕获 fprintf(stderr, ...) 的输出（[OLD-CS]/[NEW-CS] 调试行）
#  Python 的 sys.stderr 重定向抓不到 C 层直接写 fd=2 的内容，必须做 fd 级重定向。
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def capture_c_stderr(sink):
    stderr_fd = 2# 标准错误的 fd 号,固定是 2
    saved_fd = os.dup(stderr_fd)# 复制一份 fd=2,留个"备份" os.dup(2) 把 fd=2 当前指向的东西(通常是终端)复制出一个新 fd(比如 fd=7),记为 saved_fd。这相当于先拍张快照,记住"stderr 本来通往哪儿",结束时好还原。
    tmp = tempfile.TemporaryFile(mode='w+b')# 开个临时文件,等会儿让输出灌进去  tmp 是个二进制临时文件,当作"接水的桶"
    try:
        sys.stderr.flush() # 先把 Python 缓冲里的 stderr 内容刷干净
        os.dup2(tmp.fileno(), stderr_fd) # 关键:让 fd=2 改指向 tmp 文件
        yield # 把控制权交还给 with 块,让里面的代码跑
    finally:
        sys.stderr.flush() # 再刷一次,确保 with 块里 Python 那侧的输出也落地
        os.dup2(saved_fd, stderr_fd) # 把 fd=2 还原回原来的终端(用之前的备份)
        os.close(saved_fd)# 备份 fd 用完了,关掉,避免泄漏
        tmp.seek(0) # 文件读写指针拨回开头
        sink.append(tmp.read().decode('utf-8', errors='replace'))# 读出全部内容,解码成字符串,塞进 sink
        tmp.close()# 关临时文件(临时文件关闭即自动删除)


# ---------------------------------------------------------------------------
#  几何：Burgers / 滑移面 / 交滑移面 / 力的方向，全部与 C++ 内部算法一致
# ---------------------------------------------------------------------------
B_VEC      = np.array([0.5, 0.0, -0.5])           # 1/2[1 0 -1]
GLIDE_PLANE = np.array([1.0, 1.0, 1.0])           # (111)


def get_crossslip_plane(glide_plane, burg):
    """复刻 C++ get_crossslip_plane：b 哪个分量为 0，就把面法向那个分量翻号。"""
    tol = 1e-6
    bn = burg / np.linalg.norm(burg)
    g = glide_plane / np.linalg.norm(glide_plane)
    if abs(np.dot(bn, g)) > tol:
        return np.zeros(3)
    if abs(bn[0]) < tol: return np.array([-g[0],  g[1],  g[2]])
    if abs(bn[1]) < tol: return np.array([ g[0], -g[1],  g[2]])
    if abs(bn[2]) < tol: return np.array([ g[0],  g[1], -g[2]])
    return np.zeros(3)


def unit(v):
    n = np.linalg.norm(v)# 算 v 的模长(欧几里得长度)
    return v / n if n > 0 else v# 长度>0 就除以长度归一化,否则原样返回


def build_bent_arm_network():
    """复刻 bug.h 的弯臂网络：8 节点、7 段，b=1/2[1 0 -1]，plane=(111)。"""
    L = 1.0e5                                       # b 单位的大盒子，避免 ~900 b 的臂被 PBC 折叠
    cell = pyexadis.Cell(h=L * np.eye(3), is_periodic=[1, 1, 1])

    b = B_VEC.copy()# B_VEC = (0.5, 0, -0.5) = ½[1 0 -1]
    plane = unit(GLIDE_PLANE)# GLIDE_PLANE = (1,1,1),归一化成 (0.577,0.577,0.577)

    P = np.array([
        [  0.000,   0.000,    0.000],
        [ 70.711,   0.000,  -70.711],
        [141.421,   0.000, -141.421],
        [212.132,   0.000, -212.132],
        [282.843,   0.000, -282.843],
        [353.553,   0.000, -353.553],               # 0..5：沿 b 的理想螺型（5 段，每段 ~100 b，共 ~500 b） 节点 0–5:沿 (70.711, 0, -70.711) 的整数倍排开,方向正是 b=½[1 0 -1] 的方向(x 正、z 负、y=0),每段长约 100 b(√(70.711²+70.711²)=100),5 段共约 500 b。这一段是理想螺型——段方向和 b 平行,夹角 0°。
        [311.018, 230.179, -541.196],               # 5..6：拐弯，非螺型 y 分量突然跳到 230、460,方向偏离 b,这就是拐弯的非螺型段(我们算过,与 b 成 70°)。
        [268.482, 460.357, -728.839],               # 6..7：拐弯，非螺型
    ])
    P = P + np.array([L / 2, L / 2, L / 2])          # 平移到盒子中心，彻底避开周期边界折叠歧义（平移不变）

    nodes = [np.concatenate((P[i], [NodeConstraints.UNCONSTRAINED])) for i in range(len(P))]# ExaDiS 要求每个节点是一行 [x, y, z, constraint]——前三个是坐标,第四个是约束标志。这里对每个节点把坐标 P[i] 和约束标志拼在一起。UNCONSTRAINED 表示节点不受约束、可自由移动(交滑移时节点能被投影/移动)。列表推导对 8 个节点各拼一行。
    segs = [np.concatenate(([i, i + 1], b, plane)) for i in range(len(P) - 1)]#[i, i+1]:第 i 段连接节点 i 和 i+1(链式串起来:0-1、1-2、…、6-7)。 b:所有段共用同一个 Burgers 矢量 ½[1 0 -1]。 plane:所有段共用同一个滑移面法向 (111)。
    return ExaDisNet(cell, np.array(nodes), np.array(segs))#把盒子、节点表、段表打包成一个 ExaDisNet,这就是一个完整的位错网络,可以直接喂给 pyexadis.handle_cross_slip(...)。


def apply_crossslip_force(G, mag):
    """给所有节点施加 +mag * cs_line 的力（cs_line = 交滑移面内刃方向）。

    这样合力强烈投影在交滑移面刃方向上，使 |tau_cs| 远超阈值、且 > 1.1|tau_glide|。
    """
    cs_plane = get_crossslip_plane(GLIDE_PLANE, B_VEC)        # (1 -1 1)
    cs_line  = unit(np.cross(B_VEC, cs_plane))                # 交滑移面内的刃方向 把力设成沿 cs_line,是为了让 dot(total_force, cs_line) 吃满(投影最大化),|tau_cs| 因此很大;而投影到 glide 面刃方向的 |tau_glide| 只有它的约 1/3(两条 {111} 刃方向夹角 70.5°,cos=1/3)。这样 handle_bulk 的两道关卡——|tau_cs| ≥ MU·b/(10L) 和 |tau_cs| ≥ 1.1|tau_glide|——都轻松通过。注释里"远超阈值、且 > 1.1|tau_glide|"说的就是这个。
    N = G.num_nodes() # 节点总数(这里是 8)
    forces = np.tile(mag * cs_line, (N, 1))# 把 mag*cs_line 复制成 N 行 mag * cs_line 是单个节点要施加的力矢量(方向 cs_line、大小 mag,比如 mag=1e4)。 np.tile(向量, (N, 1)) 把这个 1×3 的力垂直复制 N 份,堆成一个 N×3 的数组——即每个节点都拿到完全相同的力。这正是你之前画图时"8 个节点同方向同大小力箭头"的来历。
    tags = G.get_tags()#tags 是 ExaDiS 里每个节点的标识符(通常是 (domain, index) 这类元组),set_forces 需要靠 tags 把每行力对应到正确的节点上——保证"第 i 行力"落到"第 i 个节点"身上,而不是顺序错位。
    G.net.set_forces(forces, tags)#G.net.set_forces(forces, tags) 把这 N×3 的力数组真正写入网络节点的力字段 nodes[].f。这一步是整个函数的实际效果所在。
    return cs_plane, cs_line#把交滑移面法向和刃方向返回给调用方。


def count_changed_planes(planes_before, planes_after):
    """统计滑移面法向方向发生改变的 segment 个数（对符号/归一化稳健）。"""
    changed = []
    for i in range(len(planes_before)):
        a = unit(planes_before[i])# 第 i 段交滑移前的面法向,归一化
        b = unit(planes_after[i])# 第 i 段交滑移后的面法向,归一化
        cosang = abs(float(np.dot(a, b)))# 两单位向量点积取绝对值 = |cos(夹角)|
        if cosang < 0.99:                                    # 偏离原方向 > ~8° 即认为翻面了 |cos|<0.99 → 夹角 > ~8° → 判为翻面
            changed.append(i)
    return changed


def run_handler_with_force(make_name, params, force, cs_params, mag):#make_name:要测的实现对应的工厂函数名字符串,比如 'make_cross_slip_fcc_test' mag:施加力的大小(主脚本里 MAG=1e4)。
    """新建网络→施力→建 handler→执行 handle_cross_slip，返回 (ok, 改变的段, cs_plane, stderr日志)。"""
    if not hasattr(pyexadis, make_name):
        return False, None, None, ''

    G = build_bent_arm_network()#build_bent_arm_network() 造那条弯臂(每段面都设成 (111))。注意:每次调用都新建一个全新网络,所以三个实现互不污染,各测各的干净构型。
    planes_before = G.get_segs_data()["planes"].copy()       # handle 前：应全是 (111) get_segs_data()["planes"] 取出当前每段的滑移面法向,此刻应该全是 (111)。.copy() 很关键——必须深拷贝存一份快照,否则后面交滑移改了网络数据,这个引用会跟着变,前后对比就失效了(拿到的"before"会变成"after")。
    cs_plane, _ = apply_crossslip_force(G, mag)#给所有节点装上 mag · cs_line 的力并写进网络(上一轮拆过)。返回的 cs_plane(=交滑移目标面 (1 -1 1))留着,后面返回出去供主脚本核对翻面方向;cs_line 用不上,用 _ 丢弃。

    make_fn = getattr(pyexadis, make_name)#getattr(pyexadis, make_name) 按名字字符串取出那个工厂函数(和第 1 步的 hasattr 配套——先确认存在,再取出来用)。这种"用字符串动态选函数"的写法,让同一段代码能复用在三个实现上,不用写三份。
    cs = make_fn(params=params, force=force, fcc_params=cs_params)#make_fn(...) 调用工厂,造出一个具体的交滑移处理器对象 cs(对应 C++ 的 CrossSlipFCCTest / …Wansheng / …Thermal),并把参数喂进去。

    log = []
    with capture_c_stderr(log):
        pyexadis.handle_cross_slip(net=G.net, cross_slip=cs)#在 capture_c_stderr 的保护下调用 handle_cross_slip——这是真正干活的一步:它内部走 build_screw_chains(建链)→ handle_bulk(过应力阈值/热激活)→ execute_crossslip(投影节点 + update_seg_plane 翻面)。期间 C++ 打的 [OLD-CS]/[NEW-CS]/[WS-CS] 调试行被接进 log。

    planes_after = G.get_segs_data()["planes"].copy()        # handle 后：变了的段=交滑移执行过的段
    changed = count_changed_planes(planes_before, planes_after)#再取一次每段的面(此刻交滑移过的段已被改写),.copy() 同样存快照。然后 count_changed_planes 逐段比对前后法向,数出翻了面的段(偏离 > ~8° 的),返回它们的下标列表 changed。这就是"交滑移到底执行了没有"的硬证据。
    return True, (changed, planes_after), cs_plane, log[0]#True:测成了。 (changed, planes_after):翻面段下标 + 交滑移后的完整面数据(主脚本用 len(changed) 判 PASS/FAIL,用 planes_after[i] 打印每段翻到的新面方向)。 cs_plane:交滑移目标面 (1 -1 1),供核对。 log[0]:捕获到的那段 stderr 文本。


def make_deterministic_params(cls_params):
    """把热激活参数调成“必发生”：Ea=0、Va=0 → exp=1，P=nu*dt*(L/Lref) >> 1。
    同时设 screwAngleTolerance=15（与 bug 构型一致），新实现额外设 minRunLength=0。
    """
    cls_params.screwAngleTolerance   = 15.0
    cls_params.bulkActivationEnergy  = 0.0       # 去掉势垒
    cls_params.bulkActivationVolume  = 0.0       # 去掉 Escaig 依赖：exp(...)=1
    cls_params.bulkAttemptFrequency  = 5.0e17    # 默认；nu*dt*(L/Lref) ≈ 2.5e5 >> 1
    cls_params.bulkReferenceLength   = 1000.0
    cls_params.evalFrequency         = 1
    cls_params.temperature           = 300.0
    return cls_params


def main():
    pyexadis.initialize()
    try:
        # --- 最小 FCC 系统：默认取向 R = 单位阵 → Rinv = 单位阵（与 bug.h 一致）---
        state = {
            "crystal": "fcc",
            "burgmag": 2.49e-10,
            "mu": 76.0e9,
            "nu": 0.31,
            "a": 2.0,
            "maxseg": 2000.0,
            "minseg": 10.0,
            "rann": 1.0,
            "rtol": 2.0,
            "nextdt": 1e-12,
            "maxdt": 1e-10,
            "use_glide_planes": True,                # handle() 要求 use_glide_planes=true，否则直接 fatal
        }

        params = get_exadis_params(state)#state 是你前面定义的那个 Python 字典(crystal、burgmag、mu、nu、maxseg、nextdt、use_glide_planes 等)。get_exadis_params 把这个松散的字典转换成 ExaDiS C++ 内部要用的、结构化的参数对象 params。
        calforce = CalForce(state=state, force_mode='LineTension')   # handle 内部不重算力，仅为构造接口占位 CalForce 是 ExaDiS 里负责"算位错受力"的类。force_mode='LineTension' 指定用线张力模型这种最简单的力算法。
        force, _ = get_exadis_force(calforce, state, params)#get_exadis_force 根据上面的 calforce、state、params 产出 ExaDiS 内部的力对象 force(以及第二个返回值,这里用 _ 丢弃,通常是某种附带的配置或句柄)。

        # --- 施加的力：每节点 MAG*cs_line。MAG 远大于触发阈值所需的最小值 ---
        MAG = 1.0e4                                  # 远超临界：tau_cs/threshold ~ 2.6e4（见下方打印）
        L_run = 500.0                                # 螺型 run 总长 ~500 b（5 段×100 b）
        threshold = state["mu"] * state["burgmag"] / (10.0 * L_run)
        tau_cs_pred = 5.0 * MAG / L_run              # 合力在 6 节点端点半权下 = 5*MAG；投影到 cs_line 5.0 * MAG 是那个 "5 倍" 的来历——注释写"合力在 6 节点端点半权下 = 5*MAG"。回想 compute_chain_stresses 里 total_force = Σ 0.5*(f_nk + f_nk1),对一条 5 段、6 节点的链做这个"两端半权、内部全权"的加权和,结果等效于 (6−1)=5 个节点的全权力,即 5 * MAG。
        print("施力方案：每节点 F = %.1e * cs_line(交滑移面刃方向)" % MAG)
        print("  预估 |tau_cs|   ≈ %.4g" % tau_cs_pred)
        print("  交滑移阈值      = MU*b/(10*L) ≈ %.4g   →  余量 ~%.0f 倍" %
              (threshold, tau_cs_pred / threshold))#把施力方案、预估的 tau_cs、阈值,以及最关键的 tau_cs_pred / threshold(超出阈值多少倍) 打出来。注释里"tau_cs/threshold ~ 2.6e4"说的就是这个比值——约 2.6 万倍余量,根本不可能因为"力不够"而过不了第一关。%.4g/%.0f 是格式化(有效数字/整数)。
        print("  热激活：Ea=0, Va=0 → P = nu*dt*(L/Lref) ≈ %.2g  (>>1，必触发)\n"
              % (5.0e17 * state["nextdt"] * (L_run / 1000.0)))#这是复刻 handle_bulk 的第三关——热激活概率 P = nu*dt*(L/Lref)*exp(-(Ea-Va*dSigma)/(kBT))。因为 make_deterministic_params 把 Ea=0、Va=0 → 指数项 exp(0)=1,所以 P 退化成 nu*dt*(L/Lref)。代入 nu=5e17、dt=nextdt=1e-12、L/Lref=500/1000:P ≈ 5e17 * 1e-12 * 0.5 = 2.5e5,远大于 1 → 概率抽样必中(prob>=1 则必发生)。打印出来让你确认这一关也稳过。

        # ============ 旧实现（对照，应复现 bug：施力后仍 0 段交滑移）============
        print("=" * 72)
        print("[1] 旧实现 CrossSlipFCCThermal —— 期望：施了超临界力，仍然 0 段交滑移（bug）")
        print("=" * 72)#"=" * 72 是把等号字符重复 72 次拼成一条分隔横线(Python 字符串的乘法),纯粹是排版,让输出分块清晰。中间那行写明这一段测的是谁、期望是什么——对应你截图里那条 ====... [1] 旧实现 ... 的标题。
        cs_p_old = make_deterministic_params(pyexadis.CrossSlipFCCThermal_Params())#pyexadis.CrossSlipFCCThermal_Params() 新建一个旧实现专属的参数对象(对应 C++ 里 CrossSlipFCCThermal::Params)。 make_deterministic_params(...) 把它调成"必触发"配置:Ea=0、Va=0(去掉势垒和 Escaig 依赖 → 热激活概率退化成 nu*dt*(L/Lref)≈2.5e5)、screwAngleTolerance=15、evalFrequency=1 等。这样就保证——如果链能被建出来,后面的应力关卡和热激活关卡都会放行。注意这里没设 minRunLength,因为旧实现的 Params 没有这个字段(那是新实现才有的)。
        ok_old, res_old, csp_old, log_old = run_handler_with_force(
            'make_cross_slip_fcc_thermal', params, force, cs_p_old, MAG)#调用上一轮拆过的总调度函数,对旧实现执行"建网络→施力→建 handler→交滑移→比对翻面"。第一个参数 'make_cross_slip_fcc_thermal' 是旧实现的工厂函数名字符串。四个返回值: ok_old:是否测成(绑定里有没有这个工厂函数)。 res_old:(changed, planes_after) 元组——翻面段下标 + 交滑移后的面数据。csp_old:交滑移目标面 (1 -1 1)(这里用不到,但接住)。 log_old:捕获到的 C 层 stderr 文本(那条 [OLD-CS] … REJECT(whole arm))。
        if ok_old:
            sys.stdout.write(log_old)
            changed_old = res_old[0]
            print(">>> 旧实现：滑移面发生改变的 segment 数 = %d  %s\n"
                  % (len(changed_old), changed_old if changed_old else "(无)"))#sys.stdout.write(log_old):把捕获到的 stderr 调试行原样打到标准输出。 print(...):打印翻了面的段数 len(changed_old) 和具体下标。changed_old if changed_old else "(无)" 是个三元表达式:列表非空就打列表本身,空列表(falsy)就打 "(无)"。对旧实现,期望 len=0、打 (无)——对应截图里 >>> 旧实现：滑移面发生改变的 segment 数 = 0  (无)。
        else:
            changed_old = None
            print(">>> 绑定里没有 make_cross_slip_fcc_thermal（旧 .so？）跳过\n")#changed_old = None:用 None 标记"这个实现没测",和"测了但 0 段"(空列表)区分开。后面最终判定时会用 changed_old is None 来识别"未测",打"跳过"而不是误判成 bug 复现。 打印提示绑定里没有这个工厂函数(可能 .so 是旧的、没重新编译)。

        # ============ 新实现（被测，应修复：施力后 5 段交滑移）============
        print("=" * 72)
        print("[2] 新实现 CrossSlipFCCTest —— 期望：螺型 run(第0..4段)交滑移，5 段翻面")
        print("=" * 72)
        if hasattr(pyexadis, 'make_cross_slip_fcc_test'):#注意和 [1] 旧实现的结构差异:旧实现是先无条件调 run_handler_with_force、再用返回的 ok_old 判断;这里是在外层就先用 hasattr 把关。原因是新实现 CrossSlipFCCTest 是"刚加进绑定的",很可能没编译进 .so——所以提前检查,没有就直接走 else 给出明确的"请重新编译"提示。
            cs_p_new = make_deterministic_params(pyexadis.CrossSlipFCCTest_Params())
            cs_p_new.minRunLength = 0.0#cs_p_new.minRunLength = 0.0 是新实现独有的。回想 test.h 的建链判据:keep = is_111_family(run_plane) && (run_len >= minRunLength)。这里把长度门槛设成 0,意思是"对 run 长度不设限",确保段 0–4 那条 run 一定通过(不会因为长度门被卡)。旧实现没有这个字段(它用的是别的判据),所以 [1] 里没这行;wansheng 用的是 minChainSegments(段数门),字段名又不同。这三处参数差异正反映了三个实现建链逻辑的不同。
            ok_new, res_new, csp_new, log_new = run_handler_with_force(
                'make_cross_slip_fcc_test', params, force, cs_p_new, MAG)#调 run_handler_with_force,工厂名是 'make_cross_slip_fcc_test'。
            sys.stdout.write(log_new)#sys.stdout.write(log_new) 原样打出捕获的 stderr(那条 [NEW-CS] … KEEP-as-chain)。
            changed_new, planes_after_new = res_new
            print(">>> 新实现：滑移面发生改变的 segment 数 = %d  %s"
                  % (len(changed_new), changed_new if changed_new else "(无)"))#changed_new, planes_after_new = res_new:这里直接把 res_new 元组解包成两个变量(上一段 [1] 是用 res_old[0] 取第一个)。changed_new = 翻面段下标列表,planes_after_new = 交滑移后每段的面数据。解包是因为下面要用到 planes_after_new。
            if changed_new:
                cs_hat = unit(csp_new)#cs_hat = unit(csp_new):把交滑移目标面法向 (1 -1 1) 归一化,打印出来作为参照标准——理论上翻面后每段应该变成这个方向 [0.577, -0.577, 0.577]。
                print("    交滑移目标面 (1 -1 1)/√3 = %s" % np.round(cs_hat, 3))
                for i in changed_new:
                    print("      seg %d 新滑移面法向 = %s" % (i, np.round(unit(planes_after_new[i]), 3)))#for i in changed_new: 遍历每个翻面段,打印它交滑移后的实际新法向 unit(planes_after_new[i])。
            print()
        else:
            changed_new = None
            print(">>> 绑定里没有 make_cross_slip_fcc_test。")
            print(">>> 请先重新编译并安装 pyexadis（见文件顶部用法说明），再运行本测试。\n")#changed_new = None 标记"未测"(和"测了但 0 段"的空列表区分),并提示用户去重新编译安装 pyexadis——因为新实现是新加的,大概率需要重新 build 才能用。

        # ============ wansheng 方案（被测，应修复：施力后 5 段交滑移）============
        print("=" * 72)
        print("[3] wansheng 方案 CrossSlipFCCWansheng —— 期望：螺型 run(第0..4段)交滑移，5 段翻面")
        print("=" * 72)
        if hasattr(pyexadis, 'make_cross_slip_fcc_wansheng'):
            cs_p_ws = make_deterministic_params(pyexadis.CrossSlipFCCWansheng_Params())
            cs_p_ws.minChainSegments = 4         # wansheng 用 minChainSegments(默认4)，无 minRunLength 字段
            ok_ws, res_ws, csp_ws, log_ws = run_handler_with_force(
                'make_cross_slip_fcc_wansheng', params, force, cs_p_ws, MAG)
            sys.stdout.write(log_ws)
            changed_ws, planes_after_ws = res_ws
            print(">>> wansheng：滑移面发生改变的 segment 数 = %d  %s"
                  % (len(changed_ws), changed_ws if changed_ws else "(无)"))
            if changed_ws:
                cs_hat = unit(csp_ws)
                print("    交滑移目标面 (1 -1 1)/√3 = %s" % np.round(cs_hat, 3))
                for i in changed_ws:
                    print("      seg %d 新滑移面法向 = %s" % (i, np.round(unit(planes_after_ws[i]), 3)))
            print()
        else:
            changed_ws = None
            log_ws = ''
            print(">>> 绑定里没有 make_cross_slip_fcc_wansheng。")
            print(">>> 请先接入 cross_slip_fcc_wansheng.h 并重新编译安装 pyexadis（见文件顶部用法说明），再运行本测试。\n")

        # ============ 判定 ============
        print("=" * 72)
        print("结论")
        print("=" * 72)
        if changed_old is not None:
            tag = "✓ 复现 bug（施了超临界力也没交滑移：链被整条丢弃）" if len(changed_old) == 0 \
                  else "✗ 未复现（旧实现竟发生了交滑移，构型/参数与预期不符）"
            print("  旧实现:   %d 段交滑移   %s" % (len(changed_old), tag))#len==0(0 段翻面)→ ✓ 复现 bug。 否则(竟然翻了面)→ ✗ 未复现

        if changed_ws is None:
            print("  wansheng: 未测（pyexadis 未接入 wansheng 绑定）")
        elif len(changed_ws) >= 1:
            print("  wansheng: %d 段交滑移   ✓ 螺型 run 在力驱动下正确交滑移（滑移面翻到交滑移面）"
                  % len(changed_ws))
        else:
            print("  wansheng: %d 段交滑移   ✗ 仍未触发（力/参数未达阈值或链未建出）" % len(changed_ws))#完整的三态判断:None→未测;>=1→✓ 正确交滑移(期望结果);==0→✗ 仍未触发。和旧实现相反——对被测实现,翻面才是"对"。

        if changed_new is None:
            print("  新实现:   未测（pyexadis 未重新编译）")
        elif len(changed_new) >= 1:
            print("  新实现:   %d 段交滑移   ✓ 螺型 run 在力驱动下正确交滑移" % len(changed_new))
        else:
            print("  新实现:   %d 段交滑移   ✗ 仍未触发" % len(changed_new))##完整的三态判断:None→未测;>=1→✓ 正确交滑移(期望结果);==0→✗ 仍未触发。和旧实现相反——对被测实现,翻面才是"对"。

        # —— 总判定：以 wansheng 为本测试主对象 ——
        if changed_ws is None:
            print("\n  => wansheng 未测，无法判定；请重新编译并接入绑定后再跑。")#None → 没测,无法判定,提示去编译接入。
        elif len(changed_ws) >= 1:
            print("\n  => PASS：bug 已修复。")#>=1 → PASS,bug 已修复。后面两行是把整个测试的因果讲清楚:旧实现整条丢弃 → 没交滑移;wansheng 切出 run → 同一条力翻了面。这就是你截图里看到的结果。
            print("     旧实现把弯臂整条丢弃，超临界力也驱动不了交滑移；")
            print("     wansheng 切出螺型 run，同一条力就把它真正翻了面——力把 bug 测了出来。")
        else:
            print("\n  => FAIL：wansheng 未触发交滑移（bug 未修复，或力/参数未达阈值）。")#==0 → FAIL,wansheng 没触发。这里嵌了个诊断兜底:if not log_ws.strip()——如果连 [WS-CS] 调试行都没捕获到(log_ws 去掉空白后是空),就提示可能是 DEBUG 块被注释了、或根本没走到 build_screw_chains(让你去查 crystal/use_glide_planes 设置)。strip() 去掉首尾空白,not "" 为真表示日志为空。这也是为什么上一段 [3] 的 else 里要先 log_ws = ''——防止这里读到未定义变量崩掉。
            if not log_ws.strip():
                print("     （未捕获到 [WS-CS] 调试行：可能 DEBUG 块已删除，"
                      "或未走到 build_screw_chains——请检查 crystal/use_glide_planes 设置）")
    finally:
        pyexadis.finalize()


if __name__ == "__main__":
    main()
