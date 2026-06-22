#include "system.h"
#include "cross_slip_fcc_thermal.h"   // 新版构建时换成 cross_slip_fcc_test.h
using namespace ExaDiS;

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        // 1) 大盒子，保证我们这条 ~900 b 的臂不会被 PBC 折叠
        double L = 1.0e5;                                  // b 单位
        Cell cell(Vec3(0,0,0), Vec3(L,L,L));               // ←(A) 按你的 Cell 构造器对齐

        // 2) 网络：照上表加节点 + 顺连成段
        SerialDisNet* net = new SerialDisNet(cell);
        Vec3 b  = Vec3(0.5, 0.0, -0.5);                    // 1/2[1 0 -1]
        Vec3 pl = Vec3(1.0, 1.0, 1.0).normalized();        // (111)
        std::vector<Vec3> P = {
            {  0.000,   0.000,    0.000},
            { 70.711,   0.000,  -70.711},
            {141.421,   0.000, -141.421},
            {212.132,   0.000, -212.132},
            {282.843,   0.000, -282.843},
            {353.553,   0.000, -353.553},
            {311.018, 230.179, -541.196},
            {268.482, 460.357, -728.839},
        };
        std::vector<int> id(P.size());
        for (size_t i = 0; i < P.size(); i++)
            id[i] = net->add_node(P[i], UNCONSTRAINED);    // ←(B) add_node 签名按你 fork
        for (size_t i = 0; i + 1 < P.size(); i++)
            net->add_seg(id[i], id[i+1], b, pl);           // ←(B) add_seg(n1,n2,burg,plane)
        net->generate_connectivity();                      // 建 conn[]

        // 3) 最小 System：FCC 晶体、R = 单位阵 → Rinv = 单位阵
        Crystal crystal(FCC_CRYSTAL);                      // ←(C) 默认取向；确认 Rinv 是单位阵
        System* system = make_system(net, crystal, Params());  // ←(C) 按你 fork 的建系接口
        system->params.burgmag = 2.49e-10;                 // Ni；本步不影响建链，仅占位

        // 4) 建链并报数
        CrossSlipFCCThermal::Params csp;
        csp.screwAngleTolerance = 15.0;
        csp.minRunLength        = 0.0;                      // 旧版没这个字段，删掉这行即可
        CrossSlipFCCThermal cs(system, nullptr, csp);
        std::size_t n = cs.test_num_chains(system, net);
        printf("=== build_screw_chains returned %zu chain(s) ===\n", n);
    }
    Kokkos::finalize();
    return 0;
}