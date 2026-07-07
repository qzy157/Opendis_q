import os
import glob
import sys

sys.path.append('/data/home/dg000246d/Opendis_q/core/exadis/python')
import pyexadis
from pyexadis_utils import read_paradis, write_vtk

# ========== 在这里修改路径 ==========
input_dir  = '/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_test_vis/output_cross_slip_test'   # .data 文件所在目录
output_dir = '/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_test_vis/vtk'      # .vtk 文件保存目录
# ====================================

pyexadis.initialize()
os.makedirs(output_dir, exist_ok=True)

for f in glob.glob(os.path.join(input_dir, '*.data')):
    basename = os.path.splitext(os.path.basename(f))[0]
    vtk_path = os.path.join(output_dir, basename + '.vtk')
    N = read_paradis(f)
    write_vtk(N, vtk_path)
    print(f'{os.path.basename(f)} -> {vtk_path}')

print('完成')