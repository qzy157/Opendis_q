import os
import re
import glob
import sys

sys.path.append('/data/home/dg000246d/Opendis_q/core/exadis/python/')
import pyexadis
from pyexadis_utils import read_paradis, write_vtk

# ========== 在这里修改路径 ==========
input_dir  = '/data/home/dg000246d/Opendis_q/fccNi/strain_rate/3000_001/4um_3000_001_1/output'
output_dir = '/data/home/dg000246d/Opendis_q/fccNi/strain_rate/3000_001/4um_3000_001_1/vtk'
# ====================================

def config_index(path):
    """从 config.123.data 中提取 123，用于数字排序"""
    m = re.search(r'config\.(\d+)\.data$', os.path.basename(path))
    return int(m.group(1)) if m else float('inf')   # 不匹配的放最后

pyexadis.initialize()
os.makedirs(output_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(input_dir, 'config.*.data')), key=config_index)
print(f'共找到 {len(files)} 个文件')

for f in files:
    basename = os.path.splitext(os.path.basename(f))[0]
    vtk_path = os.path.join(output_dir, basename + '.vtk')
    N = read_paradis(f)
    write_vtk(N, vtk_path)
    print(f'{os.path.basename(f)} -> {vtk_path}')

print('完成')