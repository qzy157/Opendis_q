import os
import glob
import sys

sys.path.append('/data/home/dg000246d/Opendis_q/core/exadis/python')
import pyexadis
from pyexadis_utils import read_paradis, write_vtk

# ========== 在这里修改路径和范围 ==========
input_dir  = '/data/home/dg000246d/Opendis_q/a_test_cross_slip/wansheng_vis/output_cross_slip_wansheng'
output_dir = '/data/home/dg000246d/Opendis_q/a_test_cross_slip/wansheng_vis/vtk'

# ---- 转换范围设置（直接填文件名，留空则全部转换） ----
start_file = 'config.0.data'   # 起始文件名（例如 'config.0.data'），设为 None 则从第一个开始
end_file   = 'config.100.data'  # 结束文件名（例如 'config.50.data'），设为 None 则到最后一个
step       = 1                 # 步长，默认 1
# ======================================================

def natural_sort_key(s):
    """将文件名中的数字按数值排序（如 config.10.data 排在 config.2.data 之后）"""
    import re
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]

def main():
    pyexadis.initialize()
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 .data 文件并按数字排序
    data_files = glob.glob(os.path.join(input_dir, '*.data'))
    if not data_files:
        print('未找到任何 .data 文件')
        return
    data_files.sort(key=natural_sort_key)
    total = len(data_files)

    # 确定起始索引
    if start_file is not None and start_file != '':
        if start_file not in [os.path.basename(f) for f in data_files]:
            print(f'错误：起始文件 "{start_file}" 不存在于目录中')
            return
        start_idx = [os.path.basename(f) for f in data_files].index(start_file)
    else:
        start_idx = 0

    # 确定结束索引
    if end_file is not None and end_file != '':
        if end_file not in [os.path.basename(f) for f in data_files]:
            print(f'错误：结束文件 "{end_file}" 不存在于目录中')
            return
        end_idx = [os.path.basename(f) for f in data_files].index(end_file)
    else:
        end_idx = total - 1

    if start_idx > end_idx:
        print(f'错误：起始索引 {start_idx} 大于结束索引 {end_idx}，请检查文件名顺序')
        return

    selected = data_files[start_idx:end_idx+1:step]
    if not selected:
        print('按给定范围未选中任何文件')
        return

    print(f'共 {total} 个 .data 文件，将转换第 {start_idx}~{end_idx} 个（步长 {step}），共 {len(selected)} 个文件')

    for f in selected:
        basename = os.path.splitext(os.path.basename(f))[0]
        vtk_path = os.path.join(output_dir, basename + '.vtk')
        N = read_paradis(f)
        write_vtk(N, vtk_path)
        print(f'{os.path.basename(f)} -> {vtk_path}')

    print('完成')

if __name__ == '__main__':
    main()