#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 output_with_cross_slip 目录下的 config.<i>.data 文件
批量复制到 data 目录，i 的范围可通过命令行参数调控。

用法示例：
    python copy_config.py                 # 默认复制 config.0.data ~ config.100.data
    python copy_config.py 0 200           # 复制 config.0.data ~ config.200.data
    python copy_config.py 50 150 10       # 复制 50,60,70,...,150（步长10）
    python copy_config.py --src /path/a --dst /path/b 0 100
"""

import argparse
import shutil
import sys
from pathlib import Path

# 默认路径（按需修改）
DEFAULT_SRC = "/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_test_vis/output_cross_slip_test"
DEFAULT_DST = "/data/home/dg000246d/Opendis_q/a_test_cross_slip/yes_test_vis/data"


def main():
    parser = argparse.ArgumentParser(description="批量复制 config.<i>.data 文件")
    parser.add_argument("start", nargs="?", type=int, default=0, help="起始编号（默认 0）")
    parser.add_argument("end", nargs="?", type=int, default=100, help="结束编号，含端点（默认 100）")
    parser.add_argument("step", nargs="?", type=int, default=1, help="步长（默认 1）")
    parser.add_argument("--src", default=DEFAULT_SRC, help="源目录")
    parser.add_argument("--dst", default=DEFAULT_DST, help="目标目录")
    parser.add_argument("--overwrite", action="store_true", help="目标已存在时也覆盖（默认覆盖，此选项仅为显式说明）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        sys.exit(f"[错误] 源目录不存在: {src}")

    # 目标目录不存在则自动创建
    dst.mkdir(parents=True, exist_ok=True)

    if args.step <= 0:
        sys.exit("[错误] 步长必须为正整数")

    copied, missing = 0, []
    for i in range(args.start, args.end + 1, args.step):
        fname = f"config.{i}.data"
        src_file = src / fname
        dst_file = dst / fname

        if not src_file.is_file():
            missing.append(fname)
            continue

        shutil.copy2(src_file, dst_file)  # copy2 保留修改时间等元数据
        copied += 1
        print(f"已复制: {fname}")

    print("\n===== 完成 =====")
    print(f"成功复制 {copied} 个文件到 {dst}")
    if missing:
        print(f"以下 {len(missing)} 个文件在源目录中未找到（已跳过）:")
        print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()