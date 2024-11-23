'''
Author: skyous 1019364238@qq.com
Date: 2024-04-12 16:49:34
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-11-23 16:25:48
FilePath: /ocr_table/mutil_ocr.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import glob
import sys
import argparse
from tqdm import tqdm
from ocr_table import *

def main(args):
    # 初始化日志记录
    sys.stdout = Logger(args.log_path, sys.stdout)
    sys.stderr = Logger(args.log_path, sys.stderr)

    # 初始化 OCR 引擎
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    table_engine = PPStructure(use_gpu=args.use_gpu, show_log=args.show_log)
    header_engine = None  # 可选：修改为其他 header 引擎

    # 确保保存目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有人的路径
    person_dir = glob.glob(os.path.join(args.input_dir, "*"))
    error_list = []

    if args.multi_thread:
        # 多线程处理
        import concurrent.futures

        def process_person(person_path, ocr, table_engine, header_engine, save_folder):
            os.makedirs(save_folder, exist_ok=True)
            extract_person_v2(person_path, ocr, table_engine, header_engine, save_folder=save_folder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for person_path in person_dir:
                person_name = os.path.basename(person_path)
                save_folder = os.path.join(args.output_dir, person_name)
                future = executor.submit(process_person, person_path, ocr, table_engine, header_engine, save_folder)
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    future.result()
                except Exception as e:
                    error_list.append((future, type(e).__name__))
    else:
        # 单线程处理
        for person_path in tqdm(person_dir, desc="Processing"):
            person_name = os.path.basename(person_path)
            save_folder = os.path.join(args.output_dir, person_name)
            os.makedirs(save_folder, exist_ok=True)
            try:
                extract_person_v2(person_path, ocr, table_engine, header_engine, save_folder=save_folder)
            except Exception as e:
                print(f"Error processing {person_path}: {e}")
                error_list.append((person_path, type(e).__name__))

    # 输出错误信息
    if error_list:
        print(f"Errors occurred in {len(error_list)} cases:")
        for path, error in error_list:
            print(f" - {path}: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学报告OCR多线程处理工具")

    # 通用参数
    parser.add_argument("--input_dir", type=str, default="/home/data2/public/gzyy/images", help="输入文件夹路径")
    parser.add_argument("--output_dir", type=str, default="/home/data1/skyous/gzyy_table_0329", help="输出文件夹路径")
    parser.add_argument("--log_path", type=str, default="./log.log", help="日志文件保存路径")

    # OCR引擎参数
    parser.add_argument("--use_gpu", type=bool, default=True, help="是否使用GPU")
    parser.add_argument("--show_log", type=bool, default=False, help="是否显示OCR引擎的日志")

    # 多线程参数
    parser.add_argument("--multi_thread", action="store_true", help="是否使用多线程处理")
    parser.add_argument("--num_workers", type=int, default=16, help="多线程最大并发数")

    args = parser.parse_args()
    main(args)

# python mutil_ocr.py --output_dir ./output/2024-11-23_table_output_all_test


