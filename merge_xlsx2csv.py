'''
Author: skyous 1019364238@qq.com
Date: 2024-11-23 16:47:39
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-11-23 16:50:10
FilePath: /ocr_table/merge_xlsx2csv.py
Description: 合并每一个case的多个xlsx文件为一个csv文件

Copyright (c) 2024 by 1019364238@qq.com, All Rights Reserved. 
'''


import os
import glob
import argparse
import pandas as pd


def delete_empty_rows(df):
    """删除空行并平移空列右侧数据"""
    empty_columns = df.columns[df.isna().all()]
    if len(empty_columns) > 0:
        for column in reversed(empty_columns):
            if column not in ['姓名', '患者编号', '性别', '采集日期', '出生日期', '年龄']:
                empty_column_index = df.columns.get_loc(column)
                df.iloc[:, empty_column_index:-1] = df.iloc[:, empty_column_index + 1:]
                df = df.drop(df.columns[-1], axis=1)
    return df


def prepossessing(df):
    """预处理数据表"""
    columns_list = df.columns.tolist()
    if '年龄' not in columns_list:
        df['年龄'] = None
    df[['姓名', '患者编号', '性别', '采集日期', '出生日期', '年龄']] = \
        df[['姓名', '患者编号', '性别', '采集日期', '出生日期', '年龄']].ffill()
    df = delete_empty_rows(df)
    return df


def process_person_folder(person_path, save_dir):
    """处理单个人员文件夹"""
    person_name = os.path.basename(person_path)
    save_path = os.path.join(save_dir, person_name + '.csv')

    person_tables = glob.glob(os.path.join(person_path, "*.xlsx"))
    person_tables = sorted(person_tables, key=lambda x: int(x.split("/")[-1].split("_")[2]))

    df_list = []
    for table_path in person_tables:
        df = pd.read_excel(table_path)
        df = prepossessing(df)
        df_list.append(df)

    if df_list:
        merged_df = pd.concat(df_list, axis=0, join='outer', ignore_index=True)
        merged_df.to_csv(save_path, index=False)
    print(f"Processed and saved: {save_path}")


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Process medical tables.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing person folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged CSV files.")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing for all person folders.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    batch_processing = args.batch

    os.makedirs(output_dir, exist_ok=True)
    person_dirs = glob.glob(os.path.join(input_dir, "*"))

    if batch_processing:
        print("Batch processing enabled. Processing all person folders.")
        for person_path in person_dirs:
            process_person_folder(person_path, output_dir)
    else:
        print("Batch processing disabled. Please specify a single folder to process.")
        if person_dirs:
            process_person_folder(person_dirs[0], output_dir)


if __name__ == "__main__":
    main()
