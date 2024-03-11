import cv2
import matplotlib.pyplot as plt
import os, sys
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
import glob
import numpy as np

import shutil
from tqdm import tqdm



class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲,实时输出
        self.log.flush()

    def flush(self):
        pass


# 在原图上面进行画线
def draw_lines_on_image(image, lines):
    # Read the image
    # image = cv2.imread(image_path)

    # Draw lines on the image
    if type(lines) == list:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    else:
        x1, y1, x2, y2 = lines[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image

# 在原图上面进行表格画线
def draw_table_line(image, table_lines):
    for table in table_lines:
        for line in table:
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image    

def line_detection(image):

    '''
    水平线以及分割线检测
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=600, minLineLength=300, maxLineGap=200  # threshold=600
    )

    horizontal_lines = []
    vertical_lines = []
    split_line = []


    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):  # 水平线的斜率接近于0
            
            horizontal_lines.append(line)
            horizontal_lines = sorted(
                    horizontal_lines, key=lambda x: min(x[0][1], x[0][3])
                )

        # else:  # 垂直线的斜率很大，或者接近无穷大
        #     vertical_lines.append(line)
        #     vertical_lines = sorted(vertical_lines, key=lambda x: min(x[0][0], x[0][2]))

        if line[0][0] == 0:
            split_line = line

    return horizontal_lines, split_line

def vertical_line_detection(image):
    '''检测垂直线'''
    vertical_lines = []

    # 垂直线检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_ = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines_ = cv2.HoughLinesP(
        edges_, 1, np.pi / 180, threshold=300, minLineLength=300, maxLineGap=100  # threshold=325
    )
    for line in lines_:
        x1, y1, x2, y2 = line[0]
        # if abs(x2 - x1) > abs(y2 - y1) or x1 < 600 or x1 > 700:
        if abs(x2 - x1) > abs(y2 - y1) or x1 < 500 or x1 > 1000:
            pass
        else:  # 垂直线的斜率很大，或者接近无穷大
            vertical_lines.append(line)
            vertical_lines = sorted(vertical_lines, key=lambda x: min(x[0][0], x[0][2]))         

    return vertical_lines

def merge_vertical_lines(horizontal_lines):
    '''
    删除重复的horizontal_lines
    '''
    merged_lines = []
    for line in horizontal_lines:
        if len(merged_lines) == 0:
            merged_lines.append(line)
        else:
            prev_line = merged_lines[-1]
            if abs(line[0][1] - prev_line[0][3]) < 7:
                prev_line[0][3] = line[0][3]
            else:
                merged_lines.append(line)
    return merged_lines

def find_table_lines(img, horizontal_lines):
    '''
    保存表格头尾两根线以及表格中间的线的index
    '''

    table_index = []
    table_lines = []

    # 表头信息框线
    header_lines = []

    # 表头间距
    tableheader_spacing_list = []

    # 定义最开始的线为0
    a_y1 = 0
    for i, horizontal_line in enumerate(horizontal_lines):
        if len(table_lines) == 2 :
            break
        else:
            # debug
            # print(horizontal_line[0][1] - a_y1)
        
            tableheader_spacing = horizontal_line[0][1] - a_y1
            # 设置表头间距的阈值
            if tableheader_spacing < 75 and tableheader_spacing > 20:
                table_double_line = []
                header_double_line = []
                table_index.append(i)
                table_double_line.append(horizontal_lines[i-1][0])
                table_double_line.append(horizontal_lines[i+1][0])

                
                header_double_line.append(horizontal_lines[i-2][0])
                header_double_line.append(horizontal_lines[i-1][0])

                # table_lines.append(horizontal_lines[i])
                table_lines.append(table_double_line)
                header_lines.append(header_double_line)

                tableheader_spacing_list.append(tableheader_spacing)


            a_y1 = horizontal_line[0][1]

    return table_index, table_lines, header_lines, tableheader_spacing_list
    
def extract_table(img, table_lines):
    '''提取表格'''

    table_list = []
    for line in table_lines:
        up_line = line[0]
        down_line = line[1]
        _, y1, _, _ = up_line
        _, y2, _, _ = down_line
    # 裁剪表格
        table_img = img[y1:y2, :, :]
        table_list.append(table_img)


    return table_list

# # Split the image into two based on vertical_line
def split_img_by_vertical_line(img, vertical_line):
    '''根据垂直线进行图片切割'''
    x1, _, x2, _ = vertical_line[0][0]
    img1 = img[:, :x1, :]
    img2 = img[:, x2:, :]
    return img1, img2

# 用于检测表头文字
def ocr_for_img(img_path, ocr):
    result = ocr.ocr(img_path, cls=True)
    result_dict = {}
    for i in result[0]:
        content = i[-1][0]
        if ("患者编号" in content) or ("姓名" in content) or ("性别" in content) or ("年龄" in content) or ("出生日期" in content): 
            try:
                key, value = content.split("：")    
            except ValueError:
                key, value = content.split(":")  
            result_dict[key] = value
        if "采集时间" in content:
            data = content.split("时间：")[-1].split("：")[0][:10]
            result_dict["采集日期"] = data

    df = pd.DataFrame([result_dict])
    return df

# 合并两个表格
def merge_table(xlsx1_path, xlsx2_path, save_path):
    '''合并两个表格'''
    df1 = pd.read_excel(xlsx1_path)
    df1_columns = df1.columns.tolist()

    # Read the contents of xlsx2 with column names
    df2 = pd.read_excel(xlsx2_path, header=None)

    # 当两个表格的列数不一样时，进行处理
    # 把列数少的表格补齐表头，df2再对齐df1的表头
    if len(df2.columns) > len(df1.columns):
        for i in range(len(df2.columns) - len(df1.columns)):
            df1 = pd.concat([df1, pd.DataFrame(columns=['Unname' + str(i)])], sort=False)
        
    elif len(df2.columns) < len(df1.columns):
        for i in range(len(df1.columns) - len(df2.columns)):
            df2 = pd.concat([df2, pd.DataFrame(columns=['Unname' + str(i)])], sort=False)

        # df1 = df1.assign(Unname = 'default value')
    df2.columns = df1.columns

    # Concatenate the two dataframes vertically
    df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)

    # Write the combined dataframe to xlsx1
    df_combined.to_excel(save_path, index=False)

    return df_combined

def post_process_table(table_folder):
    '''表格后处理'''
    
    # Get a list of file paths in the folder
    file_paths = glob.glob(os.path.join(table_folder, '*.xlsx'))

    # Sort the file paths by creation time
    file_paths_sorted = sorted(file_paths, key=os.path.getctime)

    if len(file_paths_sorted) == 1:
        return pd.read_excel(file_paths_sorted[0])
    
    elif len(file_paths_sorted) == 2:
        save_path = os.path.join(table_folder, 'new.xlsx')
        df_combined = merge_table(file_paths_sorted[0], file_paths_sorted[1], save_path)
        return df_combined
    
    else:
        print("There is no xlsx files in the folder")
        # df_combined = pd.DataFrame()
        return None
    # for file_path in file_paths_sorted:
    #     os.remove(file_path)

    # return df_combined


def extract_singel_img(image_path, ocr, table_engine, header_engine, save_folder="./output/table_output"):
    '''单张图片信息提取'''
    img_name = image_path.split('/')[-1].split('.')[0]

    # 提取对于的图片名字
    img_name = image_path.split('/')[-1].split('.')[0]

    # save_folder = './output'
    os.makedirs(save_folder, exist_ok=True)


    img = cv2.imread(image_path)

    # 识别所有水平线以及分割线
    horizontal_lines, split_line = line_detection(img)
    # 合并距离短的线，避免出现一线多画的情况
    horizontal_lines = merge_vertical_lines(horizontal_lines)


    table_index, table_lines, header_lines, tableheader_spacing_list = find_table_lines(img, horizontal_lines)

    table_list = extract_table(img, table_lines)
    header_list = extract_table(img, header_lines)


    # 遍历上下报告内容
    for i, (table_img, header_img, tableheader_spacing)  in enumerate(zip(table_list, header_list, tableheader_spacing_list)):
        

        # 每份检查的文件夹名称
        name = img_name + '_' + str(i)

        # 保存检测信息
        header_result = header_engine(header_img)
        save_structure_res(header_result, save_folder, name+'_header') 

        header_jpg_path = os.path.join(save_folder, name+'_header' , '*.jpg')
        header_jpg_path = glob.glob(header_jpg_path)[0]


        head_df = ocr_for_img(header_jpg_path, ocr)
            
        
        
        vertical_line = vertical_line_detection(table_img)
        print(vertical_line)


        if vertical_line:
            img1, img2 = split_img_by_vertical_line(table_img, vertical_line)

            # 对img2进行裁剪
            img2 = img2[tableheader_spacing:, :]

            result1 = table_engine(img1)
            result2 = table_engine(img2)
            save_structure_res(result1, save_folder, name+'_table')
            save_structure_res(result2, save_folder, name+'_table')

            # 合并两个表格为一个新表格
            # print(os.path.join(save_folder, name))
            table_df = post_process_table(os.path.join(save_folder, name+'_table'))
            

        else:
            result = table_engine(table_img)
            save_structure_res(result, save_folder, name+'_table')

            table_df = post_process_table(os.path.join(save_folder, name+'_table'))
            # table_df = pd.read_excel(glob.glob(os.path.join(save_folder, name+'_table', "*.xlsx"))[0])

        

        # 合并表头与表格
        if table_df is not None:
            df = pd.concat([head_df, table_df], axis=1)
            excel_path = os.path.join(save_folder, img_name+'_'+str(i)+'.xlsx')
            df.to_excel(excel_path, index=False)


        # 删除中间信息文件夹
        shutil.rmtree(os.path.join(save_folder, name+'_table'))
        shutil.rmtree(os.path.join(save_folder, name+'_header'))


def extract_person(person_dir, ocr, table_engine, header_engine, save_folder="./output/table_output"):
    '''提取一个人的信息'''
    person_images = glob.glob(os.path.join(person_dir, "*jc*.png"))



    for image_path in tqdm(person_images):
        # print(image_path)


        extract_singel_img(image_path, ocr, table_engine, header_engine, save_folder=save_folder)   




            



if __name__ == "__main__":

    sys.stdout = Logger('./log.log', sys.stdout)
    sys.stderr = Logger('./log.log', sys.stderr)


    ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
    table_engine = PPStructure(use_gpu=True, 
                            show_log=True,
                            # det_model_dir='./model/det/ch_PP-OCRv3_det_infer',
                            # rec_model_dir='./model/rec/ch_PP-ch_PP-OCRv3_rec_infer',
                            # table_model_dir='/home/skyous/git/ocr_table/model/table/ch_ppstructure_mobile_v2.0_SLANet_infer'
                            )

    header_engine = PPStructure(use_gpu=True, 
                                show_log=True,
                    )


    root_dir = "/home/data2/public/gzyy/images"
    person_dir = glob.glob(os.path.join(root_dir, "*"))

    error_list = []


    for person_path in person_dir[10:]:
        person_name = person_path.split('/')[-1]
        save_folder = os.path.join("./output/table_output", person_name)

        print(person_name)

        try:
            extract_person(person_path, ocr, table_engine, header_engine, save_folder=save_folder)

        except Exception as e:
            print(f"No table found: {person_path}")
            error_list.append((person_path, type(e).__name__))

    # Save error_list to a txt file
    with open('error_list.txt', 'w') as file:
        for item in error_list:
            file.write(f"{item[0]} {item[1]}\n")









    # image_path = "/home/data2/public/gzyy/images/叶云龙_SMR201125118/jc_page_5.png"

    # extract_singel_img(image_path, ocr, table_engine, header_engine, save_folder="./output/table_output")


    # extract_person("/home/data2/public/gzyy/images/叶云龙_SMR201125118", ocr, table_engine, header_engine, save_folder="./output/table_output")




