#%%
from ocr_table import *
from tqdm import tqdm

sys.stdout = Logger('./log.log', sys.stdout)
sys.stderr = Logger('./log.log', sys.stderr)


#%%
ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
table_engine = PPStructure(use_gpu=True, 
                            show_log=False,)

header_engine = None

save_dir = "./gzyy_table_v2"
# save_dir = "./gzyy_table"
os.makedirs(save_dir, exist_ok=True)


#%%
root_dir = "/home/data2/public/gzyy/images"
person_dir = glob.glob(os.path.join(root_dir, "*"))

error_list = []

#%%
for person_path in person_dir[:50]:
    person_name = person_path.split('/')[-1]
    save_folder = os.path.join(save_dir, person_name)
    os.makedirs(save_folder, exist_ok=True)

    extract_person_v2(person_path, ocr, table_engine, header_engine, save_folder=save_folder)



# import concurrent.futures

# def process_person(person_path, ocr, table_engine, header_engine, save_folder):
#     person_name = person_path.split('/')[-1]
#     os.makedirs(save_folder, exist_ok=True)
#     extract_person_v2(person_path, ocr, table_engine, header_engine, save_folder=save_folder)

# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = []
#     for person_path in person_dir[3:8]:
#         person_name = person_path.split('/')[-1]
#         save_folder = os.path.join(save_dir, person_name)
#         future = executor.submit(process_person, person_path, ocr, table_engine, header_engine, save_folder)
#         futures.append(future)
    
#     for future in concurrent.futures.as_completed(futures):
#         try:
#             result = future.result()
#         except Exception as e:
#             error_list.append(str(e))
