#%%
from img2table.document import Image
from img2table.ocr import PaddleOCR

import cv2

img_path = 'debug_output/肖明_MR200719001/jc_page_6_0_table/img1.jpg'
img = cv2.imread(img_path)




#%%
ocr = PaddleOCR(
    lang="ch"
)  # need to run only once to download and load model into memory



image = Image(img_path, 
              detect_rotation=False)
# %%
extract_tables = image.extract_tables(
            ocr=ocr, min_confidence=50, borderless_tables=True
        )
# %%
print(extract_tables[0].df)