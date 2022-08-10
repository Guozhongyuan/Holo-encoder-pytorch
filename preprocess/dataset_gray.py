
'''
    http://www.seeprettyface.com/mydataset_page3.html#anime

    [512,512,3] -> [256,256]
    
'''

import cv2
from pathlib import Path
from tqdm import tqdm

data_path = Path('C:\\Users\\GUO\\Downloads\\anime_face')
save_path = Path('C:\\Users\\GUO\\Downloads\\anime_face_gray')

for i in tqdm(range(140000)):
    name = str(i).zfill(7) + '.jpg'
    img = cv2.imread(str(data_path.joinpath(name)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (256, 256))
    cv2.imwrite(str(save_path.joinpath(name)), img)