'''
    save dataset as a single file
'''

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import zlib


def one_process(i):
    name = str(i).zfill(7) + '.jpg'
    img = cv2.imread(str(data_path.joinpath(name)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.array(img).astype(np.uint8)
    return zlib.compress(pickle.dumps(img))


if __name__ == '__main__':
    
    data_path = Path('C:\\Users\\GUO\\Downloads\\anime_face_gray')
    save_path = Path('C:\\Users\\GUO\\Downloads')

    #--------- single kernel -----------#
    res = []
    for i in tqdm(range(120000, 140000)):
        res.append(one_process(i))
    #--------- single kernel -----------#

    filepath = save_path.joinpath('6.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(res, f)