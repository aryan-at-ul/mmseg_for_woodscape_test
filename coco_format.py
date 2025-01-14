import os
import cv2
import numpy as np



#run this for train and val, create a copy it writes on the source file itself
data_root = '/media/annatar/NewDrive/all_random_datasets/woodscape'
ann_dir = os.path.join(data_root, 'annotations/train2017')

palette = [
    [0, 0, 0],        
    [255, 0, 255],    
    [255, 0, 0],      
    [0, 255, 0],      
    [0, 0, 255],      
    [255, 255, 255], 
    [255, 255, 0],    
    [0, 255, 255],    
    [128, 128, 255],  
    [0, 128, 128]     
]

class_to_id = {tuple(color): idx for idx, color in enumerate(palette)}

for filename in os.listdir(ann_dir):
    if not filename.endswith('.png'):
        continue

    seg_map = cv2.imread(os.path.join(ann_dir, filename), cv2.IMREAD_COLOR)
    seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
    label_map = np.zeros(seg_map.shape[:2], dtype=np.int64)

    for color, idx in class_to_id.items():
        mask = np.all(seg_map == color, axis=2)
        label_map[mask] = idx


    cv2.imwrite(os.path.join(ann_dir, filename), label_map)

