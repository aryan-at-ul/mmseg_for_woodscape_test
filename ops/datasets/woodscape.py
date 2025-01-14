from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import os
from mmengine.fileio import get_local_path

@DATASETS.register_module()
class WoodscapeDataset(BaseSegDataset):
    """
    stuffs as it is from woodscape base dataset, use as standalone
    """

    METAINFO = dict(
        classes=[ 
            'background', 'road', 'lanemarks', 'curb', 'person', 
            'rider', 'vehicles', 'bicycle', 'motorcycle', 'traffic sign'
        ],
        palette=[
            [0, 0, 0],        # background
            [255, 0, 255],    # road
            [255, 0, 0],      # lanemarks
            [0, 255, 0],      # curb
            [0, 0, 255],      # person
            [255, 255, 255],  # rider
            [255, 255, 0],    # vehicles
            [0, 255, 255],    # bicycle
            [128, 128, 255],  # motorcycle
            [0, 128, 128]     # traffic sign
        ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

    def load_data_list(self):
        """Load image and segmentation mask paths."""
        data_list = []
        img_dir = os.path.join(self.data_root, self.data_prefix['img_path'])
        ann_dir = os.path.join(self.data_root, self.data_prefix['seg_map_path'])

        for img_file in sorted(os.listdir(img_dir)):
            if not img_file.endswith(self.img_suffix):
                continue

            seg_map = img_file.replace(self.img_suffix, self.seg_map_suffix)

            if not os.path.exists(os.path.join(ann_dir, seg_map)):
                continue

            data_info = dict(
                img_path=os.path.join(self.data_prefix['img_path'], img_file),
                seg_map_path=os.path.join(self.data_prefix['seg_map_path'], seg_map),
                reduce_zero_label=self.reduce_zero_label,
                seg_fields=[]  # No semantic segmentation fields ? it was present in a few examples, might fail for some datasets
            )

            data_list.append(data_info)

        return data_list
