
import numpy as np

from mmdet.core import eval_LAMR
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class KAISTDataset(CustomDataset):
    """Modified from FLIRDataset in mmdet.

    Args:
        img_list (str): list of images used, txt file.
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
    """

    CLASSES = ("person", )

    def __init__(self,
                 img_list,
                 ann_file,
                 pipeline,
                 **kwargs):
        self.img_list = img_list

        super(KAISTDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Params:
                ann_folder: folder that contains KAIST Dataset annotation txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        with open(self.img_list, "r") as f:
            lines = f.readlines()
        ann_files = [ann_folder + line.strip("\n") for line in lines]
        data_infos = []
        if not ann_files:  # test phase
            for line in lines:
                data_info = {}
                img_name = ann_file.replace('txt', 'jpg').split("/")[-1]
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_name = ann_file.replace('txt', 'jpg').split("/")[-1]
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_occlusions = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_ignore = []

                with open(ann_file) as f:
                    s = f.readlines()
                    s = s[1:]
                    for si in s:
                        bbox_info = si.split()
                        cls_name = bbox_info[0]
                        tlx = int(bbox_info[1])
                        tly = int(bbox_info[2])
                        width = int(bbox_info[3])
                        height = int(bbox_info[4])
                        occlusion = int(bbox_info[5])
                        # This case only happens in training.
                        # So it will not affect testing.
                        if width < 1 or height < 1:
                                continue

                        ignore = False
                        if self.test_mode:
                            if cls_name != 'person':
                                cls_name = 'person'
                                ignore = True
                            # if height < 50:
                            #     ignore = True
                            # if float(bbox_info[5]) == 2:
                            #     ignore = True

                            # if ignore:
                            #     continue
                            # if not ignore:
                            #     d = height * 0.41 - width
                            #     tlx -= d/2
                            #     width += d
                        else:
                            # if height < 50:
                            #     continue
                            if cls_name in ['people', 'person?', 'person']:
                                cls_name = 'person'
                            else:
                                continue
                        
                        label = cls_map[cls_name]
                        gt_bboxes.append([tlx, tly, tlx+width, tly+height])
                        gt_occlusions.append(occlusion)
                        gt_labels.append(label)
                        gt_ignore.append(ignore)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['occlusions'] = np.array(
                        gt_occlusions, dtype=np.int64)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['gt_ignore'] = np.array(gt_ignore, dtype=np.bool8)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 4),
                                                          dtype=np.float32)
                    data_info['ann']['occlusions'] = np.array([], dtype=np.int64)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['gt_ignore'] = np.array([], dtype=np.bool8)

                if gt_bboxes_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 4), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric=['MR_all', 'MR_day', 'MR_night'],
                 logger=None,
                 iou_thr=0.5):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
        """
        metrics = metric
        allowed_metrics = ['MR_all', 'MR_day', 'MR_night']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        assert isinstance(iou_thr, float)

        MR_result = eval_LAMR(
            metrics, 
            results,
            annotations, 
            self.img_list, 
            self.img_ids, 
            iou_thr=iou_thr,
            logger=logger)
        return MR_result