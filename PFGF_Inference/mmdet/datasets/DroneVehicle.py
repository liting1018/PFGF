
import os
import os.path as osp
import numpy as np

from mmdet.core import eval_map, eval_FP
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DroneVehicleDataset(CustomDataset):
    """Modified from DroneVehicleDataset in mmrotate.

    Args:
        img_list (str/list[txt]): list of images used, txt file.
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        difficulty (bool, optional): The difficulty threshold of GT.
    """

    CLASSES = ("car", "truck", "bus", "van", "freight car")
    W = 640
    H = 512

    def __init__(self,
                 img_list,
                 ann_file,
                 pipeline,
                 difficulty=100,
                 **kwargs):
        self.img_list = img_list
        self.difficulty = difficulty

        super(DroneVehicleDataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Params:
                ann_folder: folder that contains DroneVehicle annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        if type(self.img_list) == str:
            with open(self.img_list, "r") as f:
                lines = f.readlines()
        elif type(self.img_list) == list:
            lines = []
            for img_txt in self.img_list:
                with open(img_txt, "r") as f:
                    lines_ = f.readlines()
                lines += lines_
        lines.sort()

        ann_files = [ann_folder + line.strip("\n")+'.txt' for line in lines]
        data_infos = []
        if not ann_files:  # test phase
            for line in lines:
                data_info = {}
                img_id = line.strip("\n")
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []

                with open(ann_file) as f:
                    s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    poly = np.array(bbox_info[:8], dtype=np.int64).reshape(4, 2)
                    # Clip the coordinates within the image boundaries.
                    tlx = max(0, np.min(poly[:, 0]))
                    tly = max(0, np.min(poly[:, 1]))
                    brx = min(self.W-1, np.max(poly[:, 0]))
                    bry = min(self.H-1, np.max(poly[:, 1]))
                    assert len(bbox_info)==10 or len(bbox_info)==11
                    if len(bbox_info)==10:
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                    elif len(bbox_info)==11:
                        cls_name = ' '.join(bbox_info[8:10])
                        difficulty = int(bbox_info[10])
                    label = cls_map[cls_name]
                    if difficulty > self.difficulty:
                        pass
                    else:
                        gt_bboxes.append([tlx, tly, brx, bry])
                        gt_labels.append(label)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 4),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)

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
            if self.filter_empty_gt:
                if data_info['ann']['labels'].size > 0:
                    valid_inds.append(i)
            else:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric=['mAP', 'FP'],
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if type(metric) == str:
            metric = [metric]
        
        allowed_metrics = ['mAP', 'FP']
        for _metric in metric:
            if _metric not in allowed_metrics:
                raise KeyError(f'metric {_metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        for _metric in metric:
            if _metric == 'mAP':
                assert isinstance(iou_thr, float)

                # If the results have bbox logvar or distri, we split the bbox results.
                if type(results[0][0]) == tuple:
                    results_bbox = []
                    for result in results:
                        result_bbox = []
                        for i in range(len(result)):
                            result_bbox.append(result[i][0])
                        results_bbox.append(result_bbox)
                else:
                    results_bbox = results

                mean_ap, _ = eval_map(
                    results_bbox,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc,
                    use_legacy_coordinate=True)
                eval_results['mAP'] = mean_ap
            elif _metric == 'FP':
                FP_results = eval_FP(
                    results_bbox,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    nproc=nproc,
                    use_legacy_coordinate=True)
                print('Actual False Positives:', FP_results)
            else:
                raise NotImplementedError

        return eval_results
