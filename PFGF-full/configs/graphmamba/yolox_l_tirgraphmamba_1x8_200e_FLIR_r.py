img_scale = (640, 640)  # height, width
# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet_TIR', deepen_factor=1.0, widen_factor=1.0,
                    pearlgan_ckpt='pearlgan_ckpt/FLIR_NTIR2DC/',#TODO 需要修改pearlgan_ckpt地址
                    pearlgan_half=True,
                    freeze_pearlgan=True),
    neck=dict(
        type='YOLOXPAFPN_TIRGraphMamba',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        num_intra_mamba_blocks=2,
        cgr_cfg=dict(rd_sc=32, dila=[64, 32, 16], n_iter=3,
                     num_inter_graphmambas=2, num_intra_graphmambas=2)),
    bbox_head=dict(
        type='YOLOXHead', num_classes=3, in_channels=256, feat_channels=256),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.05, nms=dict(type='nms', class_agnostic=True, iou_threshold=0.65)))

# dataset settings
dataset_type = 'FLIRDataset'
data_root = '/datasets/FLIR/'#TODO 需要修改dataset地址

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        img_list=data_root + 'train.txt',
        ann_file=data_root + 'train/trainlabelrtxt/',
        img_prefix=data_root + 'train/trainimgr/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        img_list=data_root + 'test.txt',
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix=data_root + 'val/valimgr/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        img_list=data_root + 'test.txt',
        ann_file=data_root + 'val/vallabelrtxt/',
        img_prefix=data_root + 'val/valimgr/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False))

max_epochs = 8
num_last_epochs = 1
interval = 1

evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='mAP',
    rule='greater')

# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=5e-4,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,  # epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=interval)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
load_from = 'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'#TODO 需要修改预训练模型地址