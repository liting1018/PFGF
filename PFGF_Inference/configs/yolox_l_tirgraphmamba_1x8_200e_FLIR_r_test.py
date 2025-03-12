img_scale = (640, 640)  # height, width
# dataset settings
dataset_type = 'FLIRDataset'
data_root = './data_test/FLIR/'
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
    val=dict(
        type=dataset_type,
        img_list=data_root + 'test.txt',
        ann_file=data_root + 'val/vallabelrtxt/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False
        ),
    test=dict(
        type=dataset_type,
        img_list=data_root + 'test.txt',
        ann_file=data_root + 'val/vallabelrtxt/',
        pipeline=test_pipeline,
        test_mode=True,
        filter_empty_gt=False
        ))
evaluation = dict(
    save_best='auto', metric='mAP', rule='greater')
