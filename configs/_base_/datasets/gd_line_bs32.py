# dataset settings
dataset_type = 'GDLINE_Dataset'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/gd_line/',
        ann_file='data/gd_line/train_list.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/gd_line/val_list.txt',
        data_prefix='data/gd_line/', pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/gd_line/val_list.txt',
        data_prefix='data/gd_line/', pipeline=test_pipeline))
