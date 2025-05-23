_base_ = [
    '../_base_/models/regnet/regnetx_3.2gf.py',
    '../_base_/datasets/ganta_with_tower_state_bs32.py',
    '../_base_/schedules/ganta_with_tower_state_bs64.py',
    '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'GantaWithTowerStateDataset'

img_norm_cfg = dict(
    # The mean and std are used in PyCls when training RegNets
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)

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
        data_prefix='data/ganta_with_tower_state/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='data/ganta_with_tower_state/val',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='data/ganta_with_tower_state/val',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
evaluation = dict(interval=20, metric='accuracy')
optimizer = dict(lr=0.025)
checkpoint_config = dict(interval=100)

