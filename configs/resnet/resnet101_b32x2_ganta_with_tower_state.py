_base_ = [
    '../_base_/models/resnet101.py',
    '../_base_/datasets/ganta_with_tower_state_bs32.py',
    '../_base_/schedules/ganta_with_tower_state_bs64.py',
    '../_base_/default_runtime.py'
]
evaluation = dict(interval=20, metric='accuracy')
optimizer = dict(lr=0.025)
checkpoint_config = dict(interval=100)
