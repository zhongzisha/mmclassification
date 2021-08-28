_base_ = [
    '../_base_/models/swin_transformer/base_224.py',
    '../_base_/datasets/ganta_with_tower_state_bs32_swin_224.py',
    '../_base_/schedules/ganta_with_tower_state_bs64_adamw_swin.py',
    '../_base_/default_runtime.py'
]
evaluation = dict(interval=20, metric='accuracy')
optimizer = dict(lr=0.025)
checkpoint_config = dict(interval=50)