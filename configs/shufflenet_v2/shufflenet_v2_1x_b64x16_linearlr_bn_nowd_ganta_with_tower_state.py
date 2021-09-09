_base_ = [
    '../_base_/models/shufflenet_v2_1x.py',
    # '../_base_/datasets/imagenet_bs64_pil_resize.py',
    # '../_base_/schedules/imagenet_bs1024_linearlr_bn_nowd.py',
    '../_base_/datasets/ganta_with_tower_state_bs64_pil_resize.py',
    '../_base_/schedules/ganta_with_tower_state_bs128_linearlr_bn_nowd.py',
    '../_base_/default_runtime.py'
]

checkpoint_config = dict(interval=50)