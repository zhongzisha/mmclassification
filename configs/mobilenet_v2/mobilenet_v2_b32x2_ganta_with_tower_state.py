_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/datasets/ganta_with_tower_state_bs32_pil_resize.py',
    '../_base_/schedules/ganta_with_tower_state_bs64_epochstep.py',
    '../_base_/default_runtime.py'
]
checkpoint_config = dict(interval=100)