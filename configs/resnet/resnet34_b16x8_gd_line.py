_base_ = [
    '../_base_/models/resnet34_gd_line.py',
    '../_base_/datasets/gd_line_bs16.py',
    '../_base_/schedules/gd_line_bs128.py',
    '../_base_/default_runtime.py'
]
