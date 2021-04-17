_base_ = [
    '../_base_/models/resnet50_gd_line.py',
    '../_base_/datasets/gd_line_bs32.py',
    '../_base_/schedules/gd_line_bs256_coslr.py',
    '../_base_/default_runtime.py'
]
