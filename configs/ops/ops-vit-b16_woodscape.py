# _base_ configurations
_base_ = [
    '../_base_/models/ops_vit-b16.py',
    '../_base_/datasets/woodscape.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# dataset settings
dataset_type = 'WoodscapeDataset'
data_root = '/media/annatar/NewDrive/all_random_datasets/woodscape'
crop_size = (640, 640)  # Updated crop size

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='DebugSegMapShape'),  # Remove after verification
    dict(
        type='RandomChoiceResize',
        scales=[int(640 * x * 0.1) for x in range(5, 16)],
        resize_type='ResizeShortestEdge',
        max_size=2560
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeShortestEdge', scale=crop_size, max_size=2560),
    dict(type='LoadAnnotations'),
    # dict(type='DebugSegMapShape'),  # Remove after verification
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/train2017',
            seg_map_path='annotations/train2017'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017',
            seg_map_path='annotations/val2017'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

pretrained = 'pretrains/ViT-B-16.pt'  # noqa

data_preprocessor = dict(
    mean=[122.7709, 116.7460, 104.0937],
    std=[68.5005, 66.6322, 70.3232],
    size_divisor=640,
    test_cfg=dict(size_divisor=32)
)

model = dict(
    pretrained=pretrained,
    text_encoder=dict(
        dataset_name='coco-stuff164k',
        cache_feature=True,
        cat_bg=True
    ),
    decode_head=dict(
        num_classes=10,  # Woodscape has 10 classes
        ops_cfg=dict(cfg_dao=dict(dw_kernel_size=3))
    )
)

# Training schedule for 60k
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=60000,
    val_interval=500,
    val_begin=55000
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU'
    )
)

# AdamW optimizer
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    ),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=60000,
        by_epoch=False,
    )
]
