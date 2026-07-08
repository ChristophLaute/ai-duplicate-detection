# Training config for VSC22 model v68 (ViT backbone) on VCDB data.
# Run from: VSC22-Descriptor-Track-1st/train/train_v68/

root_dir = "../../"

pretrained = f"{root_dir}checkpoints/vit_base_patch32_384.npz"

model_type = "sscd"
img_width  = 384
preprocess = "efficientnet"

model = dict(
    type="SimpleContrastRecognizer",
    backbone=dict(
        type="SSCDModel",
        name="vit_base_patch32_384",
        pool_param=3.0,
        pool="gem",
        use_classify=False,
        dims=(768, 512),
        add_head=True,
    ),
)

data = dict(
    train=dict(
        type="LabelVideoLmdbDataSet",
        vids_path=[
            f"{root_dir}/data/meta/train/train_ref_vids.txt",
            f"{root_dir}/data/meta/train/train_query_id.csv",
        ],
        meta_path=f"{root_dir}/data/lmdb/vsc/meta.npz",
        lmdb_path=f"{root_dir}/data/lmdb/vsc",
        lmdb_size=1e12,
        preprocess=preprocess,
        width=img_width,
        ann_path=f"{root_dir}/data/meta/train/train_matching_ground_truth.csv",
        # v68 has no arg_lmdb_path / mixup in the original config
        crop=0.8,
    ),
    train1=dict(
        type="VideoLmdbDataSet",
        vids_path=[],
        meta_path=f"{root_dir}/data/lmdb/vsc/meta.npz",
        lmdb_path=f"{root_dir}/data/lmdb/vsc",
        lmdb_size=1e12,
        preprocess=preprocess,
        width=img_width,
    ),
)
