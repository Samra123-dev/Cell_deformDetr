import os

if __name__ == "__main__":
    cmd = """
    python main.py \
      --dataset_file coco \
      --coco_path ./data/coco \
      --batch_size 1 \
      --epochs 60 \
      --lr_drop 40 \
      --num_workers 2 \
      --output_dir ./outputs_leukemia \
      --lr 1e-4 \
      --lr_backbone 1e-5 \
      --lr_linear_proj_mult 0.05 \
      --resume ./outputs_leukemia/checkpoint0039.pth
    """
    os.system(cmd)

# --resume /media/iml/cv-lab/Samra/Deformable-DETR/r50_deformable_detr_single_scale_dc5-checkpoint.pth