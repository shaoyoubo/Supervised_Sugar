# Baseline
python train_supervised_full_pipeline.py \
    --depth dataset/resized/depth.npy \
    --normal dataset/resized/normal.npy \
    -s dataset/resized \
    -r "supervised_sdf" \
    --strategy "const_0" \
    --gs_output_dir "output/vanilla_gs/resized"
# Linear Increase
python train_supervised_full_pipeline.py \
    --depth dataset/resized/depth.npy \
    --normal dataset/resized/normal.npy \
    -s dataset/resized \
    -r "supervised_sdf" \
    --strategy "linear" \ 
    --gs_output_dir "output/vanilla_gs/resized"
# Linear Decrease
python train_supervised_full_pipeline.py \
    --depth dataset/resized/depth.npy \
    --normal dataset/resized/normal.npy \
    -s dataset/resized \
    -r "supervised_sdf" \
    --strategy "linear2" \
    --gs_output_dir "output/vanilla_gs/resized"
# Linear Decrease 2
python train_supervised_full_pipeline.py \
    --depth dataset/resized/depth.npy \
    --normal dataset/resized/normal.npy \
    -s dataset/resized \
    -r "supervised_sdf" \
    --strategy "linear3" \
    --gs_output_dir "output/vanilla_gs/resized"
# All On
python train_supervised_full_pipeline.py \
    --depth dataset/resized/depth.npy \
    --normal dataset/resized/normal.npy \
    -s dataset/resized \
    -r "supervised_sdf" \
    --strategy "custom_3" \
    --gs_output_dir "output/vanilla_gs/resized"