python train_supervised_full_pipeline.py \
    --depth dataset/undistorted/depth.npy \
    --normal dataset/undistorted/normal.npy \
    -s dataset/undistorted \
    -r "supervised_sdf" \
    --strategy "const_0" \
    --gs_output_dir "output/vanilla_gs/undistorted" 