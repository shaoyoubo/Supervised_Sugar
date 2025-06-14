python train_full_pipeline.py \
    --depth dataset/depth/depth.npy \
    --normal dataset/normal/normal.npy \
    -s dataset/undistorted \
    -r "supervised" \
    --high_poly True \
    --export_obj True