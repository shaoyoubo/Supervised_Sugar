python metrics_coarse3dgs.py \
  --scene_config ./scene_config.json \
  --strategy "custom_0" \
  -r "supervised_sdf" \
  --coarse_iteration 15000 \
  --estimation_factor 0.2 \
  --normal_factor 0.2 \
  --gpu 0 \
  --evaluate_vanilla False \
  --evaluate_coarse True