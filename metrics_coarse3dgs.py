import argparse
import os
import json
import torch
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.lpipsPyTorch import lpips
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB
from sugar_utils.general_utils import str2bool

from rich.console import Console
CONSOLE = Console(width=120)

os.makedirs('./lpipsPyTorch/weights/', exist_ok=True)
torch.hub.set_dir('./lpipsPyTorch/weights/')

n_skip_images_for_eval_split = 8


if __name__ == "__main__":
    
    # Parser
    parser = argparse.ArgumentParser(description='Script to evaluate coarse SuGaR models.')
    
    # Custom arguments
    parser.add_argument('--strategy', type=str, default='const_1',
                        help='Strategy to use for the time-dependent regularization. '
                        'Can be "linear", "exp", "sigmoid", "const_1" or "const_0". Default is "const_1". '
                        'If "const_1", the regularization will not change over time.')

    # Config file for scenes to evaluate
    parser.add_argument('--scene_config', type=str, 
                        help='(Required) Path to the JSON file containing the scenes to evaluate. '
                        'The JSON file should be a dictionary with the following structure: '
                        '{source_images_dir_path: vanilla_gaussian_splatting_checkpoint_path}')
    
    # Coarse model parameters
    parser.add_argument('-i', '--iteration_to_load', type=int, default=7000, 
                        help='iteration to load for vanilla 3DGS.')
    parser.add_argument('-e', '--estimation_factor', type=float, default=0.2, 
                        help='Estimation factor to load for coarse model.')
    parser.add_argument('-n', '--normal_factor', type=float, default=0.2)
    parser.add_argument('-r', '--regularization_type', type=str, 
                        help='(Required) Type of regularization to evaluate for coarse SuGaR. Can be "sdf", "density", or "dn_consistency".')
    
    # Device
    parser.add_argument('--gpu', type=int, default=0, 
                        help='Index of GPU to use.')
    
    # Evaluation parameters
    parser.add_argument('--evaluate_vanilla', type=str2bool, default=True, 
                        help='If True, will evaluate vanilla 3DGS.')
    parser.add_argument('--evaluate_coarse', type=str2bool, default=True, 
                        help='If True, will evaluate coarse SuGaR models.')
    parser.add_argument('--coarse_iteration', type=int, default=15000,
                        help='Iteration of coarse SuGaR model to evaluate.')
    parser.add_argument('--use_diffuse_color_only', type=str2bool, default=False, 
                        help='If True, will use only the diffuse component in Gaussian Splatting rendering.')
    
    args = parser.parse_args()
            
    # --- Scenes dict ---
    with open(args.scene_config, 'r') as f:
        gs_checkpoints_eval = json.load(f)
    
    # --- Coarse model parameters ---
    coarse_iteration_to_load = args.iteration_to_load
    coarse_estimation_factor = args.estimation_factor
    estim_method = args.regularization_type
    coarse_normal_factor = args.normal_factor
        
    # --- Evaluation parameters ---
    evaluate_vanilla = args.evaluate_vanilla
    evaluate_coarse = args.evaluate_coarse
    coarse_iteration = args.coarse_iteration
    use_diffuse_color_only = args.use_diffuse_color_only
            
    CONSOLE.print('==================================================')
    CONSOLE.print("Starting evaluation with the following parameters:")
    CONSOLE.print(f"Coarse iteration to load: {coarse_iteration_to_load}")
    CONSOLE.print(f"Coarse estimation factor: {coarse_estimation_factor}")
    CONSOLE.print(f"Coarse normal factor: {coarse_normal_factor}")
    CONSOLE.print(f"Estimation method: {estim_method}")
    CONSOLE.print(f"GS checkpoints for evaluation: {gs_checkpoints_eval}")
    CONSOLE.print(f"Evaluate vanilla: {evaluate_vanilla}")
    CONSOLE.print(f"Evaluate coarse: {evaluate_coarse}")
    CONSOLE.print(f"Coarse iteration to evaluate: {coarse_iteration}")
    CONSOLE.print(f"Use diffuse color only: {use_diffuse_color_only}")
    CONSOLE.print('==================================================')
    
    # Set the GPU
    torch.cuda.set_device(args.gpu)
    
    # ==========================

    result_file_dir = './output/metrics/'
    os.makedirs(result_file_dir, exist_ok=True)
    results = {}
    
    for source_path in gs_checkpoints_eval.keys():
        scene_name = source_path.split('/')[-1]
        CONSOLE.print(f"\n===== Processing scene {scene_name}... =====")
        scene_results = {}
        
        # Loading vanilla 3DGS models
        gs_checkpoint_path = gs_checkpoints_eval[source_path]
        
        CONSOLE.print("Source path:", source_path)
        CONSOLE.print("Gaussian splatting checkpoint path:", gs_checkpoint_path)    
        CONSOLE.print(f"\nLoading Vanilla 3DGS model config {gs_checkpoint_path}...")
        
        nerfmodel_30k = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=gs_checkpoint_path,
            iteration_to_load=7000,
            load_gt_images=True,
            eval_split=True,
            eval_split_interval=n_skip_images_for_eval_split,
            )

        nerfmodel_7k = GaussianSplattingWrapper(
            source_path=source_path,
            output_path=gs_checkpoint_path,
            iteration_to_load=7000,
            load_gt_images=False,
            eval_split=True,
            eval_split_interval=n_skip_images_for_eval_split,
            )
        
        if use_diffuse_color_only:
            sh_deg_to_use = 0
        else:
            sh_deg_to_use = nerfmodel_30k.gaussians.active_sh_degree

        CONSOLE.print("Vanilla 3DGS Loaded.")
        CONSOLE.print("Number of test cameras:", len(nerfmodel_30k.test_cameras))
        CONSOLE.print("Using SH degree:", sh_deg_to_use)
        
        compute_lpips = True
        cam_indices = [cam_idx for cam_idx in range(len(nerfmodel_30k.test_cameras))]
        
        # Evaluating Vanilla 3DGS
        if evaluate_vanilla:
            CONSOLE.print("\n--- Starting Evaluation of Vanilla 3DGS... ---")

            gs_7k_ssims = []
            gs_7k_psnrs = []
            gs_7k_lpipss = []
            
            
            with torch.no_grad():    
                for cam_idx in cam_indices:
                    # GT image
                    gt_img = nerfmodel_30k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                    
                
                    
                    # Vanilla 3DGS image (7K)
                    gs_7k_img = nerfmodel_7k.render_image(
                        nerf_cameras=nerfmodel_30k.test_cameras,
                        camera_indices=cam_idx).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                    
                
                    
                    gs_7k_ssims.append(ssim(gs_7k_img, gt_img))
                    gs_7k_psnrs.append(psnr(gs_7k_img, gt_img))
                    gs_7k_lpipss.append(lpips(gs_7k_img, gt_img, net_type='vgg'))    
                    
            CONSOLE.print("Evaluation of Vanilla 3DGS finished.")
            scene_results['3dgs_7k'] = {}
            scene_results['3dgs_7k']['ssim'] = torch.tensor(gs_7k_ssims).mean().item()
            scene_results['3dgs_7k']['psnr'] = torch.tensor(gs_7k_psnrs).mean().item()
            scene_results['3dgs_7k']['lpips'] = torch.tensor(gs_7k_lpipss).mean().item()
        
            
            CONSOLE.print(f"\nVanilla 3DGS results (7K iterations):")
            CONSOLE.print("SSIM:", torch.tensor(gs_7k_ssims).mean())
            CONSOLE.print("PSNR:", torch.tensor(gs_7k_psnrs).mean())
            CONSOLE.print("LPIPS:", torch.tensor(gs_7k_lpipss).mean())
        
        
        # Evaluating coarse SuGaR models (before mesh extraction)
        if evaluate_coarse:
            CONSOLE.print("\n--- Starting Evaluation of Coarse SuGaR Models... ---")
            
            # Setup paths for coarse model
            coarse_estimation_factor_str = str(coarse_estimation_factor).replace('.', '')
            if(int(coarse_normal_factor)==coarse_normal_factor):
                coarse_normal_factor=int(coarse_normal_factor)
            normal_factor_str = str(coarse_normal_factor).replace('.', '')
            
            coarse_model_base_path = f'sugarcoarse_3Dgs{coarse_iteration_to_load}_{args.strategy}_{estim_method}estim{coarse_estimation_factor_str}_sdfnorm{normal_factor_str}/'
            coarse_model_base_path = os.path.join(f'./output/coarse/{scene_name}', coarse_model_base_path)
            coarse_model_path = os.path.join(coarse_model_base_path, f'{coarse_iteration}.pt')
            
            CONSOLE.print(f"Loading coarse SuGaR model: {coarse_model_path}")
            
            try:
                # Load coarse model
                checkpoint = torch.load(coarse_model_path, map_location=nerfmodel_30k.device, weights_only=False)
                coarse_sugar = SuGaR(
                    nerfmodel=nerfmodel_30k,
                    points=checkpoint['state_dict']['_points'],
                    colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
                    initialize=False,
                    sh_levels=nerfmodel_30k.gaussians.active_sh_degree+1,
                    keep_track_of_knn=False,
                    knn_to_track=0,
                    beta_mode='average',
                )
                coarse_sugar.load_state_dict(checkpoint['state_dict'])
                coarse_sugar.eval()
                
                # Evaluate coarse model
                coarse_ssims = []
                coarse_psnrs = []
                coarse_lpipss = []
                
                with torch.no_grad():
                    for cam_idx in cam_indices:
                        # GT image
                        gt_img = nerfmodel_30k.get_test_gt_image(cam_idx).permute(2, 0, 1).unsqueeze(0)
                        
                        # Coarse SuGaR image
                        coarse_img = coarse_sugar.render_image_gaussian_rasterizer(
                            nerf_cameras=nerfmodel_30k.test_cameras,
                            camera_indices=cam_idx,
                            verbose=False,
                            bg_color=None,
                            sh_deg=sh_deg_to_use,
                            compute_color_in_rasterizer=True,
                        ).clamp(min=0, max=1).permute(2, 0, 1).unsqueeze(0)
                        
                        coarse_ssims.append(ssim(coarse_img, gt_img))
                        coarse_psnrs.append(psnr(coarse_img, gt_img))
                        coarse_lpipss.append(lpips(coarse_img, gt_img, net_type='vgg'))
                
                # Store and print results
                model_name = f'coarse_{estim_method}{coarse_estimation_factor_str}_{coarse_iteration}'
                scene_results[model_name] = {}
                scene_results[model_name]['ssim'] = torch.tensor(coarse_ssims).mean().item()
                scene_results[model_name]['psnr'] = torch.tensor(coarse_psnrs).mean().item()
                scene_results[model_name]['lpips'] = torch.tensor(coarse_lpipss).mean().item()
                
                CONSOLE.print(f"\nCoarse SuGaR results ({estim_method} regularization, iteration {coarse_iteration}):")
                CONSOLE.print("SSIM:", torch.tensor(coarse_ssims).mean())
                CONSOLE.print("PSNR:", torch.tensor(coarse_psnrs).mean())
                CONSOLE.print("LPIPS:", torch.tensor(coarse_lpipss).mean())
                
            except Exception as e:
                CONSOLE.print(f"Error evaluating coarse model: {e}")
        
        # Saves results to JSON file                
        results[scene_name] = scene_results
        estim_factor_str = str(coarse_estimation_factor).replace('.', '')
        normal_factor_str = str(coarse_normal_factor).replace('.', '')
        
        # Create filename
        filename_prefix = f'results_coarse{coarse_iteration}_{args.strategy}_{estim_method}{estim_factor_str}_normal{normal_factor_str}'
        
        if use_diffuse_color_only:
            result_file_name = f'{filename_prefix}_diffuseonly.json'
        else:
            result_file_name = f'{filename_prefix}.json'
            
        result_file_name = os.path.join(result_file_dir, result_file_name)

        CONSOLE.print(f"Saving results to {result_file_name}...")
        with open(result_file_name, 'w') as f:
            json.dump(results, f, indent=4)