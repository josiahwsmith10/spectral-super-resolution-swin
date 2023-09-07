from train import main as train_main


def main():
    # Default Experiment
    exp = {
        # Model Type
        "fr_module_type": "cvswinfreq",
        
        # Signal Characteristics
        "signal_dim": 64,
        "fr_size": 4096,
        "min_sep": 0.5,
        "amplitude": "uniform",
        "noise": "gaussian_blind_strict",
        "snr": -10,
        "min_snr_db": -10,
        "max_snr_db": 40,
        "step_snr_db": 10,
        "gaussian_std": 0.12,
        
        # cResFreq
        "fr_n_layers": 24,
        "fr_n_filters": 32,
        "fr_inner_dim": 256,
        "fr_upsampling": 16,
        "fr_kernel_out": 18,
        "fr_out_padding": 0,
        
        # Channel Attention
        "fr_reduction_factor": 1,
        
        # Swin
        "fr_depths": [2, 2, 2],
        "fr_num_heads": [8, 8, 8],
        "fr_window_size": 16,
        "fr_mlp_ratio": 2.0,
        "fr_dropout": 0.0,
        "normalization": "min-max",
        "fr_optional_relu": 0,
        
        # Training
        "lr_fr": 3e-3,
        "batch_size": 256,
        "n_epochs_fr": 200,
        "optim_type": "adamw",
        
        # Data
        "n_training": 500000,
        "n_validation": 5000,
        
        # Note
        "zzz_note": "cvswin 222 888 500k"
    }
    train_main(
        experiment=exp,
        #checkpoint_path="checkpoint/swin_8888_8888_more_data_larger_batchsize_v2_07-17-1907/swinfreq/last.pth"
    )


if __name__ == "__main__":
    main()
