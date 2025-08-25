import torch 


class GloalConfig:
    DEIVCE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_MAX_EPOCHS = 10
        

class GANConfig:
    lr = 0.0002
    noise_dim = 100
        
        
class VAEConfig:
    pass
    

class DiffusionConfig:
    # Training
    batch_size = 4
    lr = 1e-4
    image_size = 64

    # Diffusion
    timesteps = 300

    # Dataset
    dataset_path = "data/diffusion_dataset"
    
