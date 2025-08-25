import torch 


class GloalConfig:
    DEIVCE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_MAX_EPOCHS = 10
        

'''NLP models' configs'''
class GANConfig:
    lr = 0.0002
    noise_dim = 100
        
        
class VAEConfig:
    pass
    

'''CV models' configs'''
class DiffusionConfig:
    batch_size = 4
    lr = 1e-4
    image_size = 64
    timesteps = 300

    # Dataset
    dataset_path = "data/diffusion_dataset"
    
