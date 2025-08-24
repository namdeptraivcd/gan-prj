import torch 


class Config:
    def __init__(self):
        self.DEIVCE = "gpu" if torch.cuda.is_available() else "cpu"
        self.NUM_MAX_EPOCHS = 10