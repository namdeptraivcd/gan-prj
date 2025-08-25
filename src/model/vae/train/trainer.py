from src.config.config import GloalConfig

global_config = GloalConfig()


class VAETrainer:
    def __init__(self):
        pass
    
    def train_minibatch(self):
        pass
    
    def train_epoch(self):
        pass
    
    def fit(self, num_epochs=global_config.NUM_MAX_EPOCHS):
        for epoch in range(num_epochs):
            self.train_epoch()