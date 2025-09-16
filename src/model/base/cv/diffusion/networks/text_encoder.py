import torch.nn as nn
from transformers import BertTokenizer, BertModel


# @TODO: reimplement this class so that it really encode prompt to embedding
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
    
    def forward(self, text):
        # @TODO: check 
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1)