import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LLMIntegrator(nn.Module):
    def __init__(self, transformer_model_name="bert-base-uncased", output_dim=256):
        super(LLMIntegrator, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.linear = nn.Linear(self.transformer.config.hidden_size, output_dim)

    def forward(self, texts):
        # texts: list of strings representing entity descriptions
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # Move all input tensors to the same device as the transformer model parameters
        device = next(self.transformer.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.transformer(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0]  # Use the CLS token representation
        embeddings = self.linear(cls_embeddings)
        return embeddings