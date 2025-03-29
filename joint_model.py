import torch
import torch.nn as nn
from encoder import GraphEncoder
from llm_model import LLMIntegrator

class JointLinkPredictionModel(nn.Module):
    def __init__(self, num_entities, num_relations, graph_embedding_dim=128, llm_embedding_dim=256):
        super(JointLinkPredictionModel, self).__init__()
        self.graph_encoder = GraphEncoder(num_entities, num_relations, graph_embedding_dim)
        self.llm_integrator = LLMIntegrator(output_dim=llm_embedding_dim)
        # Fusion layer expects a vector of size 2 (graph score and cosine similarity)
        self.fusion_linear = nn.Linear(2, 1)

    def forward(self, head_indices, relation_indices, tail_indices, head_texts, tail_texts):
        # Obtain graph-based score (a scalar per triple)
        graph_score = self.graph_encoder(head_indices, relation_indices, tail_indices)
        
        # Get LLM-based embeddings for head and tail entities
        head_llm = self.llm_integrator(head_texts)
        tail_llm = self.llm_integrator(tail_texts)
        
        # Compute cosine similarity between LLM embeddings (a scalar per triple)
        cosine_similarity = nn.functional.cosine_similarity(head_llm, tail_llm, dim=1)
        
        # Concatenate the two scalar values into a feature vector of shape [batch_size, 2]
        combined_feature = torch.cat([graph_score.unsqueeze(1), cosine_similarity.unsqueeze(1)], dim=1)
        final_score = self.fusion_linear(combined_feature).squeeze(1)
        return final_score