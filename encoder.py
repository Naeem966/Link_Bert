# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:41:47 2025

@author: admin
"""
import torch
import torch.nn as nn

class GraphEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(GraphEncoder, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, head_indices, relation_indices, tail_indices):
        head_embed = self.entity_embeddings(head_indices)
        relation_embed = self.relation_embeddings(relation_indices)
        tail_embed = self.entity_embeddings(tail_indices)
        # TransE-style scoring: head + relation should be close to tail
        score = head_embed + relation_embed - tail_embed
        score = torch.norm(score, p=1, dim=1)
        return score