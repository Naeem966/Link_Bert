import sys
import os
import torch
import torch.optim as optim
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from joint_model import JointLinkPredictionModel
from data import load_fb15k237

def get_negative_samples(batch, num_entities):
    """
    Create negative samples by randomly replacing either the head or tail for each triple.
    """
    neg_batch = []
    for head, relation, tail in batch:
        if np.random.rand() > 0.5:
            head = np.random.randint(0, num_entities)
        else:
            tail = np.random.randint(0, num_entities)
        neg_batch.append((head, relation, tail))
    return neg_batch

def train():
    print("Starting training process...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load training data from PyKEEN
    print("Loading dataset...")
    training, _, _ = load_fb15k237()
    
    # Convert triples from strings to integer ids using the provided mappings
    print("Converting triples to integer IDs using entity_to_id and relation_to_id mappings...")
    triples = [
        (
            training.entity_to_id[head],
            training.relation_to_id[relation],
            training.entity_to_id[tail]
        )
        for head, relation, tail in training.triples
    ]
    num_entities = int(training.num_entities)
    num_relations = int(training.num_relations)
    print("Total entities:", num_entities, "Total relations:", num_relations)
    
    # Initialize Joint Model and move to device
    print("Initializing JointLinkPredictionModel...")
    model = JointLinkPredictionModel(num_entities, num_relations)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MarginRankingLoss(margin=1.0)
    
    epochs = 5
    batch_size = 64
    
    print("Starting training epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(triples)
        batch_count = 0
        print(f"Epoch {epoch+1}/{epochs} started.")
        
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            if not batch:
                continue
                
            optimizer.zero_grad()
            print(f"Processing batch {batch_count+1}...")
            
            # Prepare positive batch data
            heads, relations, tails = zip(*batch)
            # For demonstration, use placeholder texts.
            head_texts = [f"Entity description {head}" for head in heads]
            tail_texts = [f"Entity description {tail}" for tail in tails]
            
            pos_heads = torch.tensor(heads, dtype=torch.long, device=device)
            pos_relations = torch.tensor(relations, dtype=torch.long, device=device)
            pos_tails = torch.tensor(tails, dtype=torch.long, device=device)
            
            pos_scores = model(pos_heads, pos_relations, pos_tails, head_texts, tail_texts)
            
            # Generate negative samples
            neg_batch = get_negative_samples(batch, num_entities)
            n_heads, n_relations, n_tails = zip(*neg_batch)
            n_head_texts = [f"Entity description {head}" for head in n_heads]
            n_tail_texts = [f"Entity description {tail}" for tail in n_tails]
            
            neg_heads = torch.tensor(n_heads, dtype=torch.long, device=device)
            neg_relations = torch.tensor(n_relations, dtype=torch.long, device=device)
            neg_tails = torch.tensor(n_tails, dtype=torch.long, device=device)
            
            neg_scores = model(neg_heads, neg_relations, neg_tails, n_head_texts, n_tail_texts)
            
            # In margin ranking loss, the positive triple should score lower than the negative triple.
            target = torch.ones(len(pos_scores), device=device)
            loss = criterion(neg_scores, pos_scores, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
        # Save trained model weights
    model_path = os.path.join(os.path.dirname(__file__), "..", "joint_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print("Training finished.")


if __name__ == "__main__":
    train()