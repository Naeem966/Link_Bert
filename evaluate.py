import sys
import os
import torch
import numpy as np

from data import load_fb15k237
from joint_model import JointLinkPredictionModel

def evaluate(model, test_triples, num_entities, device, batch_size=256, hits_k=[1, 3, 10]):
    """
    Evaluate the model using Mean Reciprocal Rank (MRR) and Hits@K.
    For each test triple, both head and tail prediction are performed.
    Lower scores indicate higher plausibility.
    """
    model.eval()
    MRR = 0.0
    hits = {k: 0 for k in hits_k}
    total = 0

    with torch.no_grad():
        for idx, triple in enumerate(test_triples):
            head, relation, tail = triple

            # --- Tail Prediction: Given head and relation, rank all candidate tails --- #
            head_tensor = torch.tensor([head], device=device, dtype=torch.long)
            relation_tensor = torch.tensor([relation], device=device, dtype=torch.long)
            correct_tail_tensor = torch.tensor([tail], device=device, dtype=torch.long)
            
            tail_ranks = []
            for start in range(0, num_entities, batch_size):
                end = min(start + batch_size, num_entities)
                candidate_tail = torch.arange(start, end, device=device, dtype=torch.long)
                head_texts = [f"Entity description {head}"] * (end - start)
                candidate_tail_texts = [f"Entity description {i}" for i in range(start, end)]
                
                scores = model(head_tensor.repeat(end - start), 
                               relation_tensor.repeat(end - start), 
                               candidate_tail, head_texts, candidate_tail_texts)
                correct_score = model(head_tensor, relation_tensor, correct_tail_tensor,
                                      [f"Entity description {head}"], [f"Entity description {tail}"])
                tail_ranks.extend((scores <= correct_score).cpu().numpy())
            
            rank = np.sum(tail_ranks) + 1
            MRR += 1.0 / rank
            for k in hits_k:
                if rank <= k:
                    hits[k] += 1

            # --- Head Prediction: Given relation and tail, rank all candidate heads --- #
            tail_tensor = torch.tensor([tail], device=device, dtype=torch.long)
            correct_head_tensor = torch.tensor([head], device=device, dtype=torch.long)
            
            head_ranks = []
            for start in range(0, num_entities, batch_size):
                end = min(start + batch_size, num_entities)
                candidate_head = torch.arange(start, end, device=device, dtype=torch.long)
                candidate_head_texts = [f"Entity description {i}" for i in range(start, end)]
                tail_texts = [f"Entity description {tail}"] * (end - start)

                scores = model(candidate_head, relation_tensor.repeat(end - start), 
                               tail_tensor.repeat(end - start), candidate_head_texts, tail_texts)
                correct_score = model(correct_head_tensor, relation_tensor, tail_tensor,
                                      [f"Entity description {head}"], [f"Entity description {tail}"])
                head_ranks.extend((scores <= correct_score).cpu().numpy())

            rank = np.sum(head_ranks) + 1
            MRR += 1.0 / rank
            for k in hits_k:
                if rank <= k:
                    hits[k] += 1

            total += 2

            # Print progress
            if idx % 100 == 0 or idx == len(test_triples) - 1:
                print(f"Progress: {idx+1}/{len(test_triples)} triples evaluated.")

    MRR /= total
    hits = {k: v / total for k, v in hits.items()}
    return MRR, hits

def main():
    print("Starting evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load dataset (we use the testing split)
    print("Loading dataset...")
    _, _, testing = load_fb15k237()
    
    # Convert test triples from strings to integer IDs using the dataset mappings.
    print("Converting test triples using entity_to_id and relation_to_id mappings...")
    test_triples = [
        (
            testing.entity_to_id[head],
            testing.relation_to_id[relation],
            testing.entity_to_id[tail]
        )
        for head, relation, tail in testing.triples
    ]
    num_entities = int(testing.num_entities)
    num_relations = int(testing.num_relations)
    print(f"Total test triples: {len(test_triples)} | Total entities: {num_entities} | Total relations: {num_relations}")

    # Initialize the Joint Model
    print("Initializing JointLinkPredictionModel...")
    model = JointLinkPredictionModel(num_entities, num_relations)
    # Load saved weights if available
    model_path = os.path.join(os.path.dirname(__file__), "..", "joint_model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        print("No trained model found. Evaluation will use random initialized parameters.")

    model.to(device)
    
    print("Evaluating model on test set...")
    MRR, hits = evaluate(model, test_triples, num_entities, device)
    
    print("\nEvaluation Results on Test Set:")
    print("MRR: {:.4f}".format(MRR))
    for k, v in hits.items():
        print("Hits@{}: {:.4f}".format(k, v))

if __name__ == "__main__":
    main()