import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    # Load data
    df = pd.read_csv(file_path)
    
    # Encode user and item IDs
    df['user_id'] = pd.factorize(df['user_id'])[0]
    df['item_id'] = pd.factorize(df['item_id'])[0]
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Convert to list of tuples
    train_interactions = list(train_df[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))
    test_interactions = list(test_df[['user_id', 'item_id', 'rating']].itertuples(index=False, name=None))
    
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    
    return train_interactions, test_interactions, num_users, num_items

def create_adjacency_matrix(interactions, num_users, num_items):
    edges = [(u, num_users + i) for u, i, _ in interactions]
    edges += [(num_users + i, u) for u, i, _ in interactions]  # Add reverse edges
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index