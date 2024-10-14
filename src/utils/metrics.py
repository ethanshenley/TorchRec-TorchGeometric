import numpy as np
from sklearn.metrics import ndcg_score, mean_squared_error

def calculate_ndcg(y_true, y_pred, k=10):
    return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_hit_rate(y_true, y_pred, k=10):
    top_k_items = np.argsort(y_pred)[-k:]
    return int(np.isin(np.argmax(y_true), top_k_items))

def evaluate_recommendations(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.features, batch.edge_index)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(out.cpu().numpy())
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return {
        'NDCG@10': calculate_ndcg(y_true, y_pred),
        'MSE': calculate_mse(y_true, y_pred),
        'HitRate@10': np.mean([calculate_hit_rate(y_true[i], y_pred[i]) for i in range(len(y_true))])
    }
