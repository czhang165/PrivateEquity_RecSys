import torch
import numpy as np
import matplotlib.pyplot as plt
from models import TwoTowerModel, PairwiseDeepRankingModel


def load_trained_models(two_tower_checkpoint_path, deep_ranking_checkpoint_path,
                       n_investors, n_deals, feature_dims):
    """Load trained models from checkpoints"""
    
    print(f"Loading Two-Tower model from: {two_tower_checkpoint_path}")
    two_tower_model = TwoTowerModel.load_from_checkpoint(
        checkpoint_path=two_tower_checkpoint_path,
        n_investors=n_investors,
        n_deals=n_deals,
        feature_dims=feature_dims
    )
    two_tower_model.eval()
    
    print(f"Loading Deep Ranking model from: {deep_ranking_checkpoint_path}")
    deep_ranking_model = PairwiseDeepRankingModel.load_from_checkpoint(
        checkpoint_path=deep_ranking_checkpoint_path,
        n_investors=n_investors,
        n_deals=n_deals,
        feature_dims=feature_dims
    )
    deep_ranking_model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    two_tower_model = two_tower_model.to(device)
    deep_ranking_model = deep_ranking_model.to(device)
    
    print(f"Models loaded and moved to: {device}")
    
    return two_tower_model, deep_ranking_model


def score_all_deals_for_investor(model, investor_id, all_deals, 
                                investor_features, deal_features, 
                                batch_size=256):
    """Score all deals for a given investor"""
    model.eval()
    all_scores = []
    device = next(model.parameters()).device
    
    # Get investor features once
    investor_data_single = {
        'type': investor_features.loc[investor_id, 'type'],
        'region': investor_features.loc[investor_id, 'region'],
        'risk': investor_features.loc[investor_id, 'risk'],
        'min_investment': investor_features.loc[investor_id, 'min_investment'],
        'max_investment': investor_features.loc[investor_id, 'max_investment'],
        'experience_years': investor_features.loc[investor_id, 'experience_years'],
        'portfolio_size': investor_features.loc[investor_id, 'portfolio_size']
    }
    
    # Process in batches
    n_deals = len(all_deals)
    n_batches = (n_deals + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_deals)
            batch_deals = all_deals[start_idx:end_idx]
            batch_size_actual = len(batch_deals)
            
            # Prepare investor data
            investor_data = {
                'id': torch.tensor([investor_id] * batch_size_actual).to(device),
                'type': torch.tensor([investor_data_single['type']] * batch_size_actual).to(device),
                'region': torch.tensor([investor_data_single['region']] * batch_size_actual).to(device),
                'risk': torch.tensor([investor_data_single['risk']] * batch_size_actual).to(device),
                'min_investment': torch.tensor([investor_data_single['min_investment']] * batch_size_actual, dtype=torch.float32).to(device),
                'max_investment': torch.tensor([investor_data_single['max_investment']] * batch_size_actual, dtype=torch.float32).to(device),
                'experience_years': torch.tensor([investor_data_single['experience_years']] * batch_size_actual, dtype=torch.float32).to(device),
                'portfolio_size': torch.tensor([investor_data_single['portfolio_size']] * batch_size_actual, dtype=torch.float32).to(device)
            }
            
            # Prepare deal data
            deal_data = {
                'id': torch.tensor(batch_deals).to(device),
                'sector': torch.tensor([deal_features.loc[d, 'sector'] for d in batch_deals]).to(device),
                'stage': torch.tensor([deal_features.loc[d, 'stage'] for d in batch_deals]).to(device),
                'region': torch.tensor([deal_features.loc[d, 'region'] for d in batch_deals]).to(device),
                'deal_size': torch.tensor([deal_features.loc[d, 'deal_size'] for d in batch_deals], dtype=torch.float32).to(device),
                'revenue_multiple': torch.tensor([deal_features.loc[d, 'revenue_multiple'] for d in batch_deals], dtype=torch.float32).to(device),
                'growth_rate': torch.tensor([deal_features.loc[d, 'growth_rate'] for d in batch_deals], dtype=torch.float32).to(device),
                'profitability': torch.tensor([deal_features.loc[d, 'profitability'] for d in batch_deals], dtype=torch.float32).to(device),
                'team_experience': torch.tensor([deal_features.loc[d, 'team_experience'] for d in batch_deals], dtype=torch.float32).to(device),
                'market_size': torch.tensor([deal_features.loc[d, 'market_size'] for d in batch_deals], dtype=torch.float32).to(device)
            }
            
            # Get scores
            if isinstance(model, TwoTowerModel):
                scores = model(investor_data, deal_data)
            else:  # PairwiseDeepRankingModel
                scores = model(investor_data, deal_data).squeeze()
            
            all_scores.append(scores.cpu().numpy())
    
    return np.concatenate(all_scores)


def get_evaluation_candidates(investor_id, all_deals, train_interactions):
    """Get evaluation candidates excluding training positives"""
    train_positives = set(
        train_interactions[
            train_interactions['investorId'] == investor_id
        ]['dealId'].values
    )
    return np.array([d for d in all_deals if d not in train_positives])


def evaluate_metrics_implicit(model, test_interactions, train_interactions,
                             investor_features, deal_features, all_deals, K=10):
    """Evaluate metrics for implicit feedback"""
    results = {
        f'hit@{K}': [],
        f'binary_ndcg@{K}': [],
        'mrr_first': [],
        f'recall@{K}': [],
        f'precision@{K}': []
    }
    
    # Group by investor
    for investor_id, group in test_interactions.groupby('investorId'):
        if investor_id not in investor_features.index:
            continue
            
        true_positives = set(group['dealId'].values)
        
        # Get candidates excluding training positives
        candidates = get_evaluation_candidates(investor_id, all_deals, train_interactions)
        
        if len(candidates) == 0:
            continue
        
        # Score candidates
        scores = score_all_deals_for_investor(
            model, investor_id, candidates,
            investor_features, deal_features
        )
        
        # Rank by score
        ranked_indices = np.argsort(-scores)
        ranked_deals = candidates[ranked_indices]
        top_k = ranked_deals[:K]
        
        # Compute metrics
        results[f'hit@{K}'].append(int(any(d in true_positives for d in top_k)))
        
        # Binary NDCG
        dcg = sum(1.0/np.log2(i+2) for i, d in enumerate(top_k) if d in true_positives)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(true_positives), K)))
        results[f'binary_ndcg@{K}'].append(dcg/idcg if idcg > 0 else 0)
        
        # MRR
        mrr_computed = False
        for rank, deal in enumerate(ranked_deals, 1):
            if deal in true_positives:
                results['mrr_first'].append(1.0/rank)
                mrr_computed = True
                break
        if not mrr_computed:
            results['mrr_first'].append(0.0)
        
        # Recall and Precision
        n_relevant_in_top_k = len(set(top_k) & true_positives)
        results[f'recall@{K}'].append(
            n_relevant_in_top_k / len(true_positives) if len(true_positives) > 0 else 0
        )
        results[f'precision@{K}'].append(n_relevant_in_top_k / K)
    
    return {k: np.mean(v) if v else 0.0 for k, v in results.items()}


def evaluate_two_stage_pipeline(retrieval_model, ranking_model, 
                               test_interactions, train_interactions,
                               investor_df, deal_df, all_deals,
                               retrieval_k=100, final_k=10):
    """Evaluate two-stage retrieval + ranking pipeline"""
    results = {
        f'hit@{final_k}': [],
        f'binary_ndcg@{final_k}': [],
        'mrr_first': [],
        f'recall@{final_k}': [],
        f'precision@{final_k}': []
    }
    
    for investor_id, group in test_interactions.groupby('investorId'):
        if investor_id not in investor_df.index:
            continue
            
        true_positives = set(group['dealId'].values)
        
        # Stage 1: Retrieval
        candidates = get_evaluation_candidates(investor_id, all_deals, train_interactions)
        retrieval_scores = score_all_deals_for_investor(
            retrieval_model, investor_id, candidates,
            investor_df, deal_df
        )
        
        # Get top-k from retrieval
        retrieval_indices = np.argsort(-retrieval_scores)[:retrieval_k]
        retrieved_deals = candidates[retrieval_indices]
        
        # Stage 2: Re-ranking
        if len(retrieved_deals) > 0:
            ranking_scores = score_all_deals_for_investor(
                ranking_model, investor_id, retrieved_deals,
                investor_df, deal_df
            )
            
            # Final ranking
            final_indices = np.argsort(-ranking_scores)[:final_k]
            final_deals = retrieved_deals[final_indices]
        else:
            final_deals = []
        
        # Compute metrics on final results
        top_k = final_deals[:final_k]
        
        results[f'hit@{final_k}'].append(int(any(d in true_positives for d in top_k)))
        
        # Binary NDCG
        dcg = sum(1.0/np.log2(i+2) for i, d in enumerate(top_k) if d in true_positives)
        idcg = sum(1.0/np.log2(i+2) for i in range(min(len(true_positives), final_k)))
        results[f'binary_ndcg@{final_k}'].append(dcg/idcg if idcg > 0 else 0)
        
        # MRR (on retrieved + reranked)
        mrr_computed = False
        for rank, deal in enumerate(final_deals, 1):
            if deal in true_positives:
                results['mrr_first'].append(1.0/rank)
                mrr_computed = True
                break
        if not mrr_computed:
            results['mrr_first'].append(0.0)
        
        # Recall and Precision
        n_relevant_in_top_k = len(set(top_k) & true_positives)
        results[f'recall@{final_k}'].append(
            n_relevant_in_top_k / len(true_positives) if len(true_positives) > 0 else 0
        )
        results[f'precision@{final_k}'].append(n_relevant_in_top_k / final_k if final_k > 0 else 0)
    
    return {k: np.mean(v) if v else 0.0 for k, v in results.items()}


def evaluate_hit_ratio(model, interactions, investor_features, deal_features,
                      all_deals, K=10, candidates=100):
    """Simple hit ratio evaluation"""
    model.eval()
    hits = []
    
    # Build positive deals per investor
    investor_positive_deals = {}
    for _, row in interactions.iterrows():
        inv_id = row['investorId']
        deal_id = row['dealId']
        if inv_id not in investor_positive_deals:
            investor_positive_deals[inv_id] = set()
        investor_positive_deals[inv_id].add(deal_id)
    
    for investor_id, positive_deals in investor_positive_deals.items():
        if investor_id not in investor_features.index:
            continue
            
        # Sample one positive
        true_deal = np.random.choice(list(positive_deals))
        
        # Sample negatives
        negative_pool = [d for d in all_deals if d not in positive_deals]
        if len(negative_pool) < candidates - 1:
            continue
            
        neg_deals = np.random.choice(negative_pool, size=candidates-1, replace=False)
        candidate_deals = np.concatenate(([true_deal], neg_deals))
        
        # Score
        scores = score_all_deals_for_investor(
            model, investor_id, candidate_deals,
            investor_features, deal_features
        )
        
        # Check if true deal in top-k
        top_k_indices = np.argsort(-scores)[:K]
        top_k_deals = candidate_deals[top_k_indices]
        hits.append(int(true_deal in top_k_deals))
    
    return np.mean(hits)


def plot_score_distributions(two_tower_model, deep_ranking_model,
                           sample_investors, investor_df, deal_df, all_deals):
    """Plot score distributions for analysis"""
    n_investors = len(sample_investors)
    fig, axes = plt.subplots(n_investors, 2, figsize=(12, 4*n_investors))
    
    if n_investors == 1:
        axes = axes.reshape(1, -1)
    
    for idx, investor_id in enumerate(sample_investors):
        # Two-tower scores
        tt_scores = score_all_deals_for_investor(
            two_tower_model, investor_id, all_deals, investor_df, deal_df
        )
        
        # Deep ranking scores  
        dr_scores = score_all_deals_for_investor(
            deep_ranking_model, investor_id, all_deals, investor_df, deal_df
        )
        
        # Plot Two-Tower
        axes[idx, 0].hist(tt_scores, bins=50, alpha=0.7, color='blue')
        axes[idx, 0].set_title(f'Two-Tower Scores - Investor {investor_id}')
        axes[idx, 0].set_xlabel('Score')
        axes[idx, 0].set_ylabel('Count')
        axes[idx, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Plot Deep Ranking
        axes[idx, 1].hist(dr_scores, bins=50, alpha=0.7, color='orange')
        axes[idx, 1].set_title(f'Deep Ranking Scores - Investor {investor_id}')
        axes[idx, 1].set_xlabel('Score')
        axes[idx, 1].set_ylabel('Count')
        
        # Add statistics
        axes[idx, 0].text(0.65, 0.95, f'Near zero: {np.sum(np.abs(tt_scores) < 1e-9)}\nMean: {np.mean(tt_scores):.4f}\nStd: {np.std(tt_scores):.4f}',
                         transform=axes[idx, 0].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[idx, 1].text(0.65, 0.95, f'Near zero: {np.sum(np.abs(dr_scores) < 1e-9)}\nMean: {np.mean(dr_scores):.4f}\nStd: {np.std(dr_scores):.4f}',
                         transform=axes[idx, 1].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('models/score_distributions.png')
    plt.show()
