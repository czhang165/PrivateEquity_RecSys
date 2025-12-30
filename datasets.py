import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


class InvestorDealDataset(Dataset):
    """Dataset for pointwise training with negative sampling"""
    
    def __init__(self, interactions_df, investor_features, deal_features, 
                 all_deal_ids, negative_samples=4, seed=42):
        self.investor_features = investor_features
        self.deal_features = deal_features
        self.all_deal_ids = all_deal_ids
        self.negative_samples = negative_samples
        self.seed = seed
        np.random.seed(seed)
        
        # Build positive interactions
        self.positive_pairs = []
        self.investor_positive_deals = {}
        
        for _, row in interactions_df.iterrows():
            inv_id = row['investorId']
            deal_id = row['dealId']
            
            self.positive_pairs.append((inv_id, deal_id))
            
            if inv_id not in self.investor_positive_deals:
                self.investor_positive_deals[inv_id] = set()
            self.investor_positive_deals[inv_id].add(deal_id)
    
    def __len__(self):
        return len(self.positive_pairs) * (1 + self.negative_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            # Positive sample
            investor_id, deal_id = self.positive_pairs[idx]
            label = 1.0
        else:
            # Negative sample
            pos_idx = (idx - len(self.positive_pairs)) // self.negative_samples
            investor_id, _ = self.positive_pairs[pos_idx]
            
            # Sample negative deal
            positive_deals = self.investor_positive_deals[investor_id]
            negative_candidates = [d for d in self.all_deal_ids if d not in positive_deals]
            deal_id = np.random.choice(negative_candidates)
            label = 0.0
        
        # Get features
        investor_data = {
            'id': investor_id,
            'type': self.investor_features.loc[investor_id, 'type'],
            'region': self.investor_features.loc[investor_id, 'region'],
            'risk': self.investor_features.loc[investor_id, 'risk'],
            'min_investment': self.investor_features.loc[investor_id, 'min_investment'],
            'max_investment': self.investor_features.loc[investor_id, 'max_investment'],
            'experience_years': self.investor_features.loc[investor_id, 'experience_years'],
            'portfolio_size': self.investor_features.loc[investor_id, 'portfolio_size']
        }
        
        deal_data = {
            'id': deal_id,
            'sector': self.deal_features.loc[deal_id, 'sector'],
            'stage': self.deal_features.loc[deal_id, 'stage'],
            'region': self.deal_features.loc[deal_id, 'region'],
            'deal_size': self.deal_features.loc[deal_id, 'deal_size'],
            'revenue_multiple': self.deal_features.loc[deal_id, 'revenue_multiple'],
            'growth_rate': self.deal_features.loc[deal_id, 'growth_rate'],
            'profitability': self.deal_features.loc[deal_id, 'profitability'],
            'team_experience': self.deal_features.loc[deal_id, 'team_experience'],
            'market_size': self.deal_features.loc[deal_id, 'market_size']
        }
        
        return investor_data, deal_data, label


class PairwiseRankingDataset(Dataset):
    """Dataset for pairwise ranking"""
    
    def __init__(self, interactions_df, investor_features, deal_features, 
                 n_pairs_per_positive=5, seed=42):
        self.investor_features = investor_features
        self.deal_features = deal_features
        self.pairs = []
        np.random.seed(seed)
        
        # Build investor -> positive deals mapping
        investor_positive_deals = {}
        for _, row in interactions_df.iterrows():
            inv_id = row['investorId']
            deal_id = row['dealId']
            if inv_id not in investor_positive_deals:
                investor_positive_deals[inv_id] = set()
            investor_positive_deals[inv_id].add(deal_id)
        
        # Get all deal IDs
        all_deals = set(range(len(deal_features)))
        
        # Generate pairs
        print("Generating pairwise training data...")
        for investor_id, positive_deals in tqdm(investor_positive_deals.items()):
            negative_deals = list(all_deals - positive_deals)
            
            for pos_deal in positive_deals:
                n_samples = min(n_pairs_per_positive, len(negative_deals))
                sampled_negatives = np.random.choice(negative_deals, size=n_samples, replace=False)
                
                for neg_deal in sampled_negatives:
                    self.pairs.append({
                        'investor_id': investor_id,
                        'better_deal_id': pos_deal,
                        'worse_deal_id': neg_deal
                    })
        
        print(f"Created {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        investor_id = pair['investor_id']
        better_deal_id = pair['better_deal_id']
        worse_deal_id = pair['worse_deal_id']
        
        # Get features (same structure as before)
        investor_data = {
            'id': investor_id,
            'type': self.investor_features.loc[investor_id, 'type'],
            'region': self.investor_features.loc[investor_id, 'region'],
            'risk': self.investor_features.loc[investor_id, 'risk'],
            'min_investment': self.investor_features.loc[investor_id, 'min_investment'],
            'max_investment': self.investor_features.loc[investor_id, 'max_investment'],
            'experience_years': self.investor_features.loc[investor_id, 'experience_years'],
            'portfolio_size': self.investor_features.loc[investor_id, 'portfolio_size']
        }
        
        better_deal_data = {
            'id': better_deal_id,
            'sector': self.deal_features.loc[better_deal_id, 'sector'],
            'stage': self.deal_features.loc[better_deal_id, 'stage'],
            'region': self.deal_features.loc[better_deal_id, 'region'],
            'deal_size': self.deal_features.loc[better_deal_id, 'deal_size'],
            'revenue_multiple': self.deal_features.loc[better_deal_id, 'revenue_multiple'],
            'growth_rate': self.deal_features.loc[better_deal_id, 'growth_rate'],
            'profitability': self.deal_features.loc[better_deal_id, 'profitability'],
            'team_experience': self.deal_features.loc[better_deal_id, 'team_experience'],
            'market_size': self.deal_features.loc[better_deal_id, 'market_size']
        }
        
        worse_deal_data = {
            'id': worse_deal_id,
            'sector': self.deal_features.loc[worse_deal_id, 'sector'],
            'stage': self.deal_features.loc[worse_deal_id, 'stage'],
            'region': self.deal_features.loc[worse_deal_id, 'region'],
            'deal_size': self.deal_features.loc[worse_deal_id, 'deal_size'],
            'revenue_multiple': self.deal_features.loc[worse_deal_id, 'revenue_multiple'],
            'growth_rate': self.deal_features.loc[worse_deal_id, 'growth_rate'],
            'profitability': self.deal_features.loc[worse_deal_id, 'profitability'],
            'team_experience': self.deal_features.loc[worse_deal_id, 'team_experience'],
            'market_size': self.deal_features.loc[worse_deal_id, 'market_size']
        }
        
        return investor_data, better_deal_data, worse_deal_data


def collate_fn(batch):
    """Custom collate function for pointwise data"""
    investor_data = {}
    deal_data = {}
    labels = []
    
    for inv_data, d_data, label in batch:
        for key, value in inv_data.items():
            if key not in investor_data:
                investor_data[key] = []
            investor_data[key].append(value)
        
        for key, value in d_data.items():
            if key not in deal_data:
                deal_data[key] = []
            deal_data[key].append(value)
        
        labels.append(label)
    
    # Convert to tensors
    for key in investor_data:
        investor_data[key] = torch.tensor(investor_data[key])
    
    for key in deal_data:
        deal_data[key] = torch.tensor(deal_data[key])
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return investor_data, deal_data, labels


def pairwise_collate_fn(batch):
    """Custom collate function for pairwise data"""
    investor_data = {}
    better_deal_data = {}
    worse_deal_data = {}
    
    for inv_data, better_data, worse_data in batch:
        for key, value in inv_data.items():
            if key not in investor_data:
                investor_data[key] = []
            investor_data[key].append(value)
        
        for key, value in better_data.items():
            if key not in better_deal_data:
                better_deal_data[key] = []
            better_deal_data[key].append(value)
        
        for key, value in worse_data.items():
            if key not in worse_deal_data:
                worse_deal_data[key] = []
            worse_deal_data[key].append(value)
    
    # Convert to tensors
    for key in investor_data:
        investor_data[key] = torch.tensor(investor_data[key])
    
    for key in better_deal_data:
        better_deal_data[key] = torch.tensor(better_deal_data[key])
    
    for key in worse_deal_data:
        worse_deal_data[key] = torch.tensor(worse_deal_data[key])
    
    return investor_data, better_deal_data, worse_deal_data
