import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class FeatureEncoder:
    """Encode feature dimensions for models"""
    
    def __init__(self):
        self.feature_dims = {}
        
    def fit(self, investor_df, deal_df):
        self.feature_dims = {
            'n_investor_types': investor_df['type'].max() + 1,
            'n_regions': max(investor_df['region'].max(), deal_df['region'].max()) + 1,
            'n_risk_profiles': investor_df['risk'].max() + 1,
            'n_sectors': deal_df['sector'].max() + 1,
            'n_stages': deal_df['stage'].max() + 1
        }
        return self


class InvestorTower(nn.Module):
    """Investor embedding tower"""
    
    def __init__(self, n_investors, feature_dims, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        
        # ID embedding
        self.investor_embedding = nn.Embedding(n_investors, embedding_dim)
        
        # Categorical embeddings
        self.type_embedding = nn.Embedding(feature_dims['n_investor_types'], 16)
        self.region_embedding = nn.Embedding(feature_dims['n_regions'], 16)
        self.risk_embedding = nn.Embedding(feature_dims['n_risk_profiles'], 8)
        
        # Calculate total input dimension
        input_dim = embedding_dim + 16 + 16 + 8 + 4  # +4 for numerical features
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, investor_data):
        # Embeddings
        id_emb = self.investor_embedding(investor_data['id'])
        type_emb = self.type_embedding(investor_data['type'])
        region_emb = self.region_embedding(investor_data['region'])
        risk_emb = self.risk_embedding(investor_data['risk'])
        
        # Numerical features
        numerical = torch.stack([
            investor_data['min_investment'],
            investor_data['max_investment'],
            investor_data['experience_years'],
            investor_data['portfolio_size']
        ], dim=-1).float()
        
        # Concatenate all features
        x = torch.cat([id_emb, type_emb, region_emb, risk_emb, numerical], dim=-1)
        
        # Pass through MLP
        embedding = self.mlp(x)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class DealTower(nn.Module):
    """Deal embedding tower"""
    
    def __init__(self, n_deals, feature_dims, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        
        # ID embedding
        self.deal_embedding = nn.Embedding(n_deals, embedding_dim)
        
        # Categorical embeddings
        self.sector_embedding = nn.Embedding(feature_dims['n_sectors'], 16)
        self.stage_embedding = nn.Embedding(feature_dims['n_stages'], 16)
        self.region_embedding = nn.Embedding(feature_dims['n_regions'], 16)
        
        # Calculate total input dimension
        input_dim = embedding_dim + 16 + 16 + 16 + 6  # +6 for numerical features
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, deal_data):
        # Embeddings
        id_emb = self.deal_embedding(deal_data['id'])
        sector_emb = self.sector_embedding(deal_data['sector'])
        stage_emb = self.stage_embedding(deal_data['stage'])
        region_emb = self.region_embedding(deal_data['region'])
        
        # Numerical features
        numerical = torch.stack([
            deal_data['deal_size'],
            deal_data['revenue_multiple'],
            deal_data['growth_rate'],
            deal_data['profitability'],
            deal_data['team_experience'],
            deal_data['market_size']
        ], dim=-1).float()
        
        # Concatenate all features
        x = torch.cat([id_emb, sector_emb, stage_emb, region_emb, numerical], dim=-1)
        
        # Pass through MLP
        embedding = self.mlp(x)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class TwoTowerModel(pl.LightningModule):
    """Two-Tower Neural Network for retrieval"""
    
    def __init__(self, n_investors, n_deals, feature_dims, 
                 embedding_dim=64, hidden_dims=[128, 64], temperature=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.investor_tower = InvestorTower(n_investors, feature_dims, embedding_dim, hidden_dims)
        self.deal_tower = DealTower(n_deals, feature_dims, embedding_dim, hidden_dims)
        self.temperature = temperature
        
        # For tracking losses
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, investor_data, deal_data):
        investor_emb = self.investor_tower(investor_data)
        deal_emb = self.deal_tower(deal_data)
        
        # Compute similarities with temperature scaling
        similarities = torch.sum(investor_emb * deal_emb, dim=-1) / self.temperature
        
        return similarities
    
    def training_step(self, batch, batch_idx):
        investor_data, deal_data, labels = batch
        similarities = self(investor_data, deal_data)
        loss = F.binary_cross_entropy_with_logits(similarities, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        investor_data, deal_data, labels = batch
        similarities = self(investor_data, deal_data)
        loss = F.binary_cross_entropy_with_logits(similarities, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        if 'train_loss' in self.trainer.logged_metrics:
            self.train_losses.append(self.trainer.logged_metrics['train_loss'].item())
        
    def on_validation_epoch_end(self):
        if 'val_loss' in self.trainer.logged_metrics:
            self.val_losses.append(self.trainer.logged_metrics['val_loss'].item())
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class PairwiseDeepRankingModel(pl.LightningModule):
    """Deep Ranking Model with Pairwise RankNet Loss"""
    
    def __init__(self, n_investors, n_deals, feature_dims, 
                 hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        self.save_hyperparameters()
        
        # Embeddings
        self.investor_embedding = nn.Embedding(n_investors, 64)
        self.deal_embedding = nn.Embedding(n_deals, 64)
        
        # Categorical embeddings
        self.investor_type_embedding = nn.Embedding(feature_dims['n_investor_types'], 16)
        self.investor_region_embedding = nn.Embedding(feature_dims['n_regions'], 16)
        self.investor_risk_embedding = nn.Embedding(feature_dims['n_risk_profiles'], 8)
        
        self.deal_sector_embedding = nn.Embedding(feature_dims['n_sectors'], 16)
        self.deal_stage_embedding = nn.Embedding(feature_dims['n_stages'], 16)
        self.deal_region_embedding = nn.Embedding(feature_dims['n_regions'], 16)
        
        # Calculate input dimension
        input_dim = (64 + 16 + 16 + 8 + 4) + (64 + 16 + 16 + 16 + 6)
        
        # Deep MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, investor_data, deal_data):
        # Investor embeddings
        inv_id_emb = self.investor_embedding(investor_data['id'])
        inv_type_emb = self.investor_type_embedding(investor_data['type'])
        inv_region_emb = self.investor_region_embedding(investor_data['region'])
        inv_risk_emb = self.investor_risk_embedding(investor_data['risk'])
        
        inv_numerical = torch.stack([
            investor_data['min_investment'],
            investor_data['max_investment'],
            investor_data['experience_years'],
            investor_data['portfolio_size']
        ], dim=-1).float()
        
        # Deal embeddings
        deal_id_emb = self.deal_embedding(deal_data['id'])
        deal_sector_emb = self.deal_sector_embedding(deal_data['sector'])
        deal_stage_emb = self.deal_stage_embedding(deal_data['stage'])
        deal_region_emb = self.deal_region_embedding(deal_data['region'])
        
        deal_numerical = torch.stack([
            deal_data['deal_size'],
            deal_data['revenue_multiple'],
            deal_data['growth_rate'],
            deal_data['profitability'],
            deal_data['team_experience'],
            deal_data['market_size']
        ], dim=-1).float()
        
        # Concatenate all features
        investor_features = torch.cat([
            inv_id_emb, inv_type_emb, inv_region_emb, inv_risk_emb, inv_numerical
        ], dim=-1)
        
        deal_features = torch.cat([
            deal_id_emb, deal_sector_emb, deal_stage_emb, deal_region_emb, deal_numerical
        ], dim=-1)
        
        combined = torch.cat([investor_features, deal_features], dim=-1)
        score = self.mlp(combined)
        
        return score
    
    def training_step(self, batch, batch_idx):
        investor_data, better_deal_data, worse_deal_data = batch
        
        better_scores = self.forward(investor_data, better_deal_data).squeeze()
        worse_scores = self.forward(investor_data, worse_deal_data).squeeze()
        
        # RankNet loss
        loss = F.binary_cross_entropy_with_logits(
            better_scores - worse_scores,
            torch.ones_like(better_scores)
        )
        
        # Track accuracy
        correct = (better_scores > worse_scores).float().mean()
        self.log('train_accuracy', correct, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        investor_data, better_deal_data, worse_deal_data = batch
        
        better_scores = self.forward(investor_data, better_deal_data).squeeze()
        worse_scores = self.forward(investor_data, worse_deal_data).squeeze()
        
        loss = F.binary_cross_entropy_with_logits(
            better_scores - worse_scores,
            torch.ones_like(better_scores)
        )
        
        correct = (better_scores > worse_scores).float().mean()
        self.log('val_accuracy', correct, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
