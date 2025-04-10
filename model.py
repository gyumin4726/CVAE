import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td    # torch distribution 임포트

import lightning as L


#############################
class CVAE(L.LightningModule):
    def __init__(self, n_dim=2, lr=1e-3):
        super().__init__()
        self.n_dim = n_dim
        self.cond_dim = 10
        self.lr = lr

        # TODO
        # 인코더
        self.encoder_mlp = nn.Sequential(
            nn.Linear(784 + self.cond_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.fc_mean = nn.Linear(128, self.n_dim)
        self.fc_logvar = nn.Linear(128, self.n_dim)

        #디코더
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.n_dim + self.cond_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    
    def encoder(self, x, c):
        x_flattened  = x.view(x.shape[0], -1)   # batch_size, 784        
        c = F.one_hot(c, num_classes=10)        # batch_size, 10
        
        # TODO
        # return mean, log_var
        xc = torch.cat([x_flattened, c], dim=1)
        h = self.encoder_mlp(xc)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

    def decoder(self, z, c):
        c = F.one_hot(c, num_classes=10)        # batch_size, 10
        zc = torch.cat([z,c], dim=1)
        
        # TODO
        # return x_pred
        x_pred = self.decoder_mlp(zc)
        return x_pred
    

    def training_step(self, batch, batch_idx):
        x, c = batch
        
        x = x.view(x.size(0), -1)
        x_pred = self(x,c)      # forward() 직접 호출

        loss = F.mse_loss(x_pred, x)
        self.log("train loss", loss, prog_bar=True)        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x, c):
        # TODO        
        # return x_pred
        mean, logvar = self.encoder(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        x_pred = self.decoder(z, c)
        return x_pred
