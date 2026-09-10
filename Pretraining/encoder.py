# encoder network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsde
from tqdm import tqdm
# 1. Sepsis encoder sub netorks
class SDEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=8):
        super().__init__()
        # cond_dim 제거, input_dim만 받음
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

class LatentSDE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.latent_dim = latent_dim
        self.f_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.g_net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Softplus()
        )

    def f_prior(self, t, y):
        return -y

    def f(self, t, y):
        y_z = y[:, :-1]
        ty = torch.cat([torch.full_like(y_z[:, :1], t), y_z], dim=-1)
        f_post = self.f_net(ty)
        f_pri = self.f_prior(t, y_z)
        g_val = self.g_net(ty) + 1e-3
        u = (f_post - f_pri) / g_val
        kl_deriv = 0.5 * (u ** 2).sum(dim=-1, keepdim=True)
        return torch.cat([f_post, kl_deriv], dim=-1)

    def g(self, t, y):
        y_z = y[:, :-1]
        ty = torch.cat([torch.full_like(y_z[:, :1], t), y_z], dim=-1)
        g_val = self.g_net(ty) + 1e-3
        return torch.cat([g_val, torch.zeros_like(y_z[:, :1])], dim=-1)

    def forward(self, y0, ts):
        batch_size = y0.shape[0]
        kl_0 = torch.zeros(batch_size, 1, device=y0.device)
        y0_aug = torch.cat([y0, kl_0], dim=-1)
        y_aug_ts = torchsde.sdeint(self, y0_aug, ts, method='euler', dt=0.05)
        z_ts = y_aug_ts[:, :, :-1]
        kl_loss = y_aug_ts[-1, :, -1].mean()
        return z_ts, kl_loss

class SDEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)

class StageClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, mu):
        return self.fc(mu)

# Sepsis encoder
class SepsisEncoder:
    def __init__(self, input_dim, latent_dim=8, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.encoder = SDEEncoder(input_dim, latent_dim=latent_dim).to(self.device)
        self.sde = LatentSDE(latent_dim=latent_dim).to(self.device)
        self.decoder = SDEDecoder(latent_dim=latent_dim, output_dim=input_dim).to(self.device)
        self.classifier = StageClassifier(latent_dim=latent_dim, num_classes=4).to(self.device)
        
        self.criterion_recon = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.sde.parameters()) + 
            list(self.decoder.parameters()) + list(self.classifier.parameters()), lr=1e-3
        )
        self.ts = torch.tensor([0.0, 1.0], device=self.device)
        self.stage_params = {} # saving stage mu & var for ML 

    def train(self, train_loader, val_loader, epochs=20):
        print("Training SDE Encoder with Validation...")
        for epoch in tqdm(range(epochs)):
            # Train
            self.encoder.train(); self.sde.train(); self.decoder.train(); self.classifier.train()
            train_loss = 0
            for batch_X, batch_s in train_loader:
                batch_X, batch_s = batch_X.to(self.device), batch_s.to(self.device)
                self.optimizer.zero_grad()
                
                mu, logvar = self.encoder(batch_X)
                std = torch.exp(0.5 * logvar)
                z0 = mu + torch.randn_like(std) * std
                z_final, kl_loss = self.sde(z0, self.ts)
                
                loss_recon = self.criterion_recon(self.decoder(z_final[-1]), batch_X)
                loss_ce = self.criterion_ce(self.classifier(mu), batch_s)
                prior_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
                
                loss = loss_recon + (0.5 * loss_ce) + (0.001 * kl_loss) + (0.001 * prior_loss)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation Phase
            print("=========validation=========")
            self.encoder.eval(); self.sde.eval(); self.decoder.eval(); self.classifier.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_s in val_loader:
                    batch_X, batch_s = batch_X.to(self.device), batch_s.to(self.device)
                    mu, logvar = self.encoder(batch_X)
                    std = torch.exp(0.5 * logvar)
                    z0 = mu + torch.randn_like(std) * std
                    z_final, kl_loss = self.sde(z0, self.ts)
                    
                    l_recon = self.criterion_recon(self.decoder(z_final[-1]), batch_X)
                    l_ce = self.criterion_ce(self.classifier(mu), batch_s)
                    p_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_X.size(0)
                    val_loss += (l_recon + 0.5 * l_ce + 0.001 * kl_loss + 0.001 * p_loss).item()
            
            print(f"Epoch {epoch+1}/{epochs} ; Train Loss: {train_loss/len(train_loader):.4f} ; Val Loss: {val_loss/len(val_loader):.4f}")

    def build_distributions(self, train_loader):
        print("Building Stage Distributions from Train Set")
        self.encoder.eval()
        mu_list, state_list = [], []
        with torch.no_grad():
            for batch_X, batch_s in train_loader:
                mu, _ = self.encoder(batch_X.to(self.device))
                mu_list.append(mu.cpu().numpy())
                state_list.append(batch_s.numpy())
                
        mu_all = np.concatenate(mu_list, axis=0)
        states_all = np.concatenate(state_list, axis=0)
        
        for st in np.unique(states_all):
            mask = (states_all == st)
            stage_mu = np.mean(mu_all[mask], axis=0)
            cov_matrix = np.cov(mu_all[mask], rowvar=False)
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-4 
            self.stage_params[st] = {'mu': stage_mu, 'cov': cov_matrix}
            print(f"Stage {st} Distribution built. (N={np.sum(mask)})")

    def evaluate_test_set(self, test_loader):
        print("Test set")
        self.encoder.eval()
        pred_ml, true_states = [], []
        
        with torch.no_grad():
            for batch_X, batch_s in tqdm(test_loader):
                mu, _ = self.encoder(batch_X.to(self.device))
                mu_np = mu.cpu().numpy()
                true_states.extend(batch_s.numpy())
                
                for cur_mu in mu_np:
                    ml_logpdfs = []
                    for st in self.stage_params.keys():
                        st_mu = self.stage_params[st]['mu']
                        st_cov = self.stage_params[st]['cov']
                        try:
                            logpdf = multivariate_normal.logpdf(cur_mu, mean=st_mu, cov=st_cov)
                        except np.linalg.LinAlgError:
                            logpdf = -np.inf
                        ml_logpdfs.append(logpdf)
                    pred_ml.append(list(self.stage_params.keys())[np.argmax(ml_logpdfs)])
        
        print("Test Set Classification Report")
        print(f"Accuracy: {accuracy_score(true_states, pred_ml):.4f}")
        print(classification_report(true_states, pred_ml))

def save_model(self, path='sde_encoder_dict.pth'):
        torch.save({
            'encoder': self.encoder.state_dict(), 
            'sde': self.sde.state_dict(),
            'decoder': self.decoder.state_dict(), 
            'classifier': self.classifier.state_dict(),
            'stage_params': self.stage_params  
        }, path)
        print(f"Model and stage_params saved to {path}")

    def load_model(self, path='sde_encoder_dict.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.sde.load_state_dict(checkpoint['sde'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        
        if 'stage_params' in checkpoint:
            self.stage_params = checkpoint['stage_params']
            print("Stage distributions (mu, cov) loaded successfully.")
        
        self.encoder.eval()
        self.sde.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.sde.parameters():
            param.requires_grad = False
            
        print(f"Model loaded and frozen for downstream tasks from {path}")