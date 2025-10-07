import torch as t
import numpy as np
from sklearn.linear_model import LogisticRegression


def learn_truth_directions(acts_centered, labels, polarities):
    # Check if all polarities are zero (handling both int and float) -> if yes learn only t_g
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]).cuda(), atol=1e-8)
    # Make the sure the labels only have the values -1.0 and 1.0
    labels_copy = labels.clone()
    labels_copy = t.where(labels_copy == 0.0, t.tensor(-1.0).cuda(), labels_copy)
    
    if all_polarities_zero:
        X = labels_copy.reshape(-1, 1)
    else:
        X = t.column_stack([labels_copy, labels_copy * polarities])

    # Compute the analytical OLS solution
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts_centered

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]

    return t_g, t_p

def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.cpu().numpy(), polarities_copy.cpu().numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc

class TTPD():
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    def from_data(acts_centered, acts, labels, polarities, device):
        acts_centered = acts_centered.to(device)
        acts = acts.to(device)
        labels = labels.to(device)
        polarities = polarities.to(device)
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.cpu().numpy()
        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts_2d, labels.cpu().numpy())
        return probe
    
    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_2d))
    
    def _project_acts(self, acts):
        proj_t_g = acts.cpu().numpy() @ self.t_g
        proj_p = acts.cpu().numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d


class Probe(t.nn.Module):
    @staticmethod
    def load_probe(path="probe_checkpoint.pth", device='cpu'):
        # Load the checkpoint
        checkpoint = t.load(path, map_location=device)
        
        # Recreate the model with the saved input dimension
        d_in = checkpoint['model_config']  # Input dimension
        probe = LRProbe(d_in).to(device)
        
        # Load the model weights
        probe.load_state_dict(checkpoint['model_state_dict'])
        
        return probe

class LRProbe(Probe):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        return self(x).round()
    
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()
        
        return probe

    def __str__():
        return "LRProbe"

    @property
    def direction(self):
        return self.net[0].weight.data[0]


# class MMProbe(Probe):
#     def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
#         super().__init__()
#         self.direction = t.nn.Parameter(direction, requires_grad=False)
#         if inv is None:
#             self.inv = t.nn.Parameter(t.linalg.pinv(covariance, hermitian=True, atol=atol), requires_grad=False)
#         else:
#             self.inv = t.nn.Parameter(inv, requires_grad=False)

#     def forward(self, x, iid=False):
#         if iid:
#             return t.nn.Sigmoid()(x @ self.inv @ self.direction)
#         else:
#             return t.nn.Sigmoid()(x @ self.direction)

#     def pred(self, x, iid=False):
#         return self(x, iid=iid).round()

#     def from_data(acts, labels, atol=1e-3, device='cpu'):
#         acts, labels
#         pos_acts, neg_acts = acts[labels==1], acts[labels==0]
#         pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
#         direction = pos_mean - neg_mean

#         centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
#         covariance = centered_data.t() @ centered_data / acts.shape[0]
        
#         probe = MMProbe(direction, covariance=covariance).to(device)

#         return probe
    
#     def __str__():
#         return "MMProbe"

class MMProbe(t.nn.Module):
    def __init__(self, direction, LR):
        super().__init__()
        self.direction = direction
        self.LR = LR

    def forward(self, acts, iid=False):
        proj = acts.cuda() @ self.direction.cuda()
        return t.tensor(self.LR.predict(proj[:, None].cpu().numpy()))

    def pred(self, x, iid=False):
        return self(x).round().cuda()

    def from_data(acts, labels, device='cpu'):
        acts, labels
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        # project activations onto direction
        proj = acts @ direction
        # fit bias
        LR = LogisticRegression(penalty=None, fit_intercept=True)
        LR.fit(proj[:, None].cpu().numpy(), labels.cpu().numpy())
        
        probe = MMProbe(direction, LR).to(device)

        return probe
    
    def from_probe(acts, labels, direction, device='cpu'):
        proj = acts @ direction

        LR = LogisticRegression(penalty=None, fit_intercept=True)
        
        LR.fit(proj[:, None].cpu().numpy(), labels.cpu().numpy())
        
        probe = MMProbe(direction, LR).to(device)

        return probe


def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)


class CCSProbe(Probe):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )
    
    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)
    
    def pred(self, acts, iid=None):
        return self(acts).round()
    
    def from_data(acts, neg_acts, labels=None, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None: # flip direction if needed
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1
        
        return probe
    
    def __str__():
        return "CCSProbe"

    @property
    def direction(self):
        return self.net[0].weight.data[0]