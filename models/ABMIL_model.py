import torch
import torch.nn as nn
import torch.nn.functional as F



class ABMIL(nn.Module):
    """
    https://github.com/AMLab-Amsterdam/AttentionDeepMIL/
    """
    def __init__(self, embedding_size, hidden_dim, attention_heads=1, n_classes=2):

        super(ABMIL, self).__init__()
        self.M = embedding_size
        self.L = hidden_dim
        self.ATTENTION_BRANCHES = attention_heads

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.attention_heads==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_classes)
        )

    def forward(self, H, label):

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL

        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, A, label
