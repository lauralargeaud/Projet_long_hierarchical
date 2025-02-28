import torch

class MessagePassing:

    def __init__(self, H_raw, P_raw, M_raw, La_raw, iter_count, device):
        self.H = torch.tensor(H_raw, dtype=torch.float32).to(device) # (N, N)
        self.P = torch.tensor(P_raw, dtype=torch.float32).to(device)
        self.M = torch.tensor(M_raw, dtype=torch.float32).to(device)
        self.La = torch.tensor(La_raw, dtype=torch.float32).to(device)
        self.iter_count = iter_count
        self.N = self.H.shape[1]
        self.batch_size = None

    # output: shape = (batch_size, N)
    def update(self, output):
        """One iteration of the message passing algorithm"""
        output += self.c_score(output) + self.d_score(output) + self.e_score(output) # (batch_size, N)

        # TODO hierarchical level-wise normalization
        # n = 0
        # for l in range(L, 0, -1):
        #     output[n:n+V[l]] = output[n:n+V[l]].softmax(dim=0)
        #     n += V[l]
        
            
        return output

    def process(self, output):
        """Apply the message passing algorithm"""
        self.batch_size = output.shape[0]
        for i in range (self.iter_count):
            output = self.update(output)
        return output
    
    # output: (batch_size, N)
    # result: (batch_size, N) 
    def c_score(self, output):
        """Pour chaque v_i = output[id_batch, i], le terme somme{proba(fils(i)).hC(fils(i),i)} stocké dans result[id_batch, i]"""
        # 1 − s[fils_v] + s[fils_v]·s[v]
        H_rep = self.H.T.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
        output_rep = output.unsqueeze(2).repeat(1, 1, self.N) # (batch_size, N, N)
        probas_fils = H_rep * output_rep # (batch_size, N , N) avec probas_fils[id_batch, :, id_noeud] = les probas des fils du noeud id_noeud
        probas_prod = probas_fils * output_rep # (batch_size, N, N)
        c_mat = H_rep - probas_fils + probas_prod # (batch_size, N, N)
        c_mat = c_mat * probas_fils # (batch_size, N, N)
        nb_fils = torch.sum(H_rep, dim=1) # (batch_size, N) avec nb_fils[id_batch, 0] le nombre de fils de la racine
        result = torch.sum(c_mat, dim=1) / torch.maximum(nb_fils, torch.tensor(1)) # (batch_size, N) TODO: vérifier qu'on doit faire le torch.maximum(...)
        return result

    def d_score(self, output):
        # 1−s[pv]+s[pv]·max(s[fils(pv,n)]n),
        H_rep = self.H.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N) (ici on travail avec H.T, mais H.T.T = H)
        output_rep = output.unsqueeze(2).repeat(1, 1, self.N) # (batch_size, N, N)
        probas_parents = H_rep * output_rep # (batch_size, N , N) avec probas_fils[id_batch, :, id_noeud] = les probas du parent du noeud id_noeud
        probas_parents = torch.sum(probas_parents, dim=1)

        peers_rep = (self.P + torch.eye(self.N)).T.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
        probas_peers = peers_rep * output_rep # (batch_size, N, N)
        probas_prod = probas_parents * torch.max(probas_peers, dim=1).values # (batch_size, N) On somme car on travail sur les parents => au plus 1 parent
        d_mat = torch.sum(H_rep, dim=1) - probas_parents + probas_prod # (batch_size, N)
        d_mat = d_mat * probas_parents # (batch_size, N)
        return d_mat

    def e_score(self, output):
        # 
        output_rep = output.unsqueeze(-1).repeat(1, 1, self.N) # (batch_size, N, N)
        peers_rep = self.P.T.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
        probas_peers = peers_rep * output_rep # (batch_size, N, N)

        probas_melee = probas_peers * output.unsqueeze(1).repeat(1, self.N, 1) # (batch_size, N, N)

        m = torch.maximum(self.P.sum(dim=1).unsqueeze(0), torch.tensor(1)) # (batch_size, N)

        e_m = -1 + probas_melee.sum(dim=1) / m # (batch_size, N)

        e_mat = e_m * (probas_peers.sum(dim=1) / m) # (batch_size, N)

        return e_mat