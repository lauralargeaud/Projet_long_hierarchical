import torch

class MessagePassing:

    def __init__(self, H_raw, P_raw, M_raw, La_raw, iter_count, device):
        self.H = torch.tensor(H_raw, dtype=torch.float32).to(device) # (N, N)
        self.P = torch.tensor(P_raw, dtype=torch.float32).to(device)
        self.M = torch.tensor(M_raw, dtype=torch.float32).to(device)
        self.La = torch.tensor(La_raw, dtype=torch.float32).to(device)
        self.h = self.La.shape[0] # height of the tree (i.e. the number of levels of the tree)
        self.iter_count = iter_count
        self.N = self.H.shape[1]
        self.batch_size = None
        self.device = device

    # output: shape = (batch_size, N)
    def update(self, output):
        """One iterationof the message passing algorithm"""
        # c_rule OK
        # d_rule OK
        # e_rule OK
        # output += self.c_score(output)
        output += self.c_score(output) + self.d_score(output) + self.e_score(output) # (batch_size, N)
        # print("output avant softmax", output)
        
        # Idea: for each level of the tree, extract the corresponding subset of output and apply the softmax on it.
        # Then, update output with its softmax-normalized values for the current level
        for l in range(0, self.h): # each level starting from the leaves
            # extract the indexes in the nodes array of the current level nodes
            indexes = torch.where(self.La[l,:] == 1)[0]
            output[:,indexes] = output[:, indexes].softmax(dim=1)

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
        probas_noeuds_rep = output_rep.transpose(1, 2)
        probas_prod = probas_fils * probas_noeuds_rep # (batch_size, N, N)
        c_mat = H_rep - probas_fils + probas_prod # (batch_size, N, N)
        c_mat = c_mat * probas_fils # (batch_size, N, N)
        nb_fils = torch.sum(H_rep, dim=1) # (batch_size, N) avec nb_fils[id_batch, 0] le nombre de fils de la racine
        result = torch.sum(c_mat, dim=1) / torch.maximum(nb_fils, torch.tensor(1)) # (batch_size, N)
        return result

    def d_score(self, output):
        # 1−s[pv]+s[pv]·max(s[fils(pv,n)]n),
        H_rep = self.H.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N) (ici on travail avec H.T, mais H.T.T = H)
        output_rep = output.unsqueeze(2).repeat(1, 1, self.N) # (batch_size, N, N)
        probas_parents = H_rep * output_rep # (batch_size, N , N) avec probas_fils[id_batch, :, id_noeud] = les probas du parent du noeud id_noeud
        probas_parents = torch.sum(probas_parents, dim=1)
        # print("probas_parents", probas_parents) # OK
        peers_rep = (self.P + torch.eye(self.N).to(self.device)).T.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
        probas_peers = peers_rep * output_rep # (batch_size, N, N)
        # print("Tranche de la 1e image du batch", probas_peers[0,:,:])
        probas_prod = probas_parents * torch.max(probas_peers, dim=1).values # (batch_size, N) On somme car on travail sur les parents => au plus 1 parent
        # print("probas_prod", probas_prod) # KO (pour le noeud 2 on n'a pas la bonne valeur)
        d_mat = torch.sum(H_rep, dim=1) - probas_parents + probas_prod # (batch_size, N)
        d_mat = d_mat * probas_parents # (batch_size, N)
        # print("hd", d_mat)
        return d_mat

    def e_score(self, output):
        # 
        output_rep = output.unsqueeze(-1).repeat(1, 1, self.N) # (batch_size, N, N)
        peers_rep = self.P.T.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
        probas_peers = peers_rep * output_rep # (batch_size, N, N)

        probas_melee = probas_peers * output.unsqueeze(1).repeat(1, self.N, 1) # (batch_size, N, N)

        m = torch.maximum(self.P.sum(dim=1).unsqueeze(0), torch.tensor(1)) # (1, N)

        h_e = -1 + probas_melee.sum(dim=1) / m # (batch_size, N)

        e_mat = h_e * (probas_peers.sum(dim=1) / m) # (batch_size, N)

        # he_rep = h_e.unsqueeze(2).repeat(1, 1, self.N)
        # e_mat = torch.sum(probas_peers * he_rep, dim=1) / m 


        # print("he", e_mat)
        return e_mat