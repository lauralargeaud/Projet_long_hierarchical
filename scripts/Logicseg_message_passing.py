import torch

class MessagePassing:

    def __init__(self, H_raw, P_raw, M_raw, iter_count, device):
        self.H = torch.tensor(H_raw, dtype=torch.float32).to(device) # (N, N)
        self.P = torch.tensor(P_raw, dtype=torch.float32).to(device)
        self.M = torch.tensor(M_raw, dtype=torch.float32).to(device)
        self.iter_count = iter_count
        self.nb_nodes = self.H.shape[1]
        self.batch_size = None
        self.N = None

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
        self.N = output.shape[1]
        for i in range (self.iter_count):
            output = self.update(output)
        return output
    
    # output: (batch_size, N)
    # result: (batch_size, N) 
    def c_score(self, output):
        """Pour chaque v_i = output[id_batch, i], le terme somme{proba(fils(i)).hC(fils(i),i)} stocké dans result[id_batch, i]"""
        # 1 − s[fils_v] + s[fils_v]·s[v]
        H_rep = self.H.T.unsqueeze(0).repeat(self.batch_size, 1, 1) # (batch_size, N, N)
        output_rep = output.unsqueeze(2).repeat(self.N) # (batch_size, N, N)
        probas_fils = H_rep * output_rep # (batch_size, N , N) avec probas_fils[id_batch, :, id_noeud] = les probas des fils du noeud id_noeud
        probas_prod = probas_fils * output_rep # (batch_size, N, N)
        c_mat = H_rep - probas_fils + probas_prod # (batch_size, N, N)
        c_mat = c_mat * probas_fils # (batch_size, N, N)
        nb_fils = torch.sum(H_rep, dim=1) # (batch_size, N) avec nb_fils[id_batch, 0] le nombre de fils de la racine
        result = torch.sum(c_mat, dim=1) / nb_fils # (batch_size, N)
        return result


    # def c_score_old(self, s_k):
    #     batch_size = s_k.shape[1]
    #     #-------------C-message (Eq. 16)-------------#
    #     # (N_p x |V| x 1)*(1 x |V| x HW)
    #     c_f = self.H.unsqueeze(-1) * s_k.unsqueeze(0)
    #     # (N_p x 1 x HW)*(N_p x |V| x HW)
    #     c_m = s_k[:self.N_p].unsqueeze(1) * c_f
    #     # 1−sk [v]+sk [v]· sk [pv ]
    #     c_m = 1 - c_f + c_m
    #     #-----gather received C-messages (Eq. 17)----#
    #     # (N_p x HW)
    #     c_s = (c_f * c_m).sum(dim=1)
    #     c_s = c_s / self.H.sum(dim=1).unsqueeze(-1)
    #     # (|V| x HW)
    #     c = torch.zeros(self.nb_nodes, batch_size)
    #     c[:self.N_p, :] = c_s
    #     return c

    # def d_score(self, s_k):
    #     #-------------D-message (Eq. 16)-------------#
    #     # (N_p x |V| x 1)*(1 x |V| x HW)
    #     d_f = T.unsqueeze(-1) * s_k.unsqueeze(0)
    #     # (N_p x HW)*(N_p x HW)
    #     d_m = s_k[:N_p] * d_f.max(dim=1)
    #     # 1−sk [v]+sk [v]·max({sk [cn
    #     v ]}n)
    #     d_m = 1-s_k[:N_p] + d_m
    #     #-----gather received D-messages (Eq. 17)----#
    #     # (N_p x HW)x(N_p x HW)
    #     d_s = s_k[:N_p] * d_m
    #     # (N_p x 1 x HW)*(N_p x |V| x 1)
    #     d_s = d_s.unsqueeze(1) * T.unsqueeze(-1)
    #     # (|V| x HW)
    #     d_s = d_s.sum(dim=0)
    # return d_s

    # def e_score(self, s_k):
    #     #-------------E-message (Eq. 16)-------------#
    #     # (|V| x |V| x 1)*(1 x |V| x HW)
    #     e_f = P.unsqueeze(-1) * s_k.unsqueeze(0)
    #     # (|V| x 1 x HW)*(|V| x |V| x HW)
    #     e_m = s_k.unsqueeze(1) * e_f
    #     # − (1− 1
    #     M
    #     PM
    #     m=1sk [v] · sk [am
    #     v ])
    #     e_m = -1+e_m.sum(dim=1)/P.sum(dim=1).unsqueeze(-1)
    #     #-----gather received E-messages (Eq. 17)----#
    #     # (|V| x HW)
    #     e_s = e_f.sum(dim=1) / P.sum(dim=1).unsqueeze(-1)
    #     # E-message should be same for all nodes in the
    #     same hierarchical level
    #     e_s = e_m * e_s
    # return e_s