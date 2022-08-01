from numpy import require
import torch
import torch.nn as nn
import networkx as nx
import dgl
import scipy as sp

class AutomatonPELayer(nn.Module):
    def __init__(self, args):
        super(AutomatonPELayer, self).__init__()
        self.device = 'cpu'
        self.num_states = args.num_states
        self.directed = args.directed
        self.num_layers = args.num_layers
        self.pe_type = args.pe_type

        requires_grad = args.pe_type == "learned"
        self.pos_initials = nn.ParameterList(
            nn.Parameter(torch.empty(self.num_states, 1, device=self.device), requires_grad=requires_grad)
            for _ in range(args.num_initials)
        )
        # self.pos_initial = nn.Parameter(
        #     torch.empty(self.num_states, 1, device=self.device), requires_grad=requires_grad)
        self.pos_transition = nn.Parameter(
            torch.empty(self.num_states, self.num_states, device=self.device), requires_grad=requires_grad)

        for pos_initial in self.pos_initials:
            nn.init.normal_(pos_initial)

        nn.init.orthogonal_(self.pos_transition)

        if self.pe_type == "random":
            self.linear = nn.Linear(self.num_states, self.num_states)

    def stack_strategy(self, g):
        """
            Given more than one initial weight vector, define the stack strategy.

            If n = number of nodes and k = number of weight vectors,
                by default, we repeat each initial weight vector n//k times
                and stack them together with final n-(n//k) weight vectors.
        """
        num_pos_initials = len(self.pos_initials)
        num_nodes = g.num_nodes()
        if num_pos_initials == 1:
            return torch.cat([self.pos_initials[0] for _ in range(num_nodes)], dim=1)

        remainder = num_nodes % num_pos_initials
        capacity = num_nodes - remainder
        out = torch.cat([self.pos_initials[i] for i in range(num_pos_initials)], dim=1)
        out = torch.repeat_interleave(out, capacity//num_pos_initials, dim=1)
        
        if remainder != 0:
            remaining_stack = torch.cat([self.pos_initials[-1] for _ in range(remainder)], dim=1)
            out = torch.cat([out, remaining_stack], dim=1)

        return out

    def forward(self, sentence_len):
        if self.directed:
            graph = nx.DiGraph()
            if sentence_len == 1:
                graph.add_node(0)
            else:
                graph.add_edges_from([(i, i+1) for i in range(sentence_len-1)])
        else:
            graph = nx.path_graph(sentence_len)

        g = dgl.from_networkx(graph)
        adj = g.adjacency_matrix().to_dense().to(self.device)

        if self.pe_type == "learned":
            # vec_init = torch.cat([self.pos_initial for _ in range(g.num_nodes())], dim=1)
            vec_init = self.stack_strategy(g)
            vec_init = vec_init.transpose(1, 0).flatten()

            kron_prod = torch.kron(adj.reshape(adj.shape[1], adj.shape[0]), self.pos_transition)
            B = torch.eye(kron_prod.shape[1], device=self.device) - kron_prod

            encs = torch.linalg.solve(B, vec_init)
            stacked_encs = torch.stack(encs.split(self.num_states), dim=1)
            pe = stacked_encs.transpose(1, 0).to(self.device)

        elif self.pe_type == "random":
            vec_init = torch.cat([self.pos_initial for _ in range(g.num_nodes())], dim=1)
            transition_inv = torch.inverse(self.pos_transition).to(torch.device('cpu'))

            # AX + XB = Q
            #  X = alpha
            #  A = mu inverse
            #  B = -A
            #  Q = mu inverse * pi
            transition_inv = transition_inv.numpy()
            adj = adj.cpu().numpy()
            vec_init = vec_init.numpy()
            pe = sp.linalg.solve_sylvester(transition_inv, -adj, transition_inv @ vec_init)
            pe = torch.from_numpy(pe.T).to(self.device)
            pe = self.linear(pe)

        return pe, self.pos_initials, self.pos_transition