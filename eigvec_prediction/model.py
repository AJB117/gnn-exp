import torch
import torch.nn as nn
import networkx as nx
import dgl

class AutomatonPELayer(nn.Module):
    def __init__(self, args):
        super(AutomatonPELayer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_states = args.num_states
        self.directed = args.directed
        self.num_layers = args.num_layers

        self.pos_initial = nn.Parameter(torch.empty(self.num_states, 1, device=self.device))
        # self.pos_initial_1 = nn.Parameter(torch.empty(self.num_states, 1, device=self.device))
        # self.pos_initial = nn.Parameter(torch.empty(self.num_states, self.num_states, device=self.device))
        self.pos_transition = nn.Parameter(torch.empty(self.num_states, self.num_states, device=self.device))

        # self.linear_list = nn.ModuleList([nn.Linear(self.num_states, self.num_states, device=self.device) for _ in range(args.num_layers)])

        # self.rff_list = nn.ModuleList([rff.layers.BasicEncoding() for _ in range(args.num_layers)])
        # self.linears = nn.ModuleList([nn.Linear(self.num_states*2, self.num_states, device=self.device) for _ in range(args.num_layers)])

        nn.init.normal_(self.pos_initial)
        # nn.init.normal_(self.pos_initial_1)
        nn.init.orthogonal_(self.pos_transition)

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
        # z = self.pos_initial.repeat(1, g.num_nodes()-1)
        z = torch.ones(self.num_states, g.num_nodes()-1, device=self.device)
        # z = torch.zeros(self.num_states, g.num_nodes()-2, device=self.device)

        vec_init = torch.cat((self.pos_initial, z), dim=1)
        vec_init = vec_init.transpose(1, 0).flatten()
        # vec_init = self.pos_initial.transpose(1, 0).flatten()

        kron_prod = torch.kron(adj.reshape(adj.shape[1], adj.shape[0]), self.pos_transition)
        B = torch.eye(kron_prod.shape[1], device=self.device) - kron_prod

        encs = torch.linalg.solve(B, vec_init)
        stacked_encs = torch.stack(encs.split(self.num_states), dim=1)
        pe = stacked_encs.transpose(1, 0).to(self.device)

        # for i in range(self.num_layers):
        #     pe = self.rff_list[i](pe)
        #     pe = self.linears[i](pe)

        # for i in range(self.num_layers):
        #     pe = self.linear_list[i](pe)

        return pe, self.pos_initial, self.pos_transition
