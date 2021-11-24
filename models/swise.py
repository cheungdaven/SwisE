from torch import nn
import torch
from torch.distributions.normal import Normal
from .rotx import RotX
from manifolds.poincare import PoincareBall
from manifolds.euclidean import Euclidean
from manifolds.sphere import Spherical
import numpy as np
import torch.nn.functional as F
import time


#torch.manual_seed(1234)  

SwisE_MODELS = ["SwisE"]


class SparseDispatcher(object):

    def __init__(self, num_experts, gates, device):
        self._gates = gates
        self.device = device
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        res = torch.split(inp_exp, self._part_sizes, dim=0)
        return res

    def combine(self, expert_out, multiply_by_gates=True, use_log_sum_exp=True):
        if use_log_sum_exp:
            stitched = torch.cat(expert_out, 0).exp()
        else:
            stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=self.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())

        combined[combined == 0] = np.finfo(float).eps

        if use_log_sum_exp:
            return combined.log()
        else:
            return combined
        

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SwisE(nn.Module):
    """Trainable curvature for each relationship."""

    def __init__(self, args, num_experts, manifolds, device, noisy_gating=True, k=2):
        super(SwisE, self).__init__()
        print(k, manifolds)
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.hidden_size = args.rank
        self.k = k
        self.device = device
        self.sizes = args.sizes
        self.loss_coef = args.loss_coef
        if args.init_size == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes = args.sizes
        self.rank = rank = args.rank
        hyperbolic = PoincareBall()
        euclidean = Euclidean()
        spherical = Spherical()
        manifold_dict = {"Hyperbolic": hyperbolic,
                         "Euclidean": euclidean,
                         "Spherical": spherical}
        self.experts = nn.ModuleList([RotX(args, manifold_dict[manifolds[i]]) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(self.hidden_size * self.num_experts * 3, num_experts),
                                   requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.hidden_size * self.num_experts * 3, num_experts),
                                    requires_grad=True)


        self.entity = nn.Embedding(args.sizes[0], self.hidden_size * self.num_experts)
        self.rel = nn.Embedding(args.sizes[1], 2 * self.hidden_size * self.num_experts)
        self.entity.weight.data.normal_(0., 1.0 / self.entity.embedding_dim )
        self.rel.weight.data.normal_(0., 1.0 / self.rel.embedding_dim )
        self.rel_diag = nn.Embedding(self.sizes[1], self.hidden_size * self.num_experts)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.hidden_size * self.num_experts),
                                                   dtype=self.data_type) - 1.0

        self.bh = nn.Embedding(self.sizes[0], 1)
        self.bh.weight.data = torch.zeros((self.sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(self.sizes[0], 1)
        self.bt.weight.data = torch.zeros((self.sizes[0], 1), dtype=self.data_type)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))

        assert (self.k <= self.num_experts)

        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1 * 1), dtype=self.data_type, device=self.device)
        else:
            c_init = torch.ones((1, 1 * 1), dtype=self.data_type, device=self.device)
        self.c = nn.Parameter(c_init, requires_grad=True)
        
        # using CNN gates
        self.drop_layer = nn.Dropout(p=0.2)

        self.width = 4
        self.height = 32
        self.cnn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=3)
        self.hidden2experts = nn.Linear(self.width*self.height, self.num_experts) 
        self.cnn_noise = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=3)  
        
    		# The linear layer that maps from hidden state space to tag space
        self.hidden2experts_noise = nn.Linear(self.width*32, self.num_experts)
        
        

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prop_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)

        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=self.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        torch.set_printoptions(profile="full")
     
        x = x.view(-1, 1, self.num_experts * 3, self.hidden_size)
        clean_logits = self.hidden2experts(self.drop_layer(self.cnn(x).view(-1, self.width* self.height)))
        raw_noise_stddev = self.hidden2experts_noise(self.cnn_noise(x).view(-1, self.width* self.height))
        noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * 1)
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev * train)
        logits = noisy_logits
        
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        if not train:
            self.space_distribution = top_k_indices
            
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prop_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        
        return gates, load

    def forward(self, queries, these_queries, eval_mode=False, target_mode=False):

        start_time = time.time()
       
        lhs_biases = self.bh(queries[:, 0])
        if eval_mode:
            rhs_biases = self.bt.weight
            rhs_e = self.entity.weight
        else:
            rhs_biases = self.bt(these_queries[:, 2])
            rhs_e = self.entity(these_queries[:, 2])

        head = self.entity(queries[:, 0])
        rel = self.rel(queries[:, 1])
        rel_diag = self.rel_diag(queries[:, 1])

        c = F.softplus(self.c[queries[:, 1]]) 

        self.iseval= eval_mode
        if not self.num_experts == self.k:
            x = torch.cat([head, rel], dim=-1)
            gates, load = self.noisy_top_k_gating(x, not eval_mode)
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= self.loss_coef
            dispatcher = SparseDispatcher(self.num_experts, gates, self.device)
            head = dispatcher.dispatch(head)
            if not eval_mode:
                rhs_e = dispatcher.dispatch(rhs_e)
            rel = dispatcher.dispatch(rel)
            rel_diag = dispatcher.dispatch(rel_diag)
            c = dispatcher.dispatch(c)
            lhs_biases = dispatcher.dispatch(lhs_biases)
            rhs_biases = dispatcher.dispatch(rhs_biases)
            gates = dispatcher.expert_to_gates()
 

        expert_outputs = []
        if self.num_experts == self.k:
            for i in range(self.num_experts):   
                expert_outputs.append(self.experts[i](queries, head[:, i * self.hidden_size:(i + 1) * self.hidden_size],
                                            rhs_e[:, i * self.hidden_size:(i + 1) * self.hidden_size],
                                            rel[:, i * self.hidden_size * 2:(i + 1) * self.hidden_size * 2],
                                            rel_diag[:, i * self.hidden_size:(i + 1) * self.hidden_size],
                                            lhs_biases,
                                            rhs_biases.transpose(1, 0) if eval_mode else rhs_biases,
                                            c, eval_mode))
    
            loss = 0
            y = torch.sum(torch.transpose(torch.stack(expert_outputs).exp(), 0, 1), dim=1).log() 
            self.space_distribution = None
        else:  
            for i in range(self.num_experts):
                if head[i].size(0) != 0:
                    expert_outputs.append(self.experts[i](queries, head[i][:, i * self.hidden_size:(i + 1) * self.hidden_size],
                                rhs_e[:, i * self.hidden_size:(i + 1) * self.hidden_size] if eval_mode else rhs_e[i][:,
                                                                                                            i * self.hidden_size:(
                                                                                                                                             i + 1) * self.hidden_size],
                                rel[i][:, i * self.hidden_size * 2:(i + 1) * self.hidden_size * 2],
                                rel_diag[i][:, i * self.hidden_size:(i + 1) * self.hidden_size],
                                lhs_biases[i].unsqueeze(dim=1), 
                                rhs_biases[i].unsqueeze(dim=1),
                                c[i].unsqueeze(dim=1) , eval_mode))
            y = dispatcher.combine(expert_outputs, multiply_by_gates=False, use_log_sum_exp = not (self.num_experts == self.k))
        factors = (self.entity(queries[:, 0]), self.rel(queries[:, 1]), self.entity(queries[:, 2]))
        
        return y, factors, loss

    def get_ranking(self, queries, filters, batch_size=1000, mode="rhs"):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            spaces = []
            
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                scores, _, _ = self.forward(these_queries, queries, eval_mode=True, target_mode=True)
                spaces.append(self.space_distribution)
                targets, _, _ = self.forward(these_queries, these_queries, eval_mode=False, target_mode=True)
                
                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
            if not self.num_experts == self.k:
                spaces = torch.cat(spaces, dim=0)
                spaces = spaces.cpu().numpy()
                np.save(mode + 'spaces', spaces)
                np.save(mode + 'queries', queries.cpu().numpy())

        return ranks

    def compute_metrics(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.

        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        start_eval = time.time()
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
        
            q = examples.clone()
            
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size, mode=m)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))
        print("eval time", time.time() - start_eval)
        return mean_rank, mean_reciprocal_rank, hits_at





