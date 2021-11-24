import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from utils.euclidean import givens_rotations, givens_reflection
BASE_MODELS = ["RotX"]
import time

from manifolds.poincare import PoincareBall
from manifolds.sphere import Spherical


class RotX(nn.Module, ABC):
    """Trainable curvature for each relationship."""

    def __init__(self, args, manifold):
        super(RotX, self).__init__()
        if  args.init_size == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes = args.sizes
        self.rank =  rank = args.rank
        self.dropout = args.dropout
        self.bias = args.bias
        self.init_size = args.init_size
        self.gamma = nn.Parameter(torch.Tensor([args.gamma]), requires_grad=False)
       
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.manifold = manifold

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight
        else:
            return self.entity(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - self.manifold.sqdist(lhs_e, rhs_e, c, eval_mode)

    def get_queries(self, queries, head, rel, rel_diag, c):
        """Compute embedding and biases of queries."""
        head = self.manifold.expmap0(head, c)
        rel1, rel2 = torch.chunk(rel, 2, dim=1)
        rel1 = self.manifold.expmap0(rel1, c)
        rel2 = self.manifold.expmap0(rel2, c)
        lhs = self.manifold.proj(self.manifold.mobius_add(head, rel1, c), c)
        
        res1 = givens_rotations(rel_diag, lhs)
        res2 = self.manifold.mobius_add(res1, rel2, c)
        return (res2, c)


    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e

    def forward(self, queries,  head, rhs_e, rel, rel_diag, lhs_biases, rhs_biases, c, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        start_time = time.time()
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries, head, rel, rel_diag, c), lhs_biases
        rhs_e, rhs_biases = rhs_e, rhs_biases
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        return predictions

    def get_ranking(self, queries, filters, batch_size=1000):
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
            candidates = self.get_rhs(queries, eval_mode=True)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False)

                scores = self.score(q, candidates, eval_mode=True)
                targets = self.score(q, rhs, eval_mode=False)

                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
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
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at