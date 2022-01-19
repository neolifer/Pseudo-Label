import sys
from typing import Tuple, Optional, Union
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric.nn as gnn
Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]




class GCN2Conv(GCN2Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weights = None
        self.require_sigmoid = False
    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = torch.addmm((1-self.alpha)*x, x, self.weight1, beta = (1 - self.beta), alpha = self.beta)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        if x_0 is None:
            return x
        x_0 = x_0[:x.size(0)]
        if self.weight2 is not None:
            out = x + torch.addmm(self.alpha * x_0, x_0, self.weight2, beta=1. - self.beta,
                                  alpha=self.beta)
        self.edge_weights = edge_weight

        return out


    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                if self.require_sigmoid:
                    edge_mask = self.__edge_mask__.sigmoid()
                else:
                    edge_mask = self.__edge_mask__
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).

                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

