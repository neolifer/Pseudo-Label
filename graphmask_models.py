import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as gnn
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data.batch import Batch
import math
from typing import Callable, Union, Tuple, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor


class GNNBasic(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def arguments_read(self, *args, **kwargs):

        data: Batch = kwargs.get('data') or None
        batch = None
        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch') or None
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(args[0].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            try:
                x, edge_index, batch = data.x, data.edge_index, data.batch
            except:
                x, edge_index = data.x, data.edge_index

        return x, edge_index, batch

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class GlobalMeanPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return gnn.global_mean_pool(x, batch)


class IdenticalPool(GNNPool):

    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

class GM_GCNconv(gnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GM_GCNconv, self).__init__(*args, **kwargs)
        self.get_vertex = True
        self.require_sigmoid = False
    def forward(self, x, edge_index,
                edge_weight: OptTensor = None, message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:

                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)

                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        self.last_edge_index = edge_index
        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None, message_scale=message_scale, message_replacement=message_replacement)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: OptTensor, message_scale: Tensor,
                message_replacement: Tensor) -> Tensor:
        original_message = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        # print(original_message.shape)
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        if self.get_vertex:
            self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_replacement is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_replacement is  not None:
            return message

        return original_message

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
                    print(out.size(self.node_dim), edge_mask.size(0))
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                    print(out.size(self.node_dim), edge_mask.size(0))
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)


    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim

class GM_GCN(nn.Module):
    def __init__(self, n_layers, input_dim, hid_dim, n_classes, dropout = 0, model_level = 'node', requires_sigmoid = False):
        super(GM_GCN, self).__init__()
        self.require_sigmoid = requires_sigmoid
        self.convs = nn.ModuleList([GM_GCNconv(input_dim, hid_dim)]
                                   + [
                                       GM_GCNconv(hid_dim, hid_dim)
                                       for _ in range(n_layers - 1)
                                   ]
                                   )
        self.hidden_dims = [hid_dim for _ in range(n_layers)]
        self.outlayer = nn.Linear(hid_dim, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(hid_dim)
        for conv in self.convs:
            conv.chache = None
            conv.require_sigmoid = requires_sigmoid
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()


    def set_get_vertex(self, get_vertex = True):
        for conv in self.convs:
            conv.get_vertex = get_vertex


    def forward(self, x, edge_index, message_scales = None, message_replacement = None, batch = None, **kwargs):
        # x, edge_index = data.x, data.edge_index

        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = self.dropout(x)
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=message_replacement[i])
                # x = self.bn(x)
                x = self.relu(x)
            x = self.dropout(x)
            x = self.readout(x, batch)
            x = self.outlayer(x)
            return x
        elif message_scales:
            for i, conv in enumerate(self.convs):
                x = self.dropout(x)
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=None)
                x = self.relu(x)

            x = self.readout(x, batch)
            x = self.outlayer(x)
            return x
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.relu(x)

        x = self.readout(x, batch)
        x = self.outlayer(x)
        return x

    def get_emb(self,*args):
        if len(args) == 1:
            x, edge_index = args[0].x, args[0].edge_index
        else:
            x, edge_index = args[0], args[1]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)

        return x
    def get_latest_vertex_embedding(self):
        latest_vertex_embeddings = []
        for conv in self.convs:
            latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
            # conv.latest_vertex_embeddings = None
        return latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs


class GM_GCN2Conv(gnn.GCN2Conv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weights = None
        self.require_sigmoid = False
        self.get_vertex = False
    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, message_scale = None, message_replacement = None) -> Tensor:
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
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, message_scale=message_scale, message_replacement=message_replacement)
        if x_0 is None:
            return x
        x_0 = x_0[:x.size(0)]
        if self.weight2 is not None:
            out = x + torch.addmm(self.alpha * x_0, x_0, self.weight2, beta=1. - self.beta,
                               alpha=self.beta)
        self.edge_weights = edge_weight

        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: OptTensor, message_scale: Tensor,
                message_replacement: Tensor) -> Tensor:
        original_message = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        # print(original_message.shape)
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        if self.get_vertex:
            self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
            # self.latest_vertex_embeddings = [x_j, x_i, message] if message_scale is not None else [x_j, x_i, original_message]

        # print(self.latest_vertex_embeddings.shape[0])
        if message_replacement is not None:
            return message
        return original_message

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

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim


class GM_GCN2(nn.Module):
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers, shared_weights,dropout = 0):
        super().__init__()

        convs = []
        for i in range( num_layers ):
            convs.append(GM_GCN2Conv(dim_hidden, alpha, theta, i + 1, shared_weights = shared_weights))
        self.convs = nn.ModuleList(
            convs
        )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(dim_node, dim_hidden))
        self.fcs.append(nn.Linear(dim_hidden, num_classes))
        self.relu = nn.ReLU()
        self.dropout = dropout
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()



    def forward(self, x, edge_index, message_scales = None, message_replacement = None, batch = None):
        """
        :param Required[data]: Batch - input data
        :return:
        """
        # x, edge_index = data.x, data.edge_index
        if message_scales is not None and message_replacement is not None:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(self.fcs[0](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(self.convs[0](x, None, edge_index, message_scale = message_scales[0], message_replacement = message_replacement[0]))
            x_0 = x
            for i, conv in enumerate(self.convs):
                if i == 0:
                    continue
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.relu(conv(x, x_0, edge_index, message_scale = message_scales[i], message_replacement = message_replacement[i]))
            x = self.readout(x, batch)
            out = self.fcs[-1](x)
            return out
        elif message_scales is not None:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(self.fcs[0](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(self.convs[0](x, None, edge_index, message_scale = message_scales[0], message_replacement = None))
            x_0 = x
            for i, conv in enumerate(self.convs):
                if i == 0:
                    continue
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.relu(conv(x, x_0, edge_index, message_scale = message_scales[i], message_replacement = None))
            x = self.readout(x, batch)
            out = self.fcs[-1](x)
            return out
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.fcs[0](x))
        x_0 = x
        for i,conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(conv(x, x_0, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.readout(x, batch)
        out = self.fcs[-1](x)

        return out

    def set_get_vertex(self, get_vertex = True):
        for conv in self.convs:
            conv.get_vertex = get_vertex

    def get_emb(self,*args):
        if len(args) == 1:
            x, edge_index = args[0].x, args[0].edge_index
        else:
            x, edge_index = args[0], args[1]
        x = self.relu(self.fcs[0](x))
        x_0 = x
        for i, conv in enumerate(self.convs):
            if i == 0:
                continue
            x = self.relu(conv(x, x_0, edge_index))
        return x

    def get_latest_vertex_embedding(self):
        self.latest_vertex_embeddings = []
        for conv in self.convs:
            # temp = conv.get_latest_vertex_embedding()
            # for i in range(self.latest_vertex_embeddings):
            #     self.latest_vertex_embeddings[i].append(temp[i])
            #     conv.latest_vertex_embeddings = None
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
            # conv.latest_vertex_embeddings = None
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)

        return self.last_edge_indexs

    def set_get_vertex(self, get_vertex):
        for conv in self.convs:
            conv.get_vertex = get_vertex

class GM_GATConv(gnn.GATConv):
    def __init__(self,*args, **kwargs):
        super(GM_GATConv, self).__init__(*args, **kwargs)
        self.fill_value = 'mean'
        self.get_vertex = True
        self.require_sigmoid = False
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None, message_scale = None, message_replacement = None):
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, OptTensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, OptTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
                             size=size, message_scale = message_scale, message_replacement = message_replacement)
        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
    def message(self, x_i: Tensor,x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int],message_scale: Tensor,
                message_replacement: Tensor) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        original_message =  x_j * alpha.unsqueeze(-1)
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message.reshape(original_message.shape[0], original_message.shape[1]*original_message.shape[2]), message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
                message = message.reshape(message.shape[0], self.heads, self.out_channels)
        if message_scale is not None:
            original_message = original_message.reshape(original_message.shape[0], self.heads, self.out_channels)
        if self.get_vertex:
            self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_replacement is not None:
            return message

        return original_message

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


    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim



class GM_GAT(nn.Module):
    def __init__(self,n_layers, input_dim, hid_dim, n_classes, dropout = 0, heads = None, model_level = 'node' ):
        super(GM_GAT, self).__init__()

        if not heads:
            heads = [3 for _ in range(n_layers)]
        self.convs = nn.ModuleList([GM_GATConv(input_dim, hid_dim, heads = heads[0], dropout = dropout)]
                                   + [
                                       GM_GATConv(heads[i]*hid_dim, hid_dim, heads = heads[i + 1], dropout = dropout)
                                       for i in range(n_layers - 1)
                                   ]
                                   )
        self.hidden_dims = [hid_dim *heads[l] for l in range(n_layers)]
        self.outlayer = nn.Linear(heads[-1]*hid_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hid_dim)
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()


    def set_get_vertex(self, get_vertex = True):
        for conv in self.convs:
            conv.get_vertex = get_vertex


    def get_emb(self,x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.readout(x, None)
        return x


    def forward(self, x, edge_index, message_scales = None, message_replacement = None):
        # x, edge_index = data.x, data.edge_index
        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=message_replacement[i])
                x = self.relu(x)
                x = self.dropout(x)
            x = self.readout(x, None)
            x = self.outlayer(x)
            return x
        elif message_scales:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=None)
                x = self.relu(x)
                x = self.dropout(x)
            x = self.readout(x, None)
            x = self.outlayer(x)
            return x
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.readout(x, None)
        x = self.outlayer(x)
        return x

    def get_latest_vertex_embedding(self):
        self.latest_vertex_embeddings = []
        for conv in self.convs:
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())

        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs


class GM_SAGEConv(gnn.SAGEConv):
    def __init__(self,*args, **kwargs):
        super(GM_SAGEConv, self).__init__()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, message_scale = message_scale, message_replacement = message_replacement)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor,  message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        original_message =  x_j
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_scale is not None:
            return message
        return original_message

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim

class GM_SAGE(nn.Module):
    def __init__(self, n_layers, input_dim, hid_dim, n_classes, dropout = 0, model_level = 'node'):
        super(GM_SAGE, self).__init__()
        self.convs = nn.ModuleList([GM_SAGEConv(input_dim, hid_dim)]
                                   + [
                                       GM_SAGEConv(hid_dim, hid_dim)
                                       for _ in range(n_layers - 1)
                                   ]
                                   )
        self.hidden_dims = [hid_dim for _ in range(n_layers)]
        self.outlayer = nn.Linear(hid_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMeanPool()


    def forward(self, x, edge_index, message_scales = None, message_replacement = None):
        # x, edge_index = data.x, data.edge_index
        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=message_replacement[i])
                x = self.relu(x)
                x = self.dropout(x)
            x = self.readout(x, None)
            x = self.outlayer(x)
            return x
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.readout(x, None)
        x = self.outlayer(x)
        return x

    def get_latest_vertex_embedding(self):
        self.latest_vertex_embeddings = []
        for conv in self.convs:
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs