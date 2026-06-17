import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import random

# 哈密顿乘积：四元数乘法
def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
    thus doing tf.matmul(Input,W) is equivalent to W * Inputs. * denotes the Hamilton product."""
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)   # 四元组拆分
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1) # Concatenate 4 quaternion components for a faster implementation.
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

# 对偶四元数乘法
def dual_quaternion_mul(A, B, input):
    '''(A, B) * (C, D) = (A * C, A * D + B * C)'''
    dim = input.size(1) // 2
    C, D = torch.split(input, [dim, dim], dim=1)
    A_hamilton = make_quaternion_mul(A)
    B_hamilton = make_quaternion_mul(B)
    AC = torch.mm(C, A_hamilton)
    AD = torch.mm(D, A_hamilton)
    BC = torch.mm(C, B_hamilton)
    AD_plus_BC = AD + BC
    return torch.cat([AC, AD_plus_BC], dim=1)

''' Quaternion graph neural networks! QGNN layer! https://arxiv.org/abs/2008.05089 '''
class QGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication! Concatenate 4 components of the quaternion input for a faster implementation.
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

''' Dual quaternion graph neural networks! '''
class DQGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(DQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        # (A, B) = A + eB, e^2 = 0。     A是实部四元数，B是对偶部四元数，一个对偶四元数是由两个四元数组成的。AB自动学习
        self.A = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2))
        self.B = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.A.size(0) + self.A.size(1)))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    # input：初始化的节点embedding；adj：节点共现权重矩阵
    def forward(self, input, adj):
        support = dual_quaternion_mul(self.A, self.B, input)
        output = torch.spmm(adj, support)   # 稀疏张量和稠密张量相乘
        output = self.bn(output)
        return self.act(output)

""" Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.relu,  bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # 邻接矩阵的对称归一化形式，D~(-1/2)·A~·D~(-1/2)，其中A~是A+I
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)

def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights

""" Simple GAT layer"""
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # h是特征矩阵
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)   # (N,N)，权重矩阵

        zero_vec = -9e15*torch.ones_like(e)     # -9e15是非常小的数字，GAT中的非法值，softmax之后趋近于0
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # (N,N)*(N,emb_dim)=(N,emb_dim)，根据注意力系数更新emb。   Wh有负值，attention：0~1，h_prime也有负值
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])   # 自注意力机制
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add，维度不同，低维数组复制到高维数组参与运算    (N,1),(1,N)
        e = Wh1 + Wh2.T     # 每个元素都相同

        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# GAT对于稀疏矩阵的处理，COO矩阵. https://github.com/Diego999/pyGAT/blob/master/layers.py
class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(1, 2*out_features)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    # input是节点特征矩阵，(N,emb_dim)，相当于非稀疏版本的h
    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]     # 多少个节点
        # 返回一个二维索引，(2,num_edge),第一行是行索引，第二行是列索引。稀疏矩阵没有nonzero()
        num_edge = adj._indices().size()[1]
        # edge = adj.nonzero().t()
        row_indices = adj._indices()[0]
        col_indices = adj._indices()[1]
        edge = torch.stack([row_indices,col_indices],dim=0)
        h = torch.mm(input, self.W)     # 隐含层：相当于之前的wh
        assert not torch.isnan(h).any()     # .any()判断torch.isnan(h)是否都为false

        # 邻接矩阵中行索引和列索引都对应节点，
        # h[edge[0, :], :]和h[edge[1, :], :]的维度：(num_edge,200)，按行拼接，变成(num_edge,400)，将两个相连节点的特征拼接起来
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()     # (400,num_edge)
        # self.a：bias (1,400)   edge_e：(1,num_edge)
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))    # squeeze：去掉维度为1的维度
        assert not torch.isnan(edge_e).any()
        # (N,1)，稀疏矩阵参与运算时其中的参数不能自动更新(pytorch中暂时没有其反向传播函数)，
        # edge：稀疏矩阵COO，edge_e：一阶张量，(num_edge,)
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        edge_e = self.dropout(edge_e)   # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)    # 稀疏*稠密, h_prime: N x out
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)     # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

# 一层GraphSAGE：领域节点采样+聚合
# class SageLayer(nn.Module):
# 	"""一层SageLayer"""
# 	def __init__(self, input_size, out_size, gcn=False):
# 		super(SageLayer, self).__init__()
# 		self.input_size = input_size
# 		self.out_size = out_size
# 		self.gcn = gcn
# 		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size)) #初始化权重参数w*input.T
# 		self.init_params() # 调整权重参数分布
#
# 	def init_params(self):
# 		for param in self.parameters():
# 			nn.init.xavier_uniform_(param)
#         # stdv = 1. / math.sqrt(self.weight.size(1))
#         # self.weight.data.uniform_(-stdv, stdv)
#         # if self.bias is not None:
#         #     self.bias.data.uniform_(-stdv, stdv)
#
# 	def forward(self, self_feats, aggregate_feats, neighs=None):
# 		"""
# 		Parameters:
# 			self_feats:源节点的特征向量
# 			aggregate_feats:聚合后的邻居节点特征
# 		"""
#
# 		if not self.gcn: # 如果不是gcn的话就要进行concatenate
# 			combined = torch.cat([self_feats, aggregate_feats], dim=1)
# 		else:
# 			combined = aggregate_feats
# 		combined = F.relu(self.weight.mm(combined.t())).t()
#         return combined
