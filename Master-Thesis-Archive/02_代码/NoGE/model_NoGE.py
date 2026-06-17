import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import empty, matmul, tensor
import torch
from torch.cuda import empty_cache
from torch.nn import Parameter, Module
from torch.nn.functional import normalize
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import math
import numpy as np
from gnn_layers import *
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''@Dai Quoc Nguyen'''
''' QGNN encoder - customized DistMult decoder '''
class NoGE_QGNN_DistMult(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1):
        super(NoGE_QGNN_DistMult, self).__init__()

        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations

        # embedding初始化
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)

        self.lst_qgnn = torch.nn.ModuleList()
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_qgnn.append(QGNN_layer(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_qgnn.append(QGNN_layer(hid_dim, hid_dim, act=torch.tanh))

        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        for _layer in range(self.num_layers):
            X = self.lst_qgnn[_layer](X, self.adj)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        hr = h * r # following the 1-N scoring strategy
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X[:self.n_entities].t())
        pred = torch.sigmoid(hrt)
        return pred

''' (Dual) QGNN encoder - customized QuatE decoder '''
class NoGE_QGNN_QuatE(torch.nn.Module):
    def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, entity_idxs, relation_idxs, num_layers=1, variant="N", semantic_type="I", pretrain_dir="cybert", semantic_dim=100):
        super(NoGE_QGNN_QuatE, self).__init__()
        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.emb_dim = emb_dim
        self.semantic_type = semantic_type
        # self.semantic_dim = semantic_dim
        self.entity_idxs = entity_idxs
        self.relation_idxs = relation_idxs
        self.pretrain_dir = pretrain_dir
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()
        self.variant = variant
        self.lst_qgnn = torch.nn.ModuleList()

        self.initial_emb()  # 获取初始化embedding

        # 1、图神经网络
        qgnn_mode = QGNN_layer
        # 使用哪个encoder
        if self.variant == "D":
            print('Using Dual QGNN!')
            qgnn_mode = DQGNN_layer
        # 多少层GNN？第一层输入维度=emb_dim，后面层的输入维度=hid_dim。——这里使用了两层
        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_qgnn.append(qgnn_mode(self.emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_qgnn.append(qgnn_mode(hid_dim, hid_dim, act=torch.tanh))
        # 2、使用普通的线性层


    # 加入语义信息
    def initial_emb(self):
        # 不需要添加语义信息
        if self.semantic_type == 'None':
            self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
            torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        # 语义信息初始化
        elif self.semantic_type == 'I':
            node_dict = dict(list(self.entity_idxs.items())+list(self.relation_idxs.items()))
            node_list = list(node_dict.keys())
            pretrained_embeddings = self.getSemantic(node_list, self.pretrain_dir, self.emb_dim)

            self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
            self.embeddings.weight.data.copy_(pretrained_embeddings)
        # 拼接语义信息
        elif self.semantic_type == 'CT':
            semantic_dim = int(self.emb_dim / 2)

            node_dict = dict(list(self.entity_idxs.items())+list(self.relation_idxs.items()))
            node_dict = dict(list(self.entity_idxs.items())+list(self.relation_idxs.items()))
            node_list = list(node_dict.keys())
            pretrained_embeddings = self.getSemantic(node_list, self.pretrain_dir, semantic_dim)

            self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, self.emb_dim)
            structure_tensor = torch.FloatTensor(self.n_entities + self.n_relations, self.emb_dim - semantic_dim) # 结构emb
            torch.nn.init.xavier_normal_(structure_tensor)
            self.embeddings.weight.data.copy_(torch.cat((pretrained_embeddings, structure_tensor), -1))
    def getSemantic(self, list, path, new_dim):
        tokenizer = BertTokenizer.from_pretrained(path)
        cybert = BertModel.from_pretrained(path).to(device)
        inputs = tokenizer(list, return_tensors="pt", padding='max_length', max_length=16, truncation=True)
        inputs = {key:inputs[key].to(device) for key in inputs}
        outputs = cybert(**inputs)
        new_embeddings = self.pcaReduction(outputs.pooler_output, new_dim) # bert固定输出768维
        return new_embeddings
    def pcaReduction(self, embeddings, new_dim):
        pca = PCA(n_components=new_dim)
        m = embeddings.detach().cpu().numpy()
        new_embeddings = pca.fit_transform(m)
        new_embeddings_tensor = torch.from_numpy(new_embeddings)

        return new_embeddings_tensor

    # e1_idx、r_idx：节点共现时的实体、关系id（由于实体、关系的不同组合，一个实体或关系会出现多次）
    # lst_indexes：节点序号
    def forward(self, e1_idx, r_idx, lst_indexes, p_tail=None, p_rel=None, p_head=None, p_h_r=None, mod=None):
        # 从初始化embedding矩阵中获取每个节点的embedding, X:节点数*emb_dim, lst_indexes:[0~节点数-1]的列表
        X = self.embeddings(lst_indexes)
        # 1、encoder：调用GNN层。根据 权重矩阵 调整 节点的初始embedding，X是特征矩阵
        for _layer in range(self.num_layers):
            X = self.lst_qgnn[_layer](X, self.adj)  # X：节点数*emb_dim，adj：节点数*节点数

        h = X[e1_idx.to(torch.long)]    # 获取一个批次的头实体嵌入
        r = X[r_idx.to(torch.long) + self.n_entities]   # 获取一个批次的关系嵌入，关系在所有节点中的id=关系id+实体数量
        T = X[:self.n_entities]     # 所有实体嵌入：实体节点数量*emb_dim

        # 得到每个节点的对偶四元数向量表示
        # 将每个节点当作 对偶四元数 ，按列分为8个部分，拆分为两个 四元数 的形式，再合并成一个四元数的形式（确保是4的倍数？）
        if self.variant == "D":
            # normalized_r = dual_normalization(r)
            # hr = vec_vec_dual_multiplication(h, normalized_r)
            size = h.size(1) // 8   # emb_dim // 8 = 200//8=25
            # 每个对偶四元数都是由两个四元数组成：h=q+ε·p （q和p都是四元数，ε平方=0）
            hr1, hi1, hj1, hk1, hr2, hi2, hj2, hk2 = torch.split(h, size, dim=1)
            h = torch.cat([hr1, hr2, hi1, hi2, hj1, hj2, hk1, hk2], dim=1)

            rr1, ri1, rj1, rk1, rr2, ri2, rj2, rk2 = torch.split(r, size, dim=1)
            r = torch.cat([rr1, rr2, ri1, ri2, rj1, rj2, rk1, rk2], dim=1)

            tr1, ti1, tj1, tk1, tr2, ti2, tj2, tk2 = torch.split(T, size, dim=1)    # 所有实体
            T = torch.cat([tr1, tr2, ti1, ti2, tj1, tj2, tk1, tk2], dim=1)

        # 2、decoder，评分函数——没有需要训练的参数？
        normalized_r = normalization(r)     # 对关系的对偶四元数进行规则化。    共现的关系数量*emb_dim
        hr = vec_vec_wise_multiplication(h, normalized_r)  # following the 1-N scoring strategy
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)

        if mod == 0:
            if p_tail is not None:  # 共享t
                pos_h = X[p_tail[:, 0]]  # 头实体id
                pos_r = X[p_tail[:, 1]]
                norm_pos_r = normalization(pos_r)
                pos_hr = vec_vec_wise_multiplication(pos_h, norm_pos_r)
                return hr, pos_hr
            # if p_rel is not None:
            #     pos_h = X[p_rel[:, 0]]
            #     pos_t = X[p_rel[:, 1]]
            #     # norm_pos_r = normalization(pos_r)
            #     pos_ht = vec_vec_wise_multiplication(pos_h, pos_t)
            #     return hr, pos_hr
        else:
            hrt = torch.mm(hr, T.t())   # (N,emb_dim)*(emb_dim,N)=N*N

            pred = torch.sigmoid(hrt)   # hr共现数量*实体数量（已经确定hr的情况下，每个实体作为尾实体构成三元组的置信度？）
            return pred

# 对比学习模块
class CL(torch.nn.Module):
    def __init__(self, rank=None, temperature=0.9, hidden_size=None):
        super().__init__()
        # self.projection = encoder_MLP(rank, hidden_size)
        self.temperature = temperature

    # 构造一个批次内的负样本
    def get_negative_mask(self, batch_size, labels1=None, labels2=None):
        if labels2 is None:
            labels1 = labels1.contiguous().view(-1, 1)
            if labels1.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels1, labels1.T).float().to(labels1.device)
        else:
            labels1 = labels1.contiguous().view(-1, 1)
            mask1 = torch.eq(labels1, labels1.T).float().to(labels1.device)
            labels2 = labels2.contiguous().view(-1, 1)
            mask2 = torch.eq(labels2, labels2.T).float().to(labels2.device)
            mask = mask1*mask2
            mask = mask.float().to(device)
        mask = mask.repeat(2, 2)
        return mask

    # 对比损失
    def pos_loss(self, self_predictions, pos_predictions, labels1=None, labels2=None):
        pos_predictions = F.normalize(pos_predictions, dim=-1)
        self_predictions = F.normalize(self_predictions, dim=-1)
        mask = self.get_negative_mask(self_predictions.shape[0], labels1, labels2).to(labels1.device)
        out = torch.cat([self_predictions, pos_predictions], dim=0)
        similarity_m = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)
        pos = (similarity_m * mask) / self.temperature
        exp_logits = torch.exp(similarity_m / self.temperature)
        pos = pos.sum(1)
        pos = pos
        neg = exp_logits * ((~mask.bool()).float())
        neg = neg.sum(dim=-1)
        pos_loss = (- pos + torch.log(neg)) / mask.sum(-1)
        pos_loss = pos_loss.mean()
        return pos_loss

    def forward(self, x1, x2, labels1=None, labels2=None):
        # x1 = self.projection(x1)
        # x2 = self.projection(x2)
        loss = self.pos_loss(x1, x2, labels1, labels2)
        return loss

''' GCN encoder - customized QuatE decoder '''
class NoGE_GCN_QuatE(torch.nn.Module):
   def __init__(self, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1):
       super(NoGE_GCN_QuatE, self).__init__()

       self.adj = adj
       self.num_layers = num_layers
       self.n_entities = n_entities
       self.n_relations = n_relations

       self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
       torch.nn.init.xavier_normal_(self.embeddings.weight.data)

       self.lst_gcn = torch.nn.ModuleList()
       for _layer in range(self.num_layers):
           if _layer == 0:
               self.lst_gcn.append(GraphConvolution(emb_dim, hid_dim, act=torch.tanh))
           else:
               self.lst_gcn.append(GraphConvolution(hid_dim, hid_dim, act=torch.tanh))

       self.bn1 = torch.nn.BatchNorm1d(hid_dim)
       self.hidden_dropout2 = torch.nn.Dropout()
       self.loss = torch.nn.BCELoss()

   def forward(self, e1_idx, r_idx, lst_indexes):
       X = self.embeddings(lst_indexes)
       for _layer in range(self.num_layers):
           X = self.lst_gcn[_layer](X, self.adj)
       h = X[e1_idx]  # I.e., simply splitting an embedding into 4 quaternion components, slightly better than using X=np.tile(X, 4) in preliminary experiments. Note that when using X=np.tile(X, 4), QuatE becomes DistMult
       r = X[r_idx + self.n_entities]

       normalized_r = normalization(r)
       hr = vec_vec_wise_multiplication(h, normalized_r) # following the 1-N scoring strategy
       hr = self.bn1(hr)
       hr = self.hidden_dropout2(hr)
       hrt = torch.mm(hr, X[:self.n_entities].t())

       pred = torch.sigmoid(hrt)
       return pred

class GAT(nn.Module):
    # nfeat: emb_dim, nhid: hid_dim, nclass: n_entities
    def __init__(self, nfeat, nhid, adj, n_entities, n_relations, variant='D', dropout=0.6, alpha=0.2, nheads=8):
        super(GAT, self).__init__()

        nclass =n_entities + n_relations
        self.adj = adj
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, nfeat)
        self.variant = variant
        # self.bn1 = torch.nn.BatchNorm1d(nhid * nheads)  # 多头注意力
        self.bn1 = torch.nn.BatchNorm1d(nhid)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()
        """Dense version of GAT."""
        self.dropout = dropout

        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 多头注意力 → 平均聚合。输出：嵌入维度
        # self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, e1_idx, r_idx, lst_indexes):
        # adj = self.adj.to_dense()
        adj = self.adj  # 稀疏矩阵
        x = self.embeddings(lst_indexes)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # (节点数，emb_dim*n_head)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))   # 聚合层，(n,1600)→(n,200)

        e1_idx = e1_idx.type(torch.long)
        r_idx = r_idx.type(torch.long)
        h = x[e1_idx]  # I.e., simply splitting an embedding into 4 quaternion components, slightly better than using X=np.tile(X, 4) in preliminary experiments. Note that when using X=np.tile(X, 4), QuatE becomes DistMult
        r = x[r_idx + self.n_entities]
        T = x[:self.n_entities]


#         if self.variant == "D":
#             # normalized_r = dual_normalization(r)
#             # hr = vec_vec_dual_multiplication(h, normalized_r)
#             size = h.size(1) // 8   # emb_dim // 8 = 200//8=25
#             # 每个对偶四元数都是由两个四元数组成：h=q+ε·p （q和p都是四元数，ε平方=0）
#             hr1, hi1, hj1, hk1, hr2, hi2, hj2, hk2 = torch.split(h, size, dim=1)
#             h = torch.cat([hr1, hr2, hi1, hi2, hj1, hj2, hk1, hk2], dim=1)

#             rr1, ri1, rj1, rk1, rr2, ri2, rj2, rk2 = torch.split(r, size, dim=1)
#             r = torch.cat([rr1, rr2, ri1, ri2, rj1, rj2, rk1, rk2], dim=1)

#             tr1, ti1, tj1, tk1, tr2, ti2, tj2, tk2 = torch.split(T, size, dim=1)    # 所有实体
#             T = torch.cat([tr1, tr2, ti1, ti2, tj1, tj2, tk1, tk2], dim=1)

#         # decoder
#         normalized_r = normalization(r)
#         hr = vec_vec_wise_multiplication(h, normalized_r)  # following the 1-N scoring strategy

        hr = torch.mul(h, r)  # 稠密矩阵，对应元素相乘（n,200）。没有使用四元素乘法
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, x[:self.n_entities].t())     # 已经确定hr时，实体t的概率

#         pred = torch.sigmoid(hrt)
        # pred = self.my_sigmoid(hrt)     # 当hrt的值＜0时，经过sigmoid可能发生指数溢出.sigmoid还可能梯度消失
        # pred = F.log_softmax(hrt, dim=1)
        pred = F.softmax(hrt, dim=1)
        return pred

    def my_sigmoid(self, X):
        Y = torch.zeros_like(X)
        dim0, dim1 = X.shape
        for i in range(dim0):
            x = X[i]
            indices_pos = torch.nonzero(x >= 0)
            indices_neg = torch.nonzero(x < 0)
            y = torch.zeros_like(x)
            y[indices_pos] = 1 / (1 + torch.exp(-x[indices_pos]))   # 只能是一维索引
            y[indices_neg] = torch.exp(x[indices_neg]) / (1 + torch.exp(x[indices_neg]))
            Y[i] = y
        return Y

# 对偶四元数的规范化
def dual_normalization(dual_q, split_dim=1): # bs x 8dim; 4xdim for each quaternion part
    '''normalization(a, b) = normalization(a) + e x ( b / norm(a) - a x (a_rb_r + a_ib_i + a_jb_j + a_kb_k) / norm(a) / norm(a)^2)
                            = normalization(a) + e x (b / norm(a) - normalization(a) x (a_rb_r + a_ib_i + a_jb_j + a_kb_k) / norm(a)^2 ) '''
    if len(dual_q.size()) == 1:
        dual_q = dual_q.unsqueeze(0)
    size = dual_q.size(1) // 2
    a, b = torch.split(dual_q, [size, size], dim=1)
    normalized_a, norm_a, q_a, dim = normalization_v2(a, split_dim) # bs x 4 x dim, bs x 1 x dim

    q_b = b.reshape(-1, 4, dim) # bs x 4 x dim
    b_div_norm_a = q_b / norm_a
    inner_ab = torch.sum(q_a * q_b, 1, True)
    normalization_a_time_inner_ab_div_norm_a_2 = normalized_a * inner_ab / (norm_a**2)
    out_b = b_div_norm_a - normalization_a_time_inner_ab_div_norm_a_2

    return torch.cat([normalized_a.reshape(-1, 4 * dim), out_b.reshape(-1, 4 * dim)], dim=1)

# 对偶四元数的乘法
def vec_vec_dual_multiplication(q, p):
    '''(a,b)*(c,d)=(a*c, a*d+b*c). * denotes the Hamilton product'''
    size = q.size(1) // 2
    a, b = torch.split(q, [size, size], dim=1)  # 两个四元数
    c, d = torch.split(p, [size, size], dim=1)
    ac = vec_vec_wise_multiplication(a, c)  # Hamilton product：四元数的乘积
    ad = vec_vec_wise_multiplication(a, d)
    bc = vec_vec_wise_multiplication(b, c)
    ad_plus_bc = ad + bc

    return torch.cat([ac, ad_plus_bc], dim=1)

# 四元数规范化
def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4  # 四元数按列拆分为4个部分
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x size，三阶张量，size是四元数1/4部分的大小
    # quaternion / norm（四元数的每部分都除以该四元数的范数），不改变shape
    quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # 按第二维求和，四元数的四部分求和
    quaternion = quaternion.reshape(-1, 4 * size)   # 四元数的四部分合并, bs * emb_dim
    return quaternion

def normalization_v2(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    norm_q = torch.sqrt(torch.sum(quaternion ** 2, 1, True)) # bs x 1 x dim
    normalized_q = quaternion / norm_q  # quaternion / norm
    return normalized_q, norm_q, quaternion, size

# 拆分四元组
def make_wise_quaternion(quaternion):  # for vector * vector quaternion element-wise multiplication
    if len(quaternion.size()) == 1:
        quaternion = quaternion.unsqueeze(0)
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    return r2, i2, j2, k2

# 列数变为原来的1/4
def get_quaternion_wise_mul(quaternion):
    size = quaternion.size(1) // 4
    quaternion = quaternion.view(-1, 4, size)
    quaternion = torch.sum(quaternion, 1)
    return quaternion

# Hamilton product：哈密顿乘积，四元数的乘法
def vec_vec_wise_multiplication(q, p):  # vector * vector
    # 四元组的拆分
    q_r, q_i, q_j, q_k = make_wise_quaternion(q)  # bs x 4dim

    qp_r = get_quaternion_wise_mul(q_r * p)  # qrpr−qipi−qjpj−qkpk
    qp_i = get_quaternion_wise_mul(q_i * p)  # qipr+qrpi−qkpj+qjpk
    qp_j = get_quaternion_wise_mul(q_j * p)  # qjpr+qkpi+qrpj−qipk
    qp_k = get_quaternion_wise_mul(q_k * p)  # qkpr−qjpi+qipj+qrpk

    return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

def regularization(quaternion):  # vectorized quaternion bs x 4dim
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

"Dual QuatE"
class DQuatE(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(DQuatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.bn1 = torch.nn.BatchNorm1d(emb_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()
    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        normalized_r = dual_normalization(r)
        hr = vec_vec_dual_multiplication(h, normalized_r)
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X[:self.n_entities].t())
        pred = torch.sigmoid(hrt)
        return pred


''' The re-implementation of Quaternion Knowledge Graph Embeddings (https://arxiv.org/abs/1904.10281), following the 1-N scoring strategy '''
class QuatE(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(QuatE, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.loss = torch.nn.BCELoss()
    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]

        # QuatE作为decoder
        normalized_r = normalization(r)
        hr = vec_vec_wise_multiplication(h, normalized_r)   # Hamilton product
        hrt = torch.mm(hr, X[:self.n_entities].t())  # following the 1-N scoring strategy in ConvE

        pred = torch.sigmoid(hrt)
        return pred


''' DistMult, following the 1-N scoring strategy '''
class DistMult(torch.nn.Module):
    def __init__(self, emb_dim, n_entities, n_relations):
        super(DistMult, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embeddings = torch.nn.Embedding(self.n_entities + self.n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.embeddings.weight.data)
        self.loss = torch.nn.BCELoss()

    def forward(self, e1_idx, r_idx, lst_indexes):
        X = self.embeddings(lst_indexes)
        h = X[e1_idx]
        r = X[r_idx + self.n_entities]
        hr = h * r
        hrt = torch.mm(hr, X[:self.n_entities].t())  # following the 1-N scoring strategy in ConvE
        pred = torch.sigmoid(hrt)
        return pred
