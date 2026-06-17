from load_data import Data
import numpy as np
import time
import torch
from collections import defaultdict
import argparse
import scipy.sparse as sp
from collections import Counter
import itertools
from scipy import sparse

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# The new weighted Adj matrix
# 根据训练集构造的【节点共现】权重矩阵，data是实体关系名，不是id
# 但entity_idxs、relation_idxs是训练集+验证集+测试集（无法扩展图结构？没当出现新的节点时需要重新计算每条边的权重，GCN通病）
def compute_weighted_adj_matrix(data, entity_idxs, relation_idxs):
    num_entities = len(entity_idxs)
    # entity_idxs、relation_idxs合并到一个新字典，更新关系id(原来的关系id+实体数)
    tok2indx = dict()
    for _key in entity_idxs.keys():
        tok2indx[_key] = entity_idxs[_key]
    for _key in relation_idxs.keys():
        tok2indx[_key] = relation_idxs[_key] + num_entities

    # 1、计算节点共现的权重邻接矩阵（无向图）
    # Skipgrams
    back_window = 2     # 尾实体向左两个token即为头实体
    front_window = 2
    skipgram_counts = Counter()     # 统计词频
    # data：训练集。iheadline：第几行；headline：三元组
    for iheadline, headline in enumerate(data):

        # 三元组转化为新的id，头尾实体最大id=实体数量-1，关系最大id=节点数量-1
        tokens = [tok2indx[tok] for tok in headline]
        for ii_word, word in enumerate(tokens):     # 三元组中的第几个元素
            ii_context_min = max(0, ii_word - back_window)  # 不超过左下标
            ii_context_max = min(len(headline) - 1, ii_word + front_window)     # 不超过右下标
            ii_contexts = [
                ii for ii in range(ii_context_min, ii_context_max + 1)
                if ii != ii_word]
            # 有三种情况：
            # 1、(ii_word, ii_context, ii_context)
            # 2、(ii_context, ii_word, ii_context)
            # 3、(ii_context, ii_context, ii_word)
            for ii_context in ii_contexts:
                skipgram = (tokens[ii_word], tokens[ii_context])    # u\v共现没有方向，只要它们处于同一个三元组
                skipgram_counts[skipgram] += 1

        # 有边权重为1，最普通的非对称邻接矩阵。效果特别差
        # tokens = [tok2indx[tok] for tok in headline]
        # skipgram_counts[(tokens[0], tokens[1])] = 1
        # skipgram_counts[(tokens[0], tokens[2])] = 1
        # skipgram_counts[(tokens[1], tokens[2])] = 1

    # Word-Word Count Matrix，稀疏矩阵，计算频数
    row_indxs = []
    col_indxs = []
    dat_values = []
    for (tok1, tok2), sg_count in skipgram_counts.items():
        row_indxs.append(tok1)
        col_indxs.append(tok2)
        dat_values.append(sg_count)
    print('dat_values:', len(dat_values))
    # (dat_values, (row_indxs, col_indxs))是一个稀疏矩阵。压缩稀疏矩阵
    wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
    num_skipgrams = wwcnt_mat.sum() # #C
    assert (sum(skipgram_counts.values()) == num_skipgrams)

    # for creating sparse matrices，计算频率
    row_indxs = []
    col_indxs = []
    weighted_edges = []
    # reusable quantities
    sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()
    # computing weights for edges
    for (tok_word, tok_context), sg_count in skipgram_counts.items():
        nwc = sg_count  # 节点v和u同时出现的三元组个数
        Pwc = nwc / num_skipgrams
        nw = sum_over_contexts[tok_word]    # 节点v出现的三元组个数
        Pw = nw / num_skipgrams
        #
        edge_val = Pwc / Pw # for entity-entity edges
        if tok_word > len(entity_idxs) or tok_context > len(entity_idxs): # for relation-entity edges
            edge_val = Pwc
        row_indxs.append(tok_word)
        col_indxs.append(tok_context)
        weighted_edges.append(edge_val)
    edge_mat = sparse.csr_matrix((weighted_edges, (row_indxs, col_indxs)))
    # adding self-loop
    adj = edge_mat + sparse.eye(edge_mat.shape[0], format="csr")    # A~=A+I
    adj = normalize_sparse(adj)     # 正规化
    adj = sparse_mx_to_torch_sparse_tensor(adj)     # 稀疏矩阵→张量
    return adj


