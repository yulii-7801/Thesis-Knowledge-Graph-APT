from load_data import Data
import numpy as np
import torch
from collections import defaultdict, Counter
import scipy.sparse as sp
from scipy import sparse
import re

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)

class SilentDynamicVocabulary:
    """静默处理OOV的动态词汇表"""
    def __init__(self, entity_idxs, relation_idxs):
        self.entity_idxs = entity_idxs
        self.relation_idxs = relation_idxs
        self.base_tokens = {}
        self.base_tokens.update(entity_idxs)
        self.base_tokens.update({k: v+len(entity_idxs) for k, v in relation_idxs.items()})
        self.next_oov_idx = len(self.base_tokens) + 1000
        self.protocol_pattern = re.compile(r"\b(HTTP|FTP|CAN_TCP|SSH)\b", re.IGNORECASE)
        self.C2_KEYWORDS = {'C2', 'CommandAndControl', 'C&C'}
        self.oov_cache = defaultdict(self._generate_oov_mapping)

    def _generate_oov_mapping(self):
        idx = self.next_oov_idx
        self.next_oov_idx += 1
        return idx

    def __getitem__(self, token):
        # 协议类型静默映射
        if self.protocol_pattern.fullmatch(token):
            return self.base_tokens.get("NET_PROTOCOL", self.oov_cache[token])
        # C2术语静默映射
        if token in self.C2_KEYWORDS:
            return self.base_tokens.get("C2_SYSTEM", self.oov_cache[token])
        # 基础词汇映射
        if token in self.base_tokens:
            return self.base_tokens[token]
        # OOV静默处理
        return self.oov_cache[token]

def normalize_sparse(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(sparse_mx.shape))

def compute_weighted_adj_matrix(data, entity_idxs, relation_idxs):
    """保留headline输出的静默版本"""
    dyn_vocab = SilentDynamicVocabulary(entity_idxs, relation_idxs)
    back_window, front_window = 2, 3
    skipgram_counts = Counter()

    for headline in data:
        tokens = [dyn_vocab[tok] for tok in headline]
        
        for ii_word, word in enumerate(tokens):
            context_min = max(0, ii_word - back_window)
            context_max = min(len(headline)-1, ii_word + front_window)
            contexts = [ii for ii in range(context_min, context_max+1) if ii != ii_word]
            for ctx in contexts:
                skipgram_counts[(word, tokens[ctx])] += 1

    row, col, data = [], [], []
    total = sum(skipgram_counts.values())
    sum_context = defaultdict(int)
    for (w, _), c in skipgram_counts.items():
        sum_context[w] += c

    for (w, c), count in skipgram_counts.items():
        Pwc = count / total
        Pw = sum_context[w] / total
        decay = 0.5 if (w >= 1000 or c >= 1000) else 1.0
        edge_weight = (Pwc / Pw) * decay if Pw else 0
        row.append(w)
        col.append(c)
        data.append(edge_weight)

    # max_dim = max(max(row+[0]), max(col+[0])) + 1
    # adj = sparse.csr_matrix((data, (row, col)), shape=(max_dim, max_dim))
    # adj += sparse.eye(max_dim, format="csr")
    # return sparse_mx_to_torch_sparse_tensor(normalize_sparse(adj))

    # --- 改为下面的最小可运行版本 ---
    core_dim = len(entity_idxs) + len(relation_idxs)

    # 过滤掉所有 OOV 节点造成的边（保底操作，先让矩阵跑通）
    filtered_data, filtered_row, filtered_col = [], [], []
    for d, r, c in zip(data, row, col):
        if r < core_dim and c < core_dim:
            filtered_data.append(d)
            filtered_row.append(r)
            filtered_col.append(c)

    # 强制锁定矩阵大小与现存模型节点数一致
    adj = sparse.csr_matrix((filtered_data, (filtered_row, filtered_col)), shape=(core_dim, core_dim))
    adj += sparse.eye(core_dim, format="csr")
    return sparse_mx_to_torch_sparse_tensor(normalize_sparse(adj))
