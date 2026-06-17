import numpy as np
import torch
from utils import artanh, p_exp_map, p_log_map, full_p_exp_map, p_sum


class MuRP(torch.nn.Module):
    def __init__(self, d, dim, pretrained_entity_embedding=None, pretrained_relation_embedding=None):
        super(MuRP, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations),
                                        dim)), dtype=torch.double, requires_grad=True))
        self.bs = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True))
        self.bo = torch.nn.Parameter(torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True))
        self.loss = torch.nn.BCEWithLogitsLoss()

        # 请根据你 model.py 里的实际变量名替换 self.E 和 self.R
        # 比如可能是 self.entity_embeddings 或者 self.Eh

        if pretrained_entity_embedding is not None:
            print("🔥 警告: 正在强行覆盖 Entity 权重 (Eh)!")
            self.Eh.weight.data.copy_(pretrained_entity_embedding)

        if pretrained_relation_embedding is not None:
            # ⚠️ 注意：这里 self.rvh 需要根据你上面实际的变量名替换（可能是 self.Rh, self.r_emb 等）
            # 如果找不到，可以先把 if pretrained_relation_embedding 这一小段注释掉，先确保 Entity 跑通！
            print("🔥 警告: 正在强行覆盖 Relation 权重!")
            self.rvh.weight.data.copy_(pretrained_relation_embedding)  # 请核对变量名

    def forward(self, u_idx, r_idx, v_idx):
        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        Ru = self.Wu[r_idx]
        rvh = self.rvh.weight[r_idx]

        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1,
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1,
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1,
                          rvh/(torch.norm(rvh, 2, dim=-1, keepdim=True)-1e-5), rvh)
        u_e = p_log_map(u)
        u_W = u_e * Ru
        u_m = p_exp_map(u_W)
        v_m = p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1,
                          u_m/(torch.norm(u_m, 2, dim=-1, keepdim=True)-1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1,
                          v_m/(torch.norm(v_m, 2, dim=-1, keepdim=True)-1e-5), v_m)

        sqdist = (2.*artanh(torch.clamp(torch.norm(p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1-1e-5)))**2

        return -sqdist + self.bs[u_idx] + self.bo[v_idx]
