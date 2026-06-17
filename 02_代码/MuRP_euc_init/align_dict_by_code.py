import torch
import sys
import os

try:
    from load_data import Data as DataFlat
    from load_data_hier import Data as DataHier
except ImportError:
    print("错误：找不到 load_data.py 或 load_data_hier.py")
    sys.exit(1)

# ================= 修复后的严格绝对路径 =================
# 1. 严格带上 "_副本" 的数据目录
data_dir = "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_副本/data/保留/目前最好/"

# 2. 严格指向该目录下的 pt 文件
pt_file_path = "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_副本/data/保留/目前最好/noge_euclidean_best.pt"
# ===================================================

print(f"正在读取数据目录: {data_dir}")
print("1. 正在解析图谱实体...")
d_flat = DataFlat(data_dir=data_dir)
flat_ent2id = {ent: idx for idx, ent in enumerate(d_flat.entities)}

d_hier = DataHier(data_dir=data_dir)
hier_ent2id = {ent: idx for idx, ent in enumerate(d_hier.entities)}

print(f"-> Flat (NoGE使用) 实体数: {len(flat_ent2id)}")
print(f"-> Hier (MuRP使用) 实体数: {len(hier_ent2id)}")

print("\n2. 正在加载 NoGE 特征...")
try:
    pretrained = torch.load(pt_file_path)
except FileNotFoundError:
    print(f"❌ 致命错误：找不到文件 {pt_file_path}")
    sys.exit(1)

flat_emb = pretrained["entity_embeddings"]

# 创建给 MuRP 用的新矩阵，为新增的层级节点赋一个极小的随机数
hier_emb = torch.randn((len(hier_ent2id), flat_emb.shape[1]), dtype=torch.float64) * 1e-4

print("\n3. 正在进行特征精准对齐...")
match_count = 0
for ent_name, hier_id in hier_ent2id.items():
    if ent_name in flat_ent2id:
        flat_id = flat_ent2id[ent_name]
        hier_emb[hier_id] = flat_emb[flat_id]
        match_count += 1
    else:
        print(f"  [新增层级节点] {ent_name} (分配ID:{hier_id}) -> 采用近原点初始化")

print(f"\n对齐完成！成功匹配 {match_count} 个底层实体。")

# 保存在当前执行目录下（MuRP能直接读到的地方）
out_file = "noge_aligned_for_murp.pt"
torch.save({
    "entity_embeddings": hier_emb,
    "relation_embeddings": None  # 清空关系特征，让 MuRP 重新学
}, out_file)

print(f"✅ 已生成对齐后的特征文件: {os.path.abspath(out_file)}")
print("👉 下一步：确认 main_hier.py 加载的是 'noge_aligned_for_murp.pt'，然后运行 MuRP！")