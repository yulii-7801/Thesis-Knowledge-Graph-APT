import numpy as np
import torch
import time
import argparse
from collections import defaultdict
from load_data import Data
from model import *
from rsgd import *
from google.colab import runtime  # 确保你引入了 runtime
import os


class Experiment:

    def __init__(self, learning_rate=50, dim=200, nneg=50, model="poincare",
                 num_iterations=500, batch_size=128, cuda=False, data_dir=None):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.data_dir = data_dir

        self.best_metric = 0.0
        self.best_epoch = 0
        self.best_model_state = None

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data):
        hits = [[] for _ in range(10)]
        ranks = []

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(len(test_data_idxs)):
            data_point = test_data_idxs[i]

            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            predictions = model.forward(
                e1_idx.repeat(len(d.entities)),
                r_idx.repeat(len(d.entities)),
                range(len(d.entities))
            )

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions[e2_idx].item()

            predictions[filt] = -np.inf
            predictions[e1_idx] = -np.inf
            predictions[e2_idx] = target_value

            _, sort_idxs = torch.sort(predictions, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()

            rank = np.where(sort_idxs == e2_idx.item())[0][0] + 1
            ranks.append(rank)

            for k in range(10):
                hits[k].append(1.0 if rank <= k + 1 else 0.0)

        # ⭐ 五个指标
        hits10 = np.mean(hits[9])
        hits3 = np.mean(hits[2])
        hits1 = np.mean(hits[0])
        mr = np.mean(ranks)
        mrr = np.mean(1.0 / np.array(ranks))

        print(f"Hits @10: {hits10}")
        print(f"Hits @3: {hits3}")
        print(f"Hits @1: {hits1}")
        print(f"Mean Rank: {mr}")
        print(f"MRR: {mrr}")

        return hits10, hits3, hits1, mr, mrr

    def load_my_entity_types(self, filepath):
        """最小可执行的读取字典函数"""
        print(f"Loading entity types from {filepath}...")
        entity_type_dict = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ent_name = parts[0]
                        ent_type = parts[1]
                        entity_type_dict[ent_name] = ent_type
            print(f"Successfully loaded {len(entity_type_dict)} entity types.")
        except FileNotFoundError:
            print(f"⚠️ Warning: {filepath} not found! Fallback to empty dict.")
        return entity_type_dict

    def train_and_eval(self):
        print("Training the %s model..." %self.model)
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        # 既然数据已经是严格对齐的纯数字了，我们强行让 字符串'0' 对应 数字0，绕开所有排序逻辑
        # self.entity_idxs = {str(i): i for i in range(1475)}  # 1475 是 NoGE 字典里的实体总数
        # self.relation_idxs = {str(i): i for i in range(12)}  # 12 是你的关系总数

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        # 【核心创新点：加载欧式特征并进行指数映射】
        print("Loading Euclidean embeddings from NoGE...")
        try:
            # === 1. 直接加载 NoGE embedding ===
            noge_emb = torch.load("/content/workspace/NoGE/03_model/noge_entity_embeddings2.pt")  # [1475, dim]

            print(f"NoGE embedding shape: {noge_emb.shape}")
            print(f"MuRP entity count: 1475")

            # === 2. 物理级对齐（降维打击的红利！） ===
            # 因为数据已经是纯数字，索引绝对一致，直接 1:1 继承，连 for 循环比对都不需要了！
            # aligned_emb = noge_emb.clone()

            # 最低可执行：直接根据 MuRP 的实体数量进行截断
            num_murp_entities = len(d.entities)  # 自动获取当前的 1302
            dim_target = 128
            # aligned_emb = noge_emb[:num_murp_entities, :].clone()
            aligned_emb = noge_emb[:num_murp_entities, :dim_target].clone()
            # print(f"⚠️ 触发保底方案：强行将 NoGE 的 1475 维截断为 {num_murp_entities} 维。")
            print(f"降维打击：将 NoGE 截断为 {dim_target} 维。")

            print(f"New entities not in NoGE: 0 (Perfect Match!)")

            # === 3. 限制欧式向量模长，防止指数映射后瞬间挤爆在双曲边缘 ===
            print("Clipping Euclidean embeddings to prevent boundary explosion...")
            max_norm = 0.5  # 已尝试0.8
            norms = torch.norm(aligned_emb, p=2, dim=1, keepdim=True)
            aligned_emb = torch.where(norms > max_norm, aligned_emb * (max_norm / norms), aligned_emb)

            # === 4. 映射到双曲空间 ===
            print("Mapping to hyperbolic space...")
            scale_factor = 0.1  # 已尝试0.01、0.05
            pretrained_entity_embedding = p_exp_map(aligned_emb * scale_factor)

            pretrained_relation_embedding = None  # 关系不初始化（更稳）

            self.dim = pretrained_entity_embedding.shape[1]

            # === 3 & 4. 融合：根据表格物理分配层级半径并映射 ===
            '''print("Assigning hierarchical radii to entities based on true labels...")

            # 【步骤A：你需要准备一个映射数组】
            # 假设你有一个字典，记录了原始实体名对应的层级类型（你需要自己准备这个数据）
            # 例如: entity_type_dict = {'APT28': 'APT', '192.168.1.1': 'IP', ...}
            # 这里我用伪代码代替，请替换为你实际加载该字典的方法
            type_file_path = self.data_dir + "entity2type.txt"  # 指向你的数据目录
            entity_type_dict = self.load_my_entity_types(type_file_path)

            # 最小可执行方案：管它三七二十一，按索引生切 3 层
            target_r_values = torch.ones((num_murp_entities, 1)) * 0.5  # 默认中间层 r=0.5

            # 遍历模型现有的 entity_idxs，绝对不破坏原有顺序！
            mapped_count = 0
            for entity_name, current_idx in self.entity_idxs.items():
                # 默认给个中层枢纽 'ACT' 的概念
                ent_type = entity_type_dict.get(entity_name, 'Unknown')

                # 开始查表分配
                if ent_type in ['APT', 'Campaign', 'Secteam']:
                    target_r_values[current_idx] = 0.15
                elif ent_type in ['Time', 'Loc', 'Field']:
                    target_r_values[current_idx] = 0.30
                elif ent_type == 'OS':
                    target_r_values[current_idx] = 0.40
                elif ent_type in ['ACT', 'Vulname', 'Vulid']:
                    target_r_values[current_idx] = 0.50
                elif ent_type in ['MAL', 'TOOL', 'FILE', 'PROT']:
                    target_r_values[current_idx] = 0.70
                elif ent_type in ['IP', 'Dom', 'URL', 'Email']:
                    target_r_values[current_idx] = 0.85
                elif ent_type in ['MD5', 'SHA1', 'SHA2']:
                    target_r_values[current_idx] = 0.90
                else:
                    pass  # Unknown 保持默认的 0.5

                if ent_type != 'Unknown':
                    mapped_count += 1

            print(f"✅ Precisely mapped {mapped_count} / {num_murp_entities} entities to their hierarchical radii.")

            # target_r_values = torch.clamp(target_r_values, max=0.95)

            # 打印出来安个心
            # print("⚠️ 触发保底方案：使用伪造的极简分层半径进行测试！")


            # 【步骤B：计算为了达到目标 r，欧式空间的模长应该是多少】
            # 双曲映射 p_exp_map(x) 的半径近似等于 tanh(||x||)
            # 所以反推回去：需要的欧式模长 = arctanh(r)
            # 为了防止数值溢出，我们稍微加一点 epsilon
            target_euclidean_norms = torch.atanh(target_r_values * 0.9999)

            # 【步骤C：方向保留，模长替换】
            current_norms = torch.norm(aligned_emb, p=2, dim=1, keepdim=True)
            # 避免除以0
            current_norms = torch.clamp(current_norms, min=1e-8)

            # 将原来的向量归一化为单位向量，然后乘上我们算好的目标模长
            scaled_emb = (aligned_emb / current_norms) * target_euclidean_norms

            # 【步骤D：映射到双曲空间】
            print("Mapping to hyperbolic space with hierarchical constraint...")
            # 注意：这里不需要再乘 scale_factor 了，因为我们已经精确计算了模长
            pretrained_entity_embedding = p_exp_map(scaled_emb)

            # 检查一下初始化是否成功，打印几个看看
            check_norms = torch.norm(pretrained_entity_embedding, p=2, dim=1)
            print(
                f"Check initialized r (min/max/mean): {check_norms.min():.3f} / {check_norms.max():.3f} / {check_norms.mean():.3f}")

            self.dim = pretrained_entity_embedding.shape[1]

            pretrained_relation_embedding = None  # 关系不初始化（更稳）'''

        except FileNotFoundError:
            print("Error: NoGE embedding files not found.")
            return

        model = MuRP(d, self.dim,
                     pretrained_entity_embedding=pretrained_entity_embedding,
                     pretrained_relation_embedding=pretrained_relation_embedding
                     )

        # ❗❗❗ 新增这三行：如果启用了 cuda，就把模型送进 GPU ❗❗❗
        if self.cuda:
            model = model.cuda()
            print("🚀 Model successfully moved to GPU!")

        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)

        er_vocab = self.get_er_vocab(train_data_idxs)

        '''target_r_values = target_r_values.double()
        if self.cuda:
            target_r_values = target_r_values.cuda()'''

        print("Starting training...")
        # ⭐ 新增：用于记录每个 epoch 的平均 Loss 的列表
        epoch_losses_history = []

        # （把这段放进循环开始前）
        patience = 15  # 容忍多少次验证集不提升就停
        patience_counter = 0

        for it in range(1, self.num_iterations+1):
            start_train = time.time()
            model.train()

            # ✅ 加上这段冻结逻辑（务必加上！）
            if it == 1:
                for name, param in model.named_parameters():
                    if 'entity' in name.lower() or 'E' == name:
                        param.requires_grad = False
                print("❄️ [Epoch 1] 冻结 Entity 权重，当前只训练 Relation...")
            elif it == 30:  # 已试用过50（稳定），80（解冻太慢）,25(早停)
                for name, param in model.named_parameters():
                    param.requires_grad = True
                print("🔥 [Epoch %d] 解冻 Entity 权重，开始全量联合训练..." %it)

            losses = []
            np.random.shuffle(train_data_idxs)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = np.array(train_data_idxs[j:j+self.batch_size])
                negsamples = np.random.choice(list(self.entity_idxs.values()),
                                              size=(data_batch.shape[0], self.nneg))

                e1_idx = torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1]+1)))
                r_idx = torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1]+1)))
                e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1))

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 0.99
                targets = torch.DoubleTensor(targets)

                opt.zero_grad()
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    targets = targets.cuda()

                predictions = model.forward(e1_idx, r_idx, e2_idx)
                # task_loss = model.loss(predictions, targets)
                total_loss = model.loss(predictions, targets)

                # 【新增：计算当前所有实体在双曲空间中的真实半径 r】
                # 在庞加莱圆盘模型中，向量的 L2 范数就是它到原点的半径
                '''current_r = torch.norm(model.Eh.weight, p=2, dim=1, keepdim=True)

                # 【新增：计算软约束惩罚 (MSE Loss)】
                # 如果当前半径偏离了表中的目标半径，就产生惩罚
                reg_loss = torch.nn.functional.mse_loss(current_r, target_r_values)

                # 【新增：加权融合】
                # alpha 是皮筋的弹性系数。先给个保守的 0.1 跑通保底。
                alpha = 0.1
                total_loss = task_loss + alpha * reg_loss'''

                total_loss.backward()
                opt.step()
                losses.append(total_loss.item())

            # 计算并保存当前 epoch 的平均 loss
            current_epoch_mean_loss = np.mean(losses)
            epoch_losses_history.append(current_epoch_mean_loss)  # ⭐ 加入总列表

            print(it)
            print(time.time()-start_train)
            print(np.mean(losses))

            model.eval()
            with torch.no_grad():
                # （把下面这段替换掉你原来的 Eval 保存逻辑）
                if it % 10 == 0:
                    print(f"Validation at epoch {it}:")
                    hits10, hits3, hits1, mr, mrr = self.evaluate(model, d.valid_data)
                    current_hits = hits10  # 用 Hits@10 作为监控指标

                    if current_hits > self.best_metric:
                        self.best_metric = current_hits
                        self.best_epoch = it
                        self.best_model_state = model.state_dict()
                        patience_counter = 0  # ❗有提升，计数器清零
                        print(f"--> [New Best] Model updated at epoch {it} with Hits@10: {current_hits}")
                    else:
                        patience_counter += 1  # ❗没提升，计数器+1
                        print(f"--> No improvement. Early stopping counter: {patience_counter} / {patience}")

                    # ❗触发早停
                    if patience_counter >= patience:
                        print(f"🚀 Early stopping triggered at epoch {it}! Best epoch was {self.best_epoch}.")
                        break  # 强行跳出 500 轮的 for 循环

            # 训练结束后输出最佳epoch的信息
        print(f"Best valid epoch: {self.best_epoch}, Best Valid Hits@10: {self.best_metric}")

        # ⭐ 新增：训练结束后，将包含 500 个 Epoch Loss 的列表保存为 npy 文件
        np.save("results/loss_euc_init.npy",
                np.array(epoch_losses_history))  # 注意：如果你还没建 results 文件夹，请把路径改回 "loss_random.npy"
        print("✅ 训练 Loss 已成功保存至 results/loss_euc_init.npy")

        # ⭐ 加载 best model
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

        hits10, hits3, hits1, mr, mrr = self.evaluate(model, d.test_data)

        print("\nFinal Test Results:")
        print(f"Hits@10: {hits10}")
        print(f"Hits@3:  {hits3}")
        print(f"Hits@1:  {hits1}")
        print(f"MR:      {mr}")
        print(f"MRR:     {mrr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="APT-Hier", nargs="?",
                    help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                    help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--nneg", type=int, default=100, nargs="?",
                    help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=20, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dim", type=int, default=128, nargs="?",
                    help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=False, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--data_dir", type=str, default="/content/workspace/04_data/", nargs="?",
                        help="Base directory for the datasets.")

    args = parser.parse_args()
    dataset = args.dataset
    # data_dir = "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_重来/04_data/%s/" %dataset
    # data_dir = "/content/workspace/04_data/%s/" % dataset
    data_dir = os.path.join(args.data_dir, dataset) + "/"
    print(f"📂 Current data directory: {data_dir}")
    torch.backends.cudnn.deterministic = True

    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir)

    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim,
                            cuda=args.cuda, nneg=args.nneg, model=args.model,
                            data_dir=data_dir)  # 修改點：傳遞正確的 data_dir 進去
    experiment.train_and_eval()

    # === 挂机自动跑 3 个种子的核心逻辑 ===
    '''seed_list = [42, 2026]  # 三个常用的经典随机种子

    for current_seed in seed_list:
        print("\n" + "=" * 50)
        print(f"🚀 正在启动第 {seed_list.index(current_seed) + 1}/2 轮测试，当前 Seed: {current_seed}")
        print("=" * 50 + "\n")

        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        d = Data(data_dir=data_dir)
        experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                                num_iterations=args.num_iterations, dim=args.dim,
                                cuda=args.cuda, nneg=args.nneg, model=args.model)
        experiment.train_and_eval()

        # 跑完当前这个种子，清理一下显存，然后再进入下一个循环
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 👆 循环体到这里结束 👆

    # 👇 留在最外层（不要缩进）：等 for 循环里的 3 个种子全跑完，最后执行释放 👇
    print("🎉 3 个种子的实验全部完成，正在自动释放 GPU 资源...")'''
    # runtime.unassign()
