# 对比学习相关函数

# 四种正样本
# dic_tr、dic_hr、dic_t、dic_h：共享h、t、hr、tr的三元组，以共享部分为key
def get_pos(self, examples):
    dic_tr ={}
    dic_hr = {}
    dic_t = {}
    dic_h = {}
    # 创建空字典
    for i in examples:
        dic_tr[i[0].item()] = []    # 共享头实体,d
        dic_hr[i[2].item()] = []    # 共享尾实体,c
        dic_t[(i[0].item(), i[1].item())] =[]   # 共享头实体、关系,b
        dic_h[(i[2].item(), i[1].item())] = []  # 共享尾实体、关系,a
    for i in examples:  # (h,r,t)→(h,t,r)
        dic_tr[i[0].item()].append([i[2].item(), i[1].item()])
        dic_hr[i[2].item()].append([i[0].item(), i[1].item()])
        dic_t[(i[0].item(), i[1].item())].append(i[2].item())
        dic_h[(i[2].item(), i[1].item())].append(i[0].item())
    return dic_tr, dic_hr, dic_h, dic_t