class Data:

    def __init__(self, data_dir = r"C:\Users\Administrator\Desktop\研究生\知识图谱链接预测\APT组织攻击偏好\data\新数据\\"):
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]

            data = [i for i in data if len(i) >= 3]

            data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
