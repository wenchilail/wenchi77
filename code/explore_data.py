import pandas as pd
import os

# 设置数据路径
data_dir = "word2vec-nlp-tutorial"

# 读取训练数据
print("正在读取训练数据...")
train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                    header=0, delimiter="\t", quoting=3)
print(f"训练数据形状: {train.shape}")
print(f"训练数据列: {train.columns.values}")
print("\n前5条评论:")
print(train.head())

print("\n\n第一条评论内容:")
print(train["review"][0])

# 读取测试数据
print("\n\n正在读取测试数据...")
test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                   header=0, delimiter="\t", quoting=3)
print(f"测试数据形状: {test.shape}")
print(f"测试数据列: {test.columns.values}")

# 读取未标注训练数据
print("\n\n正在读取未标注训练数据...")
unlabeled_train = pd.read_csv(os.path.join(data_dir, "unlabeledTrainData.tsv", "unlabeledTrainData.tsv"), 
                               header=0, delimiter="\t", quoting=3)
print(f"未标注训练数据形状: {unlabeled_train.shape}")
