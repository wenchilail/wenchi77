import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os

# 英文停用词列表（内置，不依赖NLTK）
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
}

def review_to_words(raw_review):
    """
    将原始电影评论转换为处理后的单词序列
    """
    # 1. 移除HTML标签
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    
    # 2. 移除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 3. 转换为小写并分割成单词
    words = letters_only.lower().split()
    
    # 4. 移除停用词
    meaningful_words = [w for w in words if not w in ENGLISH_STOPWORDS]
    
    # 6. 将单词连接成一个字符串，用空格分隔
    return(" ".join(meaningful_words))

def main():
    print("=== Part 1: Bag of Words ===")
    
    # 设置数据路径
    data_dir = "word2vec-nlp-tutorial"
    
    # 读取训练数据
    print("\n1. 读取训练数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    
    # 清理所有训练评论
    print("2. 清理训练数据评论...")
    num_reviews = train["review"].size
    clean_train_reviews = []
    
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print(f"   已处理 {i+1} / {num_reviews} 条评论")
        clean_train_reviews.append(review_to_words(train["review"][i]))
    
    # 创建词袋模型
    print("\n3. 创建词袋模型...")
    vectorizer = CountVectorizer(analyzer="word", 
                                 tokenizer=None, 
                                 preprocessor=None, 
                                 stop_words=None, 
                                 max_features=5000)
    
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    
    print(f"   训练特征形状: {train_data_features.shape}")
    
    # 查看词汇表
    vocab = vectorizer.get_feature_names_out()
    print(f"   词汇表大小: {len(vocab)}")
    
    # 训练随机森林
    print("\n4. 训练随机森林分类器 (100棵树)...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])
    
    # 读取测试数据
    print("\n5. 读取和处理测试数据...")
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    
    num_reviews = len(test["review"])
    clean_test_reviews = []
    
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print(f"   已处理 {i+1} / {num_reviews} 条评论")
        clean_test_reviews.append(review_to_words(test["review"][i]))
    
    # 转换测试数据
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()
    
    # 预测
    print("\n6. 预测测试数据...")
    result = forest.predict(test_data_features)
    
    # 创建提交文件
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
    
    print("\n=== 完成！提交文件已保存为 Bag_of_Words_model.csv ===")

if __name__ == "__main__":
    main()
