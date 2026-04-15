import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import os

# 英文停用词列表
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

def review_to_wordlist(review, remove_stopwords=False):
    """将评论转换为单词列表"""
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        words = [w for w in words if not w in ENGLISH_STOPWORDS]
    return words

def makeFeatureVec(words, model, num_features):
    """为给定的评论创建平均特征向量"""
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model.wv[word])
    
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    """计算所有评论的平均特征向量"""
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    
    for review in reviews:
        if counter % 1000 == 0:
            print(f"评论 {counter} 共 {len(reviews)}")
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs

def main():
    print("=== 使用 Word2Vec 向量进行情感分析 (向量平均方法) ===")
    
    data_dir = "word2vec-nlp-tutorial"
    model_name = "300features_40minwords_10context"
    
    # 检查模型是否存在
    if not os.path.exists(model_name):
        print(f"错误: 找不到模型 {model_name}，请先运行 word2vec_implementation.py")
        return
    
    # 加载模型
    print("\n1. 加载 Word2Vec 模型...")
    model = Word2Vec.load(model_name)
    num_features = model.vector_size
    print(f"   模型已加载，向量维度: {num_features}")
    
    # 测试模型
    print("\n2. 测试模型...")
    try:
        print(f"   'woman' 最相似的词: {model.wv.most_similar('woman')[:3]}")
        print(f"   'king' - 'man' + 'woman' = {model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[0]}")
    except Exception as e:
        print(f"   某些词不在词汇表中: {e}")
    
    # 读取数据
    print("\n3. 读取数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    
    # 处理评论
    print("\n4. 处理训练和测试评论...")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    
    # 创建平均向量
    print("\n5. 创建平均向量...")
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
    
    # 训练随机森林
    print("\n6. 训练随机森林分类器...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["sentiment"])
    
    # 预测
    print("\n7. 预测测试数据...")
    result = forest.predict(testDataVecs)
    
    # 保存结果
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print("\n=== 完成！结果已保存为 Word2Vec_AverageVectors.csv ===")

if __name__ == "__main__":
    main()
