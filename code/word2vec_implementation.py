import pandas as pd
import numpy as np
import re
import logging
from bs4 import BeautifulSoup
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import time
import os

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    """
    将评论转换为单词列表
    """
    # 移除HTML标签
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    
    # 移除非字母字符
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 转换为小写并分割
    words = review_text.lower().split()
    
    # 可选：移除停用词
    if remove_stopwords:
        words = [w for w in words if not w in ENGLISH_STOPWORDS]
    
    return words

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    """
    将评论分割为句子，每个句子是单词列表
    """
    # 使用NLTK分词器将段落分割为句子
    raw_sentences = tokenizer.tokenize(review.strip())
    
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    
    return sentences

def makeFeatureVec(words, model, num_features):
    """
    为给定的评论创建平均特征向量
    """
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    
    # 索引词汇表用于更快的查找
    index2word_set = set(model.wv.index_to_key)
    
    # 遍历评论中的每个单词，如果在模型的词汇表中，就添加到特征向量中
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model.wv[word])
    
    # 除以单词数量得到平均值
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    """
    计算所有评论的平均特征向量
    """
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    
    for review in reviews:
        if counter % 1000 == 0:
            print(f"评论 {counter} 共 {len(reviews)}")
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    
    return reviewFeatureVecs

def create_bag_of_centroids(wordlist, word_centroid_map):
    """
    创建词袋质心
    """
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    
    return bag_of_centroids

def main():
    print("=== Word2Vec 完整实现 ===")
    
    data_dir = "word2vec-nlp-tutorial"
    
    # ========== Part 2: 训练 Word2Vec 模型 ==========
    print("\n--- Part 2: 训练 Word2Vec 模型 ---")
    
    # 读取所有数据
    print("\n1. 读取数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(data_dir, "unlabeledTrainData.tsv", "unlabeledTrainData.tsv"), 
                                   header=0, delimiter="\t", quoting=3)
    
    print(f"已读取 {len(train)} 条标注训练数据, {len(test)} 条测试数据, {len(unlabeled_train)} 条未标注训练数据")
    
    # 简单的句子分割器（不使用NLTK）
    print("\n2. 准备训练数据...")
    sentences = []
    
    print("   处理标注训练数据...")
    for review in train["review"]:
        # 简单的句子分割
        text = BeautifulSoup(review, features="html.parser").get_text()
        text = re.sub("[^a-zA-Z.!?]", " ", text)
        text = text.lower()
        # 按句子结束符号分割
        raw_sentences = re.split('[.!?]', text)
        for sent in raw_sentences:
            words = sent.strip().split()
            if len(words) > 0:
                sentences.append(words)
    
    print("   处理未标注训练数据...")
    for review in unlabeled_train["review"]:
        text = BeautifulSoup(review, features="html.parser").get_text()
        text = re.sub("[^a-zA-Z.!?]", " ", text)
        text = text.lower()
        raw_sentences = re.split('[.!?]', text)
        for sent in raw_sentences:
            words = sent.strip().split()
            if len(words) > 0:
                sentences.append(words)
    
    print(f"   总共 {len(sentences)} 个句子用于训练")
    
    # 设置Word2Vec参数
    num_features = 300    # 词向量维度
    min_word_count = 40   # 最小词频
    num_workers = 4       # 并行线程数
    context = 10          # 上下文窗口大小
    downsampling = 1e-3   # 高频词下采样
    
    print("\n3. 训练 Word2Vec 模型...")
    print("   这可能需要几分钟...")
    
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              vector_size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    
    # 不需要继续训练，释放内存
    model.init_sims(replace=True)
    
    # 保存模型
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    print(f"   模型已保存为 {model_name}")
    
    # 测试模型
    print("\n4. 测试模型...")
    try:
        print(f"   'woman' 最相似的词: {model.wv.most_similar('woman')[:3]}")
        print(f"   'king' - 'man' + 'woman' = {model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[0]}")
    except:
        print("   某些词不在词汇表中")
    
    # ========== Part 3: 使用 Word2Vec 向量 ==========
    print("\n--- Part 3: 使用 Word2Vec 向量进行情感分析 ---")
    
    # 方法1: 向量平均
    print("\n方法1: 向量平均")
    
    print("\n1. 为训练集和测试集创建平均向量...")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
    
    print("\n2. 训练随机森林...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["sentiment"])
    
    print("\n3. 预测并保存结果...")
    result = forest.predict(testDataVecs)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print("   向量平均方法结果已保存为 Word2Vec_AverageVectors.csv")
    
    # 方法2: 聚类
    print("\n\n方法2: 聚类 (K-Means)")
    
    start = time.time()
    
    # 设置聚类数为词汇表大小的1/5
    word_vectors = model.wv.vectors
    num_clusters = int(word_vectors.shape[0] / 5)
    
    print(f"\n1. 使用 K-Means 对 {word_vectors.shape[0]} 个单词进行聚类，分为 {num_clusters} 个簇...")
    print("   这可能需要一些时间...")
    
    kmeans_clustering = KMeans(n_clusters=num_clusters, n_init=10)
    idx = kmeans_clustering.fit_predict(word_vectors)
    
    end = time.time()
    elapsed = end - start
    print(f"   聚类耗时: {elapsed:.2f} 秒")
    
    # 创建单词到质心的映射
    word_centroid_map = dict(zip(model.wv.index_to_key, idx))
    
    print("\n2. 查看几个聚类的内容...")
    for cluster in range(min(10, num_clusters)):
        words = []
        for i in range(len(idx)):
            if idx[i] == cluster:
                words.append(model.wv.index_to_key[i])
                if len(words) > 10:
                    break
        print(f"   聚类 {cluster}: {', '.join(words)}")
    
    print("\n3. 创建质心包...")
    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
    
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1
    
    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
    
    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1
    
    print("\n4. 训练随机森林...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_centroids, train["sentiment"])
    
    print("\n5. 预测并保存结果...")
    result = forest.predict(test_centroids)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_Clustering.csv", index=False, quoting=3)
    print("   聚类方法结果已保存为 Word2Vec_Clustering.csv")
    
    print("\n=== 所有部分完成！ ===")
    print("\n生成的文件:")
    print("  - Bag_of_Words_model.csv (来自 Part 1)")
    print("  - Word2Vec_AverageVectors.csv (向量平均方法)")
    print("  - Word2Vec_Clustering.csv (聚类方法)")
    print("  - 300features_40minwords_10context (Word2Vec模型)")

if __name__ == "__main__":
    main()
