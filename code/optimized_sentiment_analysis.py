import pandas as pd
import numpy as np
import re
import os
import pickle
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import logging

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

def clean_review(review, remove_stopwords=False):
    """优化的数据清洗函数"""
    # 移除HTML标签
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    
    # 移除URL
    review_text = re.sub(r'https?://\S+|www\.\S+', '', review_text)
    
    # 移除非字母字符，但保留一些重要的标点用于情感
    review_text = re.sub("[^a-zA-Z.!?']", " ", review_text)
    
    # 转换为小写
    review_text = review_text.lower()
    
    # 处理缩写
    review_text = re.sub(r"n't", " not", review_text)
    review_text = re.sub(r"'re", " are", review_text)
    review_text = re.sub(r"'s", " is", review_text)
    review_text = re.sub(r"'d", " would", review_text)
    review_text = re.sub(r"'ll", " will", review_text)
    review_text = re.sub(r"'ve", " have", review_text)
    review_text = re.sub(r"'m", " am", review_text)
    
    # 分割为单词
    words = review_text.split()
    
    # 移除停用词
    if remove_stopwords:
        words = [w for w in words if not w in ENGLISH_STOPWORDS]
    
    return words

def review_to_sentences(review):
    """将评论分割为句子"""
    # 简单的句子分割
    text = BeautifulSoup(review, features="html.parser").get_text()
    text = re.sub("[^a-zA-Z.!?']", " ", text)
    text = text.lower()
    raw_sentences = re.split('[.!?]', text)
    sentences = []
    for sent in raw_sentences:
        words = sent.strip().split()
        if len(words) > 0:
            sentences.append(words)
    return sentences

def makeFeatureVec(words, model, num_features):
    """创建平均特征向量"""
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    """获取所有评论的平均特征向量"""
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    
    for review in reviews:
        if counter % 1000 == 0:
            print(f"评论 {counter} 共 {len(reviews)}")
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

def load_or_train_word2vec(data_dir, cache_dir="cache"):
    """加载或训练Word2Vec模型"""
    model_file = os.path.join(cache_dir, "word2vec_model")
    sentences_file = os.path.join(cache_dir, "sentences.pkl")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 如果模型已存在，直接加载
    if os.path.exists(model_file):
        print("加载已有的Word2Vec模型...")
        model = Word2Vec.load(model_file)
        return model
    
    # 如果句子已缓存，加载句子
    if os.path.exists(sentences_file):
        print("加载已缓存的句子...")
        with open(sentences_file, 'rb') as f:
            sentences = pickle.load(f)
    else:
        # 准备句子
        print("准备训练数据...")
        train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                            header=0, delimiter="\t", quoting=3)
        unlabeled_train = pd.read_csv(os.path.join(data_dir, "unlabeledTrainData.tsv", "unlabeledTrainData.tsv"), 
                                       header=0, delimiter="\t", quoting=3)
        
        sentences = []
        
        print("处理标注训练数据...")
        for review in train["review"]:
            sentences.extend(review_to_sentences(review))
        
        print("处理未标注训练数据...")
        for review in unlabeled_train["review"]:
            sentences.extend(review_to_sentences(review))
        
        # 保存句子
        with open(sentences_file, 'wb') as f:
            pickle.dump(sentences, f)
    
    print(f"共 {len(sentences)} 个句子用于训练")
    
    # 优化的Word2Vec参数
    print("训练Word2Vec模型...")
    num_features = 400    # 增加向量维度
    min_word_count = 20    # 降低最小词频
    num_workers = 4
    context = 15            # 增加上下文窗口
    downsampling = 1e-3
    
    model = Word2Vec(sentences, workers=num_workers,
                   vector_size=num_features, min_count=min_word_count,
                   window=context, sample=downsampling, sg=1)  # 使用skip-gram
    
    model.save(model_file)
    print(f"模型已保存到 {model_file}")
    
    return model

def load_or_create_embeddings(model, data_dir, cache_dir="cache"):
    """加载或创建embeddings"""
    train_emb_file = os.path.join(cache_dir, "train_embeddings.npy")
    test_emb_file = os.path.join(cache_dir, "test_embeddings.npy")
    train_clean_file = os.path.join(cache_dir, "clean_train.pkl")
    test_clean_file = os.path.join(cache_dir, "clean_test.pkl")
    sentiment_file = os.path.join(cache_dir, "sentiment.npy")
    test_ids_file = os.path.join(cache_dir, "test_ids.pkl")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 如果embeddings已存在，直接加载
    if os.path.exists(train_emb_file) and os.path.exists(test_emb_file):
        print("加载已有的embeddings...")
        trainDataVecs = np.load(train_emb_file)
        testDataVecs = np.load(test_emb_file)
        train_sentiment = np.load(sentiment_file)
        with open(test_ids_file, 'rb') as f:
            test_ids = pickle.load(f)
        return trainDataVecs, testDataVecs, train_sentiment, test_ids
    
    # 读取数据
    print("读取数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    
    # 清洗评论
    print("清洗训练评论...")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(clean_review(review, remove_stopwords=False))  # 不移除停用词
    
    print("清洗测试评论...")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(clean_review(review, remove_stopwords=False))
    
    # 保存清洗后的评论
    with open(train_clean_file, 'wb') as f:
        pickle.dump(clean_train_reviews, f)
    with open(test_clean_file, 'wb') as f:
        pickle.dump(clean_test_reviews, f)
    
    # 创建embeddings
    num_features = model.vector_size
    print("创建训练集embeddings...")
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    
    print("创建测试集embeddings...")
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
    
    # 保存embeddings
    np.save(train_emb_file, trainDataVecs)
    np.save(test_emb_file, testDataVecs)
    np.save(sentiment_file, train["sentiment"].values)
    with open(test_ids_file, 'wb') as f:
        pickle.dump(test["id"].values)
    
    return trainDataVecs, testDataVecs, train["sentiment"].values, test["id"].values

def main():
    print("=== 优化的情感分析 (Word2Vec + 逻辑回归) ===")
    
    data_dir = "word2vec-nlp-tutorial"
    cache_dir = "cache"
    
    # Step 1: 训练或加载Word2Vec模型
    print("\n--- Step 1: Word2Vec模型 ---")
    model = load_or_train_word2vec(data_dir, cache_dir)
    num_features = model.vector_size
    print(f"词向量维度: {num_features}")
    
    # 测试模型
    try:
        print(f"'woman' 最相似的词: {model.wv.most_similar('woman')[:3]}")
        print(f"'king' - 'man' + 'woman' = {model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[0]}")
    except Exception as e:
        print(f"模型测试跳过: {e}")
    
    # Step 2: 创建或加载embeddings
    print("\n--- Step 2: 创建Embeddings ---")
    trainDataVecs, testDataVecs, train_sentiment, test_ids = load_or_create_embeddings(model, data_dir, cache_dir)
    
    # Step 3: 标准化特征
    print("\n--- Step 3: 标准化特征 ---")
    scaler = StandardScaler()
    trainDataVecs_scaled = scaler.fit_transform(trainDataVecs)
    testDataVecs_scaled = scaler.transform(testDataVecs)
    
    # Step 4: 训练逻辑回归
    print("\n--- Step 4: 训练逻辑回归 ---")
    print("使用交叉验证评估模型...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    # 交叉验证
    cv_scores = cross_val_score(lr, trainDataVecs_scaled, train_sentiment, cv=5, scoring='roc_auc')
    print(f"交叉验证 ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 在全部数据上训练
    print("在全部训练数据上训练...")
    lr.fit(trainDataVecs_scaled, train_sentiment)
    
    # Step 5: 预测
    print("\n--- Step 5: 预测测试数据 ---")
    result = lr.predict(testDataVecs_scaled)
    result_proba = lr.predict_proba(testDataVecs_scaled)[:, 1]  # 获取正面情感的概率
    
    # 保存结果
    print("\n--- Step 6: 保存结果 ---")
    output = pd.DataFrame(data={"id": test_ids, "sentiment": result})
    output.to_csv("Optimized_Word2Vec_LogisticRegression.csv", index=False, quoting=3)
    
    # 也保存概率版本（可能更好）
    output_proba = pd.DataFrame(data={"id": test_ids, "sentiment": (result_proba >= 0.5).astype(int)})
    output_proba.to_csv("Optimized_Word2Vec_LogisticRegression_Proba.csv", index=False, quoting=3)
    
    print("\n=== 完成！ ===")
    print("生成的文件:")
    print("  - Optimized_Word2Vec_LogisticRegression.csv (使用类别预测)")
    print("  - Optimized_Word2Vec_LogisticRegression_Proba.csv (使用概率阈值0.5)")

if __name__ == "__main__":
    main()
