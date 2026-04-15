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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def clean_review_for_word2vec(raw_review):
    """
    专为Word2Vec优化的清洗：
    - 移除HTML标签
    - 保留否定词
    - 保留所有单词（不删除停用词，Word2Vec需要上下文）
    - 小写化
    """
    # 1. 移除HTML标签
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    
    # 2. 移除非字母字符，但保留基本的单词结构
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 3. 转换为小写并分割
    words = letters_only.lower().split()
    
    # 4. 不删除任何词！Word2Vec需要上下文信息
    return words

def review_to_sentences(raw_review):
    """将评论分割为句子列表"""
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    review_text = re.sub("[^a-zA-Z.!?]", " ", review_text)
    review_text = review_text.lower()
    raw_sentences = re.split('[.!?]', review_text)
    sentences = []
    for sent in raw_sentences:
        words = sent.strip().split()
        if len(words) > 0:
            sentences.append(words)
    return sentences

def load_or_train_word2vec(data_dir, cache_dir="cache_final"):
    """加载或训练Word2Vec模型，并缓存结果"""
    model_file = os.path.join(cache_dir, "word2vec_final.model")
    sentences_file = os.path.join(cache_dir, "sentences_final.pkl")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if os.path.exists(model_file):
        print("加载已有的Word2Vec模型...")
        model = Word2Vec.load(model_file)
        return model
    
    if os.path.exists(sentences_file):
        print("加载已缓存的句子...")
        with open(sentences_file, 'rb') as f:
            sentences = pickle.load(f)
    else:
        print("准备训练数据...")
        train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                            header=0, delimiter="\t", quoting=3)
        unlabeled_train = pd.read_csv(os.path.join(data_dir, "unlabeledTrainData.tsv", "unlabeledTrainData.tsv"), 
                                       header=0, delimiter="\t", quoting=3)
        
        sentences = []
        print("   处理标注训练数据...")
        for review in train["review"]:
            sentences.extend(review_to_sentences(review))
        
        print("   处理未标注训练数据...")
        for review in unlabeled_train["review"]:
            sentences.extend(review_to_sentences(review))
        
        with open(sentences_file, 'wb') as f:
            pickle.dump(sentences, f)
    
    print(f"共 {len(sentences)} 个句子用于训练")
    
    print("训练Word2Vec模型...")
    num_features = 400
    min_word_count = 10
    num_workers = 4
    context = 15
    downsampling = 1e-3
    
    model = Word2Vec(sentences, workers=num_workers,
                   vector_size=num_features, min_count=min_word_count,
                   window=context, sample=downsampling, sg=1)
    
    model.save(model_file)
    print(f"模型已保存到 {model_file}")
    
    return model

def get_avg_feature_vector(words, model, num_features):
    """获取评论的平均词向量"""
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    vocab_set = set(model.wv.index_to_key)
    
    for word in words:
        if word in vocab_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def load_or_create_embeddings(model, data_dir, cache_dir="cache_final"):
    """加载或创建embeddings"""
    train_emb_file = os.path.join(cache_dir, "train_embeddings.npy")
    test_emb_file = os.path.join(cache_dir, "test_embeddings.npy")
    sentiment_file = os.path.join(cache_dir, "sentiment.npy")
    test_ids_file = os.path.join(cache_dir, "test_ids.pkl")
    
    if os.path.exists(train_emb_file) and os.path.exists(test_emb_file):
        print("加载已有的embeddings...")
        train_vecs = np.load(train_emb_file)
        test_vecs = np.load(test_emb_file)
        sentiment = np.load(sentiment_file)
        with open(test_ids_file, 'rb') as f:
            test_ids = pickle.load(f)
        return train_vecs, test_vecs, sentiment, test_ids
    
    print("读取数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    
    num_features = model.vector_size
    
    print("处理训练评论...")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(clean_review_for_word2vec(review))
    
    print("处理测试评论...")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(clean_review_for_word2vec(review))
    
    print("创建训练集embeddings...")
    train_vecs = np.zeros((len(clean_train_reviews), num_features), dtype="float32")
    for i, review in enumerate(clean_train_reviews):
        if i % 1000 == 0:
            print(f"   进度: {i}/{len(clean_train_reviews)}")
        train_vecs[i] = get_avg_feature_vector(review, model, num_features)
    
    print("创建测试集embeddings...")
    test_vecs = np.zeros((len(clean_test_reviews), num_features), dtype="float32")
    for i, review in enumerate(clean_test_reviews):
        if i % 1000 == 0:
            print(f"   进度: {i}/{len(clean_test_reviews)}")
        test_vecs[i] = get_avg_feature_vector(review, model, num_features)
    
    np.save(train_emb_file, train_vecs)
    np.save(test_emb_file, test_vecs)
    np.save(sentiment_file, train["sentiment"].values)
    with open(test_ids_file, 'wb') as f:
        pickle.dump(test["id"].values)
    
    return train_vecs, test_vecs, train["sentiment"].values, test["id"].values

def main():
    print("=" * 60)
    print("最终优化方案：Word2Vec + 均值Embedding + 逻辑回归")
    print("=" * 60)
    
    data_dir = "word2vec-nlp-tutorial"
    cache_dir = "cache_final"
    
    print("\n--- 第1步：Word2Vec模型 ---")
    model = load_or_train_word2vec(data_dir, cache_dir)
    num_features = model.vector_size
    print(f"词向量维度: {num_features}")
    
    try:
        print(f"\n模型测试：")
        print(f"   'woman' 最相似: {[w[0] for w in model.wv.most_similar('woman')[:3]]}")
        print(f"   'king' - 'man' + 'woman' = {model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[0][0]}")
    except Exception as e:
        print(f"   模型测试跳过: {e}")
    
    print("\n--- 第2步：生成Embeddings ---")
    train_vecs, test_vecs, sentiment, test_ids = load_or_create_embeddings(model, data_dir, cache_dir)
    
    print("\n--- 第3步：标准化特征 ---")
    scaler = StandardScaler()
    train_vecs_scaled = scaler.fit_transform(train_vecs)
    test_vecs_scaled = scaler.transform(test_vecs)
    
    print("\n--- 第4步：训练逻辑回归 ---")
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    
    print("5折交叉验证...")
    cv_scores = cross_val_score(lr, train_vecs_scaled, sentiment, cv=5, scoring='roc_auc')
    print(f"交叉验证 ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    print("在全部数据上训练...")
    lr.fit(train_vecs_scaled, sentiment)
    
    print("\n--- 第5步：预测 ---")
    result = lr.predict(test_vecs_scaled)
    result_proba = lr.predict_proba(test_vecs_scaled)[:, 1]
    
    print("\n--- 第6步：保存结果 ---")
    output1 = pd.DataFrame(data={"id": test_ids, "sentiment": result})
    output1.to_csv("Final_Word2Vec_LogisticRegression.csv", index=False, quoting=3)
    
    output2 = pd.DataFrame(data={"id": test_ids, "sentiment": (result_proba >= 0.5).astype(int)})
    output2.to_csv("Final_Word2Vec_LogisticRegression_Proba.csv", index=False, quoting=3)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n生成的文件：")
    print(f"  - Final_Word2Vec_LogisticRegression.csv")
    print(f"  - Final_Word2Vec_LogisticRegression_Proba.csv")
    print(f"\n缓存目录：{cache_dir}/ (下次运行将直接加载，无需重新训练！)")

if __name__ == "__main__":
    main()
