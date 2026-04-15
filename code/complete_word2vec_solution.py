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

def clean_review_for_word2vec(raw_review):
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    return words

def get_avg_feature_vector(words, model, num_features):
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

def main():
    print("=" * 60)
    print("快速使用已训练的Word2Vec模型")
    print("=" * 60)
    
    data_dir = "word2vec-nlp-tutorial"
    cache_dir = "cache_final"
    
    print("\n1. 加载已训练的Word2Vec模型...")
    model = Word2Vec.load(os.path.join(cache_dir, "word2vec_final.model"))
    num_features = model.vector_size
    print(f"   词向量维度: {num_features}")
    print(f"   词汇表大小: {len(model.wv.index_to_key)}")
    
    try:
        print(f"\n2. 模型测试:")
        print(f"   'woman' 最相似: {[w[0] for w in model.wv.most_similar('woman')[:3]]}")
        print(f"   'king' - 'man' + 'woman' = {model.wv.most_similar(positive=['king', 'woman'], negative=['man'])[0][0]}")
    except Exception as e:
        print(f"   模型测试跳过: {e}")
    
    print("\n3. 读取数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    
    print("\n4. 处理评论并生成embeddings...")
    
    print("   处理训练集...")
    clean_train_reviews = []
    for i, review in enumerate(train["review"]):
        if i % 1000 == 0:
            print(f"      进度: {i}/{len(train)}")
        clean_train_reviews.append(clean_review_for_word2vec(review))
    
    print("   处理测试集...")
    clean_test_reviews = []
    for i, review in enumerate(test["review"]):
        if i % 1000 == 0:
            print(f"      进度: {i}/{len(test)}")
        clean_test_reviews.append(clean_review_for_word2vec(review))
    
    print("   生成训练集embeddings...")
    train_vecs = np.zeros((len(clean_train_reviews), num_features), dtype="float32")
    for i, review in enumerate(clean_train_reviews):
        if i % 1000 == 0:
            print(f"      进度: {i}/{len(clean_train_reviews)}")
        train_vecs[i] = get_avg_feature_vector(review, model, num_features)
    
    print("   生成测试集embeddings...")
    test_vecs = np.zeros((len(clean_test_reviews), num_features), dtype="float32")
    for i, review in enumerate(clean_test_reviews):
        if i % 1000 == 0:
            print(f"      进度: {i}/{len(clean_test_reviews)}")
        test_vecs[i] = get_avg_feature_vector(review, model, num_features)
    
    print("\n5. 标准化特征...")
    scaler = StandardScaler()
    train_vecs_scaled = scaler.fit_transform(train_vecs)
    test_vecs_scaled = scaler.transform(test_vecs)
    
    print("\n6. 训练逻辑回归...")
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    
    print("   5折交叉验证...")
    cv_scores = cross_val_score(lr, train_vecs_scaled, train["sentiment"], cv=5, scoring='roc_auc')
    print(f"   交叉验证 ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    print("   在全部数据上训练...")
    lr.fit(train_vecs_scaled, train["sentiment"])
    
    print("\n7. 预测...")
    result = lr.predict(test_vecs_scaled)
    result_proba = lr.predict_proba(test_vecs_scaled)[:, 1]
    
    print("\n8. 保存结果...")
    output1 = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output1.to_csv("Final_Word2Vec_LogisticRegression.csv", index=False, quoting=3)
    
    output2 = pd.DataFrame(data={"id": test["id"], "sentiment": (result_proba >= 0.5).astype(int)})
    output2.to_csv("Final_Word2Vec_LogisticRegression_Proba.csv", index=False, quoting=3)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n可提交的文件:")
    print(f"  1. Improved_Bag_of_Words_TFIDF_Proba.csv (TF-IDF方案，交叉验证0.9616)")
    print(f"  2. Final_Word2Vec_LogisticRegression_Proba.csv (Word2Vec方案)")
    print(f"\n推荐使用方案1，因为交叉验证分数更高！")

if __name__ == "__main__":
    main()
