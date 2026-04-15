import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    review_text = review_text.replace('!', ' EXCLAMATION ')
    review_text = review_text.replace('?', ' QUESTION ')
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    
    processed_words = []
    for word in words:
        if word == 'exclamation':
            processed_words.append('!')
        elif word == 'question':
            processed_words.append('?')
        else:
            processed_words.append(word)
    
    stops = {
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
        'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should', 'now'
    }
    
    meaningful_words = [w for w in processed_words if not w in stops]
    return(" ".join(meaningful_words))

def main():
    print("=" * 60)
    print("提交预测概率版本（适用于ROC AUC）")
    print("=" * 60)
    
    data_dir = "word2vec-nlp-tutorial"
    
    print("\n1. 读取训练数据...")
    train = pd.read_csv(os.path.join(data_dir, "labeledTrainData.tsv", "labeledTrainData.tsv"), 
                        header=0, delimiter="\t", quoting=3)
    
    print("2. 清理训练数据...")
    num_reviews = train["review"].size
    clean_train_reviews = []
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print(f"   已处理 {i+1} / {num_reviews} 条评论")
        clean_train_reviews.append(review_to_words(train["review"][i]))
    
    print("\n3. 创建TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    train_data_features = tfidf_vectorizer.fit_transform(clean_train_reviews)
    print(f"   特征形状: {train_data_features.shape}")
    
    print("\n4. 训练逻辑回归...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(train_data_features, train["sentiment"])
    
    print("\n5. 读取和处理测试数据...")
    test = pd.read_csv(os.path.join(data_dir, "testData.tsv", "testData.tsv"), 
                       header=0, delimiter="\t", quoting=3)
    
    num_reviews = len(test["review"])
    clean_test_reviews = []
    for i in range(0, num_reviews):
        if (i+1) % 1000 == 0:
            print(f"   已处理 {i+1} / {num_reviews} 条评论")
        clean_test_reviews.append(review_to_words(test["review"][i]))
    
    test_data_features = tfidf_vectorizer.transform(clean_test_reviews)
    
    print("\n6. 预测（输出概率而非二分类结果）...")
    result_proba = lr.predict_proba(test_data_features)[:, 1]
    
    print("\n7. 保存提交文件（包含概率）...")
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result_proba})
    output.to_csv("Submission_With_Probabilities.csv", index=False, quoting=3)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n📁 生成的文件: Submission_With_Probabilities.csv")
    print("💡 这个文件包含预测概率，适用于ROC AUC评估！")

if __name__ == "__main__":
    main()
