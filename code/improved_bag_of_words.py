import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import os

def review_to_words(raw_review):
    """
    优化后的文本预处理：
    1. 移除HTML标签
    2. 保留重要标点（感叹号、问号）用于情感
    3. 处理否定词（不删除！）
    4. 小写化
    """
    # 1. 移除HTML标签
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    
    # 2. 处理标点：保留!和?，其他非字母字符替换为空格
    # 先把!和?替换成特殊标记，最后再换回来
    review_text = review_text.replace('!', ' EXCLAMATION ')
    review_text = review_text.replace('?', ' QUESTION ')
    
    # 3. 移除非字母字符（但保留了上面的标记）
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # 4. 转换为小写
    words = letters_only.lower().split()
    
    # 5. 处理特殊标记
    processed_words = []
    for word in words:
        if word == 'exclamation':
            processed_words.append('!')
        elif word == 'question':
            processed_words.append('?')
        else:
            processed_words.append(word)
    
    # 6. 移除停用词，但保留否定词！
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
    # 注意：我们从停用词中移除了 'no', 'nor', 'not', 'don' 等否定词！
    
    meaningful_words = [w for w in processed_words if not w in stops]
    
    return(" ".join(meaningful_words))

def main():
    print("=== 改进的 Bag of Words + TF-IDF + 逻辑回归 ===")
    print("(保留否定词，保留情感标点)")
    
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
    
    # 创建TF-IDF特征
    print("\n3. 创建TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=20000,  # 增加词汇量
        ngram_range=(1, 2),  # 使用uni-gram和bi-gram
        sublinear_tf=True     # 亚线性TF缩放
    )
    
    train_data_features = tfidf_vectorizer.fit_transform(clean_train_reviews)
    
    print(f"   训练特征形状: {train_data_features.shape}")
    print(f"   词汇表大小: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    # 训练逻辑回归
    print("\n4. 训练逻辑回归分类器...")
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    # 交叉验证评估
    print("   进行5折交叉验证...")
    cv_scores = cross_val_score(lr, train_data_features, train["sentiment"], cv=5, scoring='roc_auc')
    print(f"   交叉验证 ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # 在全部数据上训练
    print("   在全部训练数据上训练...")
    lr.fit(train_data_features, train["sentiment"])
    
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
    test_data_features = tfidf_vectorizer.transform(clean_test_reviews)
    
    # 预测
    print("\n6. 预测测试数据...")
    result = lr.predict(test_data_features)
    result_proba = lr.predict_proba(test_data_features)[:, 1]
    
    # 创建提交文件
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Improved_Bag_of_Words_TFIDF.csv", index=False, quoting=3)
    
    # 也保存概率版本（可能更好）
    output_proba = pd.DataFrame(data={"id": test["id"], "sentiment": (result_proba >= 0.5).astype(int)})
    output_proba.to_csv("Improved_Bag_of_Words_TFIDF_Proba.csv", index=False, quoting=3)
    
    print("\n=== 完成！ ===")
    print("生成的文件:")
    print("  - Improved_Bag_of_Words_TFIDF.csv (类别预测)")
    print("  - Improved_Bag_of_Words_TFIDF_Proba.csv (概率阈值0.5)")

if __name__ == "__main__":
    main()
