import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 60)
print("展示逻辑回归如何输出真实预测概率")
print("=" * 60)

# 简单示例数据
X_train_simple = [
    "this movie is great !",
    "i love this film",
    "best movie ever",
    "terrible movie",
    "hate this film",
    "worst ever"
]
y_train_simple = [1, 1, 1, 0, 0, 0]

print("\n1. 简单训练数据:")
for text, label in zip(X_train_simple, y_train_simple):
    print(f"   '{text}' → {label} ({'正面' if label==1 else '负面'})")

# 训练简单模型
print("\n2. 训练TF-IDF + 逻辑回归...")
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_simple)

lr = LogisticRegression()
lr.fit(X_train_vec, y_train_simple)

# 测试预测
X_test_simple = [
    "this is a great movie",
    "terrible film",
    "okay movie not great not bad"
]

X_test_vec = vectorizer.transform(X_test_simple)

print("\n3. 预测结果:")
predictions = lr.predict(X_test_vec)
probabilities = lr.predict_proba(X_test_vec)

print("\n   predict() 输出的是二分类结果:")
for text, pred in zip(X_test_simple, predictions):
    print(f"   '{text}' → {pred}")

print("\n   predict_proba() 输出的是真实概率:")
for i, text in enumerate(X_test_simple):
    prob_negative = probabilities[i][0]
    prob_positive = probabilities[i][1]
    print(f"   '{text}'")
    print(f"      → 负面概率: {prob_negative:.4f}")
    print(f"      → 正面概率: {prob_positive:.4f}")
    print(f"      → 两者相加: {prob_negative + prob_positive:.4f}")
    print()

print("=" * 60)
print("说明:")
print("- predict() 返回的是 0 或 1（二分类）")
print("- predict_proba() 返回的是 [负面概率, 正面概率]")
print("- 两个概率相加总是等于 1.0")
print("- 我们提交的是正面概率那一列！")
print("=" * 60)
