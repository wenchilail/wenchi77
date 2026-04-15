# Bag of Words Meets Bags of Popcorn - Kaggle Competition

IMDB电影评论情感分析比赛项目

## 项目结构

```
.
├── code/              # 实验代码
├── report/            # 实验报告和文档
├── results/           # 实验结果（提交文件、日志等）
├── cache/             # 缓存目录（不提交到git）
├── cache_final/       # Word2Vec缓存目录（不提交到git）
├── word2vec-nlp-tutorial/  # 原始数据（不提交到git）
└── README.md          # 项目说明
```

## 实验进展

### 实验1: 原始Bag of Words + 随机森林
- **文件**: `code/part1_bag_of_words.py`
- **提交文件**: `results/Bag_of_Words_model.csv`
- **问题**: 停用词包含否定词，伤害情感分析
- **分数**: ~0.84-0.86 ❌

### 实验2: 改进的TF-IDF + 逻辑回归（二分类结果）
- **文件**: `code/improved_bag_of_words.py`
- **关键改进**:
  - 从停用词中移除否定词（not, no, never等）
  - 保留情感标点（! , ?）
  - 使用TF-IDF替代简单词频
  - 使用逻辑回归替代随机森林
  - 加入n-gram特征
- **提交文件**: `results/Improved_Bag_of_Words_TFIDF_Proba.csv`
- **交叉验证**: 0.9616 ✅
- **问题**: 提交的是二分类结果（0/1），不是概率

### 实验3: 改进的TF-IDF + 逻辑回归（预测概率）⭐推荐
- **文件**: `code/submission_with_probabilities.py`
- **关键改进**:
  - 提交预测概率而不是二分类结果
  - 适用于ROC AUC评估
- **提交文件**: `results/Submission_With_Probabilities.csv`
- **推荐使用**: ✅ 这个应该能获得最好的分数！

### 实验4: Word2Vec + 均值Embedding + 逻辑回归
- **文件**: `code/final_optimized_solution.py`
- **提交文件**: `results/Final_Word2Vec_LogisticRegression_Proba.csv`

## 使用说明

### 推荐方案
直接使用 `results/Submission_With_Probabilities.csv` 提交到Kaggle！

### 重新运行实验
```bash
# 进入code目录
cd code

# 运行推荐方案
python submission_with_probabilities.py
```

## 注意事项

- 不要将 `word2vec-nlp-tutorial/`、`cache/`、`cache_final/` 目录提交到git
- 不要提交大模型文件
- 每次实验后及时提交代码、报告和结果
