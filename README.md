Project Objective:
This project focuses on developing a fashion recommendation system that suggests clothing colors based on user interactions, preferences, and product attributes. By leveraging machine learning (ML) and collaborative filtering techniques, it predicts colors users are likely to prefer. The system is divided into three branches, each using a different recommendation methodology—memory-based filtering, user-item collaborative filtering, and item-based filtering—to enhance accuracy and performance.
Branch 1: Memory-Based Recommendation with LightGBM
1. Data Preparation:
• We perform StratifiedKFold or train_test_split to create balanced splits.
• ADASYN oversampling is employed to augment minority classes, tackling skewed distributions in target labels.
2. Modeling with LightGBM:
• LightGBMClassifier leverages gradient-boosted decision trees.
• RandomizedSearchCV and Hyperparameter Tuning aim to find optimal parameters (e.g., max_depth, learning_rate, n_estimators).
3. Evaluation:
• Metrics include ROC AUC, Confusion Matrix, Precision/Recall/F1 to assess classification performance.
• CatBoostClassifier can also be tested for comparison to handle categorical features natively.

Branch 2: Collaborative Filtering (CF) with LightGBM and SVD
1. User–Item Interaction & Imbalance Handling:
• SMOTE or ADASYN oversampling addresses dataset imbalance in user–item interactions.
• Cosine similarity / matrix factorization refine our user-based or item-based relationships.
2. LightGBM for Interaction Prediction:
• Model features include user, item (colour), and content-based attributes.
• Predictive tasks might classify user interest or potential rating to produce recommendations.
3. Singular Value Decomposition (SVD):
• SVD decomposes large user–item matrices into lower-dimensional latent factors.
• Combined with CF, it reveals underlying user preferences and item characteristics.
• svds from SciPy is used to factorize the sparse matrix.
4. Evaluation:
• Accuracy metrics: RMSE, MAE, AUC for rating-based predictions.
• Precision–Recall curves for classification tasks, if rating thresholds are used.

Branch 3: Item-Based CF Using KNN
1. Item Similarity:
• We compute the K-Nearest Neighbors among items to find the most similar ones based on rating vectors or other content-based features.
• The mean_squared_error and mean_absolute_error gauge numeric prediction accuracy.
2. Data Flow:
• Preprocessed data (e.g., scaled, encoded).
• Sparse matrices handle item-based rating vectors efficiently for fast neighbor lookups.
3. KNN Approach:
• The model identifies top-k similar items for any given item, recommending them to users who liked the reference item.
4. Evaluation:
• MSE, RMSE, and MAE measure rating prediction quality.
• Visualization with Seaborn or Matplotlib to interpret residuals and distribution of predicted vs. actual ratings.
