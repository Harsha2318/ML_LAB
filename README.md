# ML_LAB

This repository contains 10 lab programs demonstrating various machine learning concepts and techniques. Below is a summary explanation, viva questions, and answers for each lab.

---

## Lab 1: Exploratory Data Analysis on California Housing Dataset

**Explanation:**  
This lab loads the California housing dataset and performs exploratory data analysis (EDA). It visualizes the distribution of numerical features using histograms and boxplots to understand data spread and detect outliers. The interquartile range (IQR) method is used to identify outliers for each feature. Finally, a statistical summary of the dataset is printed.

**Viva Questions and Answers:**  
1. What is the purpose of exploratory data analysis?  
   - To understand the main characteristics of the data, detect patterns, spot anomalies, and check assumptions.  
2. How do histograms and boxplots help in understanding data?  
   - Histograms show the distribution and frequency of data values; boxplots highlight the spread, median, and outliers.  
3. Explain the IQR method for detecting outliers.  
   - Outliers are values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where IQR = Q3 - Q1.  
4. Why is it important to detect outliers before modeling?  
   - Outliers can skew model training and affect performance, so detecting them helps in data cleaning.

---

## Lab 2: Correlation and Pairwise Relationships in California Housing Dataset

**Explanation:**  
This lab computes the correlation matrix of the California housing dataset features and visualizes it using a heatmap. It also creates a pair plot to visualize pairwise relationships and distributions of features, helping to identify feature dependencies and potential multicollinearity.

**Viva Questions and Answers:**  
1. What does a correlation matrix represent?  
   - It shows the strength and direction of linear relationships between pairs of variables.  
2. How can a heatmap help in visualizing correlations?  
   - It uses color gradients to represent correlation values, making patterns easier to spot.  
3. What insights can be gained from a pair plot?  
   - It reveals relationships, clusters, and distributions between pairs of features.  
4. Why is understanding feature correlation important in machine learning?  
   - Highly correlated features can cause multicollinearity, affecting model stability and interpretation.

---

## Lab 3: Principal Component Analysis (PCA) on Iris Dataset

**Explanation:**  
This lab applies PCA to the Iris dataset to reduce its dimensionality from 4 to 2 principal components. It visualizes the reduced data in a 2D scatter plot, showing how PCA can help in visualizing high-dimensional data and possibly improving model performance by reducing noise.

**Viva Questions and Answers:**  
1. What is the goal of PCA?  
   - To reduce dimensionality while preserving as much variance as possible.  
2. How does PCA reduce dimensionality?  
   - By projecting data onto orthogonal principal components ordered by variance.  
3. What are principal components?  
   - New variables that are linear combinations of original features capturing maximum variance.  
4. How can PCA help in data visualization and modeling?  
   - By simplifying data structure and reducing noise, making patterns clearer.

---

## Lab 4: Find-S Algorithm for Concept Learning

**Explanation:**  
This lab implements the Find-S algorithm, a simple concept learning algorithm. It reads training data from a CSV file and iteratively updates the hypothesis to find the most specific hypothesis consistent with all positive examples.

**Viva Questions and Answers:**  
1. What is the Find-S algorithm?  
   - A method to find the most specific hypothesis that fits all positive training examples.  
2. How does Find-S update its hypothesis?  
   - By generalizing attributes only when necessary to accommodate positive examples.  
3. What are the limitations of the Find-S algorithm?  
   - It only learns conjunctive concepts and ignores negative examples.  
4. What is the difference between specific and general hypotheses?  
   - Specific hypotheses are restrictive; general hypotheses cover more instances.

---

## Lab 5: k-Nearest Neighbors (k-NN) Classification on Synthetic Data

**Explanation:**  
This lab implements a simple k-NN classifier on synthetic 1D data. It classifies test points for various values of k, prints some classification results, and visualizes the training and test points with their predicted labels.

**Viva Questions and Answers:**  
1. How does the k-NN algorithm work?  
   - It classifies a point based on the majority class among its k nearest neighbors.  
2. What is the effect of different values of k on classification?  
   - Smaller k can lead to noisy decisions; larger k smooths boundaries but may miss details.  
3. How is distance calculated in k-NN?  
   - Typically using Euclidean distance or absolute difference in 1D.  
4. What are the advantages and disadvantages of k-NN?  
   - Advantages: simple, effective; Disadvantages: computationally expensive, sensitive to irrelevant features.

---

## Lab 6: Locally Weighted Linear Regression (LWLR)

**Explanation:**  
This lab implements LWLR, a non-parametric regression method that fits a linear model weighted by proximity to the query point. It uses a Gaussian kernel to assign weights and fits the model locally to predict values on noisy sine wave data.

**Viva Questions and Answers:**  
1. What is locally weighted linear regression?  
   - A regression method that fits models weighted by closeness to the query point.  
2. How are weights assigned in LWLR?  
   - Using a kernel function, typically Gaussian, based on distance.  
3. What is the role of the bandwidth parameter (tau)?  
   - Controls the width of the kernel, affecting locality of fitting.  
4. How does LWLR differ from ordinary linear regression?  
   - LWLR fits a model for each query point locally, while ordinary fits one global model.

---

## Lab 7: Linear and Polynomial Regression

**Explanation:**  
This lab demonstrates linear regression on the California housing dataset and polynomial regression (degree 2) on the Auto MPG dataset. It splits data into training and testing sets, fits models, and visualizes predictions against actual values.

**Viva Questions and Answers:**  
1. What is the difference between linear and polynomial regression?  
   - Linear fits a straight line; polynomial fits curves by including powers of features.  
2. How does polynomial regression model non-linear relationships?  
   - By transforming features into polynomial terms.  
3. Why is train-test splitting important?  
   - To evaluate model generalization on unseen data.  
4. How do you evaluate regression model performance?  
   - Using metrics like MSE, RMSE, R-squared.

---

## Lab 8: Decision Tree Classifier on Breast Cancer Dataset

**Explanation:**  
This lab trains a decision tree classifier on the breast cancer dataset. It evaluates accuracy on the test set, predicts the class of a new sample, and visualizes the decision tree structure.

**Viva Questions and Answers:**  
1. How does a decision tree classifier work?  
   - It splits data based on feature values to create a tree of decisions.  
2. What criteria are used to split nodes in a decision tree?  
   - Metrics like Gini impurity or information gain.  
3. How can decision trees be visualized?  
   - Using tree plots showing splits and leaf nodes.  
4. What are the advantages and limitations of decision trees?  
   - Advantages: interpretable, handles non-linear data; Limitations: prone to overfitting.

---

## Lab 9: Gaussian Naive Bayes Classification on Olivetti Faces Dataset

**Explanation:**  
This lab applies Gaussian Naive Bayes classification to the Olivetti faces dataset. It trains the model, evaluates accuracy, and visualizes some test images with their true and predicted labels.

**Viva Questions and Answers:**  
1. What is the Naive Bayes classifier?  
   - A probabilistic classifier based on Bayes theorem with independence assumptions.  
2. Why is it called "naive"?  
   - Because it assumes feature independence.  
3. How does Gaussian Naive Bayes handle continuous features?  
   - By modeling them with Gaussian distributions.  
4. What are the assumptions behind Naive Bayes?  
   - Features are conditionally independent given the class.

---

## Lab 10: KMeans Clustering on Breast Cancer Dataset

**Explanation:**  
This lab applies KMeans clustering to the breast cancer dataset after standardizing features. It compares cluster labels to true labels using confusion matrix and classification report, performs PCA for visualization, and plots clusters and cluster centers.

**Viva Questions and Answers:**  
1. What is KMeans clustering?  
   - An unsupervised algorithm that partitions data into k clusters by minimizing within-cluster variance.  
2. How does KMeans assign cluster labels?  
   - By assigning points to the nearest cluster centroid iteratively.  
3. What is the role of PCA in clustering visualization?  
   - To reduce dimensionality for 2D plotting of clusters.  
4. How do you evaluate clustering performance?  
   - Using metrics like silhouette score, or comparing to true labels if available.

---
