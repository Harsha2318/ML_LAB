import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)

kmeans = KMeans(2, random_state=42).fit(X)
labels = kmeans.labels_

print(confusion_matrix(data.target, labels))
print(classification_report(data.target, labels))

pca = PCA(2).fit_transform(X)
df = pd.DataFrame(pca, columns=['PC1', 'PC2'])
df['Cluster'] = labels

sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='Set1', s=100, edgecolor='k')
plt.scatter(*PCA(2).fit_transform(kmeans.cluster_centers_).T, c='red', marker='X', s=200)
plt.show()
