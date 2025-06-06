import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
data = np.random.rand(100)
train_x, test_x = data[:50], data[50:]
train_y = np.array(['Class1' if x <= 0.5 else 'Class2' for x in train_x])

def knn_classifier(test_point, k):
    distances = np.abs(train_x - test_point)
    unique_labels, counts = np.unique(train_y[np.argsort(distances)[:k]], return_counts=True)
    label_counts = list(zip(counts, unique_labels))
    return max(label_counts)[1]

print("=== k-NN Classification Results ===")
for k in [1, 2, 3, 4, 5, 20, 30]:
    predictions = [knn_classifier(x, k) for x in test_x]
    
    # Print first 3 results for demonstration
    print(f"\nk={k}:")
    for i in range(3):
        print(f"x{51+i}: {test_x[i]:.3f} → {predictions[i]}")
    
    # Visualize results
    plt.figure(figsize=(8, 3))
    plt.scatter(train_x, [0]*50, c=['C0' if c=='Class1' else 'C1' for c in train_y], label='Train')
    plt.scatter(test_x, [1]*50, c=['C0' if c=='Class1' else 'C1' for c in predictions], marker='X', label='Test')
    plt.title(f"k-NN (k={k})"), plt.yticks([0,1], ['Train', 'Test']), plt.legend()
    plt.show()
