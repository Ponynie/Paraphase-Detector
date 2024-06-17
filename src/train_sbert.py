import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Create the results directory
results_dir = 'results/train_test_results_sbert'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load Sentence BERT model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def get_sentence_vector(sentence):
    # Encode the single sentence and return the first (and only) embedding
    return model.encode([sentence])[0]

def create_combined_vector(s1, s2, method='concatenation'):
    vec1 = get_sentence_vector(s1)
    vec2 = get_sentence_vector(s2)
    
    if method == 'concatenation':
        return np.concatenate((vec1, vec2))
    elif method == 'mean':
        return (vec1 + vec2) / 2
    elif method == 'max_pooling':
        return np.maximum(vec1, vec2)
    else:
        raise ValueError("Method not supported")

# Preprocess the data to fix issues and convert to DataFrame
def preprocess_data(filepath):
    clean_data = {'Quality': [], '#1 ID': [], '#2 ID': [], '#1 String': [], '#2 String': []}
    with open(filepath, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                clean_data['Quality'].append(parts[0])
                clean_data['#1 ID'].append(parts[1])
                clean_data['#2 ID'].append(parts[2])
                clean_data['#1 String'].append(parts[3])
                clean_data['#2 String'].append(parts[4])
            else:
                print(f"Skipping malformed line: {line}")
    return pd.DataFrame(clean_data)

# Preprocess and load the training data
train_data = preprocess_data('JAIST-intern-data/MRPC_train.txt')

# Extract features and labels for training
print("Extracting features and labels for training...")
X_train_concat = np.array([create_combined_vector(row['#1 String'], row['#2 String'], method='concatenation') for _, row in train_data.iterrows()])
X_train_mean = np.array([create_combined_vector(row['#1 String'], row['#2 String'], method='mean') for _, row in train_data.iterrows()])
X_train_max = np.array([create_combined_vector(row['#1 String'], row['#2 String'], method='max_pooling') for _, row in train_data.iterrows()])
y_train = train_data['Quality']

# Train SVM models with different kernels
kernels = ['rbf', 'linear', 'poly', 'sigmoid']
svm_models = {}

print("Training SVM models...")
for kernel in kernels:
    svm_concat = SVC(kernel=kernel)
    svm_concat.fit(X_train_concat, y_train)
    svm_models[f'concat_{kernel}'] = svm_concat
    joblib.dump(svm_concat, os.path.join(results_dir, f'svm_concat_{kernel}.joblib'))  # Save the model

    svm_mean = SVC(kernel=kernel)
    svm_mean.fit(X_train_mean, y_train)
    svm_models[f'mean_{kernel}'] = svm_mean
    joblib.dump(svm_mean, os.path.join(results_dir, f'svm_mean_{kernel}.joblib'))  # Save the model

    svm_max = SVC(kernel=kernel)
    svm_max.fit(X_train_max, y_train)
    svm_models[f'max_{kernel}'] = svm_max
    joblib.dump(svm_max, os.path.join(results_dir, f'svm_max_{kernel}.joblib'))  # Save the model
print("Training complete.")

# Load the test data
test_data = preprocess_data('JAIST-intern-data/MRPC_test.txt')

# Extract test features and labels
print("Extracting features and labels for testing...")
X_test_concat = np.array([create_combined_vector(row['#1 String'], row['#2 String'], method='concatenation') for _, row in test_data.iterrows()])
X_test_mean = np.array([create_combined_vector(row['#1 String'], row['#2 String'], method='mean') for _, row in test_data.iterrows()])
X_test_max = np.array([create_combined_vector(row['#1 String'], row['#2 String'], method='max_pooling') for _, row in test_data.iterrows()])
y_test = test_data['Quality']

# Evaluate models and save results to CSV
print("Evaluating models...")
results = []

for method in ['concat', 'mean', 'max']:
    for kernel in kernels:
        model = svm_models[f'{method}_{kernel}']
        if method == 'concat':
            X_test = X_test_concat
        elif method == 'mean':
            X_test = X_test_mean
        elif method == 'max':
            X_test = X_test_max
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Method': method, 'Kernel': kernel, 'Accuracy': accuracy})
print("Evaluation complete.")
# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, 'sbert_evaluation_results.csv'), index=False)

# Print results
for index, row in results_df.iterrows():
    print(f"Method: {row['Method']}, Kernel: {row['Kernel']}, Accuracy: {row['Accuracy']}")
