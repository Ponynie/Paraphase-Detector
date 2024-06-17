# Import Defualt Libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Data directory
train_dir = 'JAIST-intern-data/testrun_train.txt'
test_dir = 'JAIST-intern-data/testrun_train.txt'

# Create the results directory
results_dir = 'results/train_test_results_bow'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Conversion of Sentence to Word List
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_sentence(sentence, order='stem/stop'):
    if order == 'stop/stem':
        words = word_tokenize(sentence)
        words = [ps.stem(word.lower()) for word in words if word.lower() not in stop_words]
        return words
    elif order == 'stem/stop':
        words = word_tokenize(sentence)
        words = [ps.stem(w.lower()) for w in words]
        words = [w for w in words if w not in stop_words]
        return words
    else:
        raise ValueError(f"Invalid order: {order}")


def preprocess_data_bow(filepath):
    clean_data = {'Quality': [], '#1 ID': [], '#2 ID': [], '#1 String': [], '#2 String': []}
    with open(filepath, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                clean_data['Quality'].append(parts[0])
                clean_data['#1 ID'].append(parts[1])
                clean_data['#2 ID'].append(parts[2])
                clean_data['#1 String'].append(preprocess_sentence(parts[3]))
                clean_data['#2 String'].append(preprocess_sentence(parts[4]))
            else:
                print(f"Skipping malformed line: {line}")
    return pd.DataFrame(clean_data)

# Preprocess the training and test data for bag-of-words
train_data_bow = preprocess_data_bow(train_dir)
test_data_bow = preprocess_data_bow(test_dir)

# Making Feature List
def make_feature_list(row):
    features = [word + '/s1' for word in row['#1 String']] + [word + '/s2' for word in row['#2 String']]
    return features

train_data_bow['Features'] = train_data_bow.apply(make_feature_list, axis=1)
test_data_bow['Features'] = test_data_bow.apply(make_feature_list, axis=1)

# Making Dictionary of Features
from collections import defaultdict

def create_feature_dictionary(train_data, test_data):
    feature_dict = defaultdict(int)
    idx = 0
    for features in pd.concat([train_data['Features'], test_data['Features']]):
        for feature in features:
            if feature not in feature_dict:
                feature_dict[feature] = idx
                idx += 1
    return feature_dict

feature_dict = create_feature_dictionary(train_data_bow, test_data_bow)

# Make Vector of Sentence Pair
def vectorize_features(features, feature_dict):
    vector = [0] * len(feature_dict)
    for feature in features:
        if feature in feature_dict:
            vector[feature_dict[feature]] += 1
    return vector

train_vectors_bow = np.array([vectorize_features(features, feature_dict) for features in train_data_bow['Features']])
test_vectors_bow = np.array([vectorize_features(features, feature_dict) for features in test_data_bow['Features']])
y_train_bow = train_data_bow['Quality'].astype(int)
y_test_bow = test_data_bow['Quality'].astype(int)

# Training SVM and Evaluation
kernels = ['rbf', 'linear', 'poly', 'sigmoid']
svm_models_bow = {}

for kernel in kernels:
    svm = SVC(kernel=kernel)
    svm.fit(train_vectors_bow, y_train_bow)
    svm_models_bow[kernel] = svm
    joblib.dump(svm, os.path.join(results_dir, f'bow_{kernel}.joblib'))  # Save the model

# Evaluate models and save results to CSV
results_bow = []

for kernel in kernels:
    model = svm_models_bow[kernel]
    y_pred_bow = model.predict(test_vectors_bow)
    accuracy_bow = accuracy_score(y_test_bow, y_pred_bow)
    results_bow.append({'Method': 'bag-of-words', 'Kernel': kernel, 'Accuracy': accuracy_bow})

# Convert results to DataFrame and save to CSV
results_df_bow = pd.DataFrame(results_bow)
results_df_bow.to_csv(os.path.join(results_dir, 'bow_evaluation_results.csv'), index=False)

# Print results
for index, row in results_df_bow.iterrows():
    print(f"Method: {row['Method']}, Kernel: {row['Kernel']}, Accuracy: {row['Accuracy']}")

