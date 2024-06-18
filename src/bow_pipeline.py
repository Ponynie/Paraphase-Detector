import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

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

def make_feature_list(row):
    features = [word + '/s1' for word in row['#1 String']] + [word + '/s2' for word in row['#2 String']]
    return features

def create_feature_dictionary(train_data, test_data):
    feature_dict = defaultdict(int)
    idx = 0
    for features in pd.concat([train_data['Features'], test_data['Features']]):
        for feature in features:
            if feature not in feature_dict:
                feature_dict[feature] = idx
                idx += 1
    return feature_dict

def vectorize_features(features, feature_dict):
    vector = [0] * len(feature_dict)
    for feature in features:
        if feature in feature_dict:
            vector[feature_dict[feature]] += 1
    return vector

def train_and_evaluate_svm(train_vectors, y_train, test_vectors, y_test, kernels, results_dir, tag):
    svm_models = {}
    results = []
    
    for kernel in kernels:
        svm = SVC(kernel=kernel)
        svm.fit(train_vectors, y_train)
        svm_models[kernel] = svm
        if tag == '' or tag is None:
            joblib.dump(svm, os.path.join(results_dir, f'bow_{kernel}.joblib'))
        else:
            joblib.dump(svm, os.path.join(results_dir, f'bow_{kernel}_{tag}.joblib'))  # Save the model
    
    for kernel in kernels:
        model = svm_models[kernel]
        y_pred = model.predict(test_vectors)
        accuracy = accuracy_score(y_test, y_pred)
        results.append({'Method': 'bag-of-words', 'Kernel': kernel, 'Accuracy': accuracy})
    
    results_df = pd.DataFrame(results)
    if tag == '' or tag is None:
        results_df.to_csv(os.path.join(results_dir, 'bow_evaluation_results.csv'), index=False)
    else:
        results_df.to_csv(os.path.join(results_dir, f'bow_evaluation_results_{tag}.csv'), index=False)
    
    return results_df

def ensure_relative_path(path):
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, '..', path)
    return path

def run_pipeline(train_path, test_path, results_dir, tag=None):
    results_dir = ensure_relative_path(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print("Preprocessing data for bag-of-words...")
    train_data = preprocess_data_bow(train_path)
    test_data = preprocess_data_bow(test_path)
    
    print("Making feature lists for bag-of-words...")
    train_data['Features'] = train_data.apply(make_feature_list, axis=1)
    test_data['Features'] = test_data.apply(make_feature_list, axis=1)
    
    print("Creating feature dictionary for bag-of-words...")
    feature_dict = create_feature_dictionary(train_data, test_data)
    
    print("Vectorizing features for bag-of-words...")
    train_vectors = np.array([vectorize_features(features, feature_dict) for features in train_data['Features']])
    test_vectors = np.array([vectorize_features(features, feature_dict) for features in test_data['Features']])
    y_train = train_data['Quality'].astype(int)
    y_test = test_data['Quality'].astype(int)
    
    print("Training and evaluating SVM models for bag-of-words...")
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    results_df = train_and_evaluate_svm(train_vectors, y_train, test_vectors, y_test, kernels, results_dir, tag)
    
    print("Results:")
    for _, row in results_df.iterrows():
        print(f"Method: {row['Method']}, Kernel: {row['Kernel']}, Accuracy: {row['Accuracy']}")

