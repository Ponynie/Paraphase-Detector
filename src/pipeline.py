import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from tqdm import tqdm  # Import tqdm

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NLPStaticComponents:
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    sbert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    
class NLPPipeline:
    
    original_sbert_cache = pd.DataFrame()
    original_bow_cache = pd.DataFrame() 
    
    testset_sbert_cache = pd.DataFrame()
    testset_bow_cache = pd.DataFrame()
    
    def __init__(self, train_path, test_path, results_dir, prefix, save_models=False):
        self.train_path = self.ensure_relative_path(train_path)
        self.test_path = self.ensure_relative_path(test_path)
        self.results_dir = self.ensure_relative_path(results_dir)
        self.stop_words = NLPStaticComponents.stop_words
        self.ps = NLPStaticComponents.ps
        self.sbert_model = NLPStaticComponents.sbert_model
        self.kernels = NLPStaticComponents.kernels
        self.save_models = save_models
        self.prefix = prefix

    def ensure_relative_path(self, path):
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, '..', path)

    def preprocess_sentence(self, sentence, order='stem/stop'):
        words = word_tokenize(sentence)
        if order == 'stop/stem':
            words = [self.ps.stem(word.lower()) for word in words if word.lower() not in self.stop_words]
        elif order == 'stem/stop':
            words = [self.ps.stem(w.lower()) for w in words]
            words = [w for w in words if w not in self.stop_words]
        else:
            raise ValueError(f"Invalid order: {order}")
        return words

    def preprocess_data(self, filepath, method='bow'):
        clean_data = {'Quality': [], '#1 ID': [], '#2 ID': [], '#1 String': [], '#2 String': []}
        total_lines = sum(1 for _ in open(filepath, 'r'))
        with open(filepath, 'r') as file:
            next(file)  # Skip the header line
            for line in tqdm(file, total=total_lines - 1, desc=f"Preprocessing {method.upper()} data"):
                parts = line.strip().split('\t')
                if len(parts) == 5:
                    clean_data['Quality'].append(parts[0])
                    clean_data['#1 ID'].append(parts[1])
                    clean_data['#2 ID'].append(parts[2])
                    if method == 'bow':
                        clean_data['#1 String'].append(self.preprocess_sentence(parts[3]))
                        clean_data['#2 String'].append(self.preprocess_sentence(parts[4]))
                    else:
                        clean_data['#1 String'].append(self.get_sentence_vector(parts[3]))
                        clean_data['#2 String'].append(self.get_sentence_vector(parts[4]))
                else:
                    print(f"Skipping malformed line: {line}")
        return pd.DataFrame(clean_data)

    def get_sentence_vector(self, sentence):
        return self.sbert_model.encode([sentence])[0]

    def create_combined_vector(self, vec1, vec2, method='concatenation'):
        if method == 'concatenation':
            return np.concatenate((vec1, vec2))
        elif method == 'mean':
            return (vec1 + vec2) / 2
        elif method == 'max_pooling':
            return np.maximum(vec1, vec2)
        else:
            raise ValueError("Method not supported")

    def make_feature_list(self, row):
        return [word + '/s1' for word in row['#1 String']] + [word + '/s2' for word in row['#2 String']]

    def create_feature_dictionary(self, train_data, test_data):
        feature_dict = defaultdict(int)
        idx = 0
        for features in tqdm(pd.concat([train_data['Features'], test_data['Features']]), desc="Creating feature dictionary"):
            for feature in features:
                if feature not in feature_dict:
                    feature_dict[feature] = idx
                    idx += 1
        return feature_dict

    def vectorize_features(self, features, feature_dict):
        vector = [0] * len(feature_dict)
        for feature in features:
            if feature in feature_dict:
                vector[feature_dict[feature]] += 1
        return vector

    def train_and_evaluate_svm(self, train_vectors, y_train, test_vectors, y_test, tag):
        svm_models = {}
        results = []
        
        for kernel in tqdm(self.kernels, desc=f"Training SVM models ({tag})"):
            svm = SVC(kernel=kernel)
            svm.fit(train_vectors, y_train)
            svm_models[kernel] = svm
            if self.save_models:
                joblib.dump(svm, os.path.join(self.results_dir, f'{tag}_{kernel}.joblib')) # Save the model
        
        for kernel in tqdm(self.kernels, desc=f"Evaluating SVM models ({tag})"):
            model = svm_models[kernel]
            y_pred = model.predict(test_vectors)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({'Method': tag, 'Kernel': kernel, 'Accuracy': accuracy})
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.results_dir, f'{tag}_evaluation_results.csv'), index=False)
        
        return results_df

    def original_data_cache(self, train_data: pd.DataFrame , pipeline_type):

        def original_train_path():
            path = re.sub(r'/(full_augmented|augmented_\d+)', '/original', self.train_path)
            path = re.sub(r'/(FA_|A\d+_)', '/O_', path)
            return path
        
        if pipeline_type == 'bow':
            if not NLPPipeline.original_bow_cache.empty:
                return pd.concat([NLPPipeline.original_bow_cache, train_data])
            elif self.prefix == 'O':
                NLPPipeline.original_bow_cache = train_data.copy()
                return train_data
            else:
                original_train_data = self.preprocess_data(original_train_path(), method='bow')
                return pd.concat([original_train_data, train_data])
        elif pipeline_type == 'sbert':
            if not NLPPipeline.original_sbert_cache.empty:
                return pd.concat([NLPPipeline.original_sbert_cache, train_data])
            elif self.prefix == 'O':
                NLPPipeline.original_sbert_cache = train_data.copy()
                return train_data
            else:
                original_train_data = self.preprocess_data(original_train_path(), method='sbert')
                return pd.concat([original_train_data, train_data])
        else:
            raise ValueError("Invalid pipeline type")
        
    def testset_data_cache(self, pipeline_type):
        if pipeline_type == 'bow':
            if not NLPPipeline.testset_bow_cache.empty:
                return NLPPipeline.testset_bow_cache.copy()
            else:
                NLPPipeline.testset_bow_cache = self.preprocess_data(self.test_path, method='bow')
                return NLPPipeline.testset_bow_cache.copy()
        elif pipeline_type == 'sbert':
            if not NLPPipeline.testset_sbert_cache.empty:
                return NLPPipeline.testset_sbert_cache.copy()
            else:
                NLPPipeline.testset_sbert_cache = self.preprocess_data(self.test_path, method='sbert')
                return NLPPipeline.testset_sbert_cache.copy()

    def run_bow_pipeline(self):
        print("Running Bag-of-Words pipeline...")
        train_data = self.preprocess_data(self.train_path, method='bow')
        test_data = self.testset_data_cache(pipeline_type='bow')
        
        train_data = self.original_data_cache(train_data, pipeline_type='bow')
        
        train_data['Features'] = train_data.apply(self.make_feature_list, axis=1)
        test_data['Features'] = test_data.apply(self.make_feature_list, axis=1)
        
        feature_dict = self.create_feature_dictionary(train_data, test_data)
        
        train_vectors = np.array([self.vectorize_features(features, feature_dict) for features in tqdm(train_data['Features'], desc="Vectorizing train features")])
        test_vectors = np.array([self.vectorize_features(features, feature_dict) for features in tqdm(test_data['Features'], desc="Vectorizing test features")])
        y_train = train_data['Quality'].astype(int)
        y_test = test_data['Quality'].astype(int)
        
        return self.train_and_evaluate_svm(train_vectors, y_train, test_vectors, y_test, tag=f'{self.prefix}_bow')

    def run_sbert_pipeline(self):
        print("Running Sentence-BERT pipeline...")
        train_data = self.preprocess_data(self.train_path, method='sbert')
        test_data = self.testset_data_cache(pipeline_type='sbert')
        
        train_data = self.original_data_cache(train_data, pipeline_type='sbert')
        
        methods = ['concatenation', 'mean', 'max_pooling']
        results = []
        
        for method in methods:
            print(f"Processing SBERT with {method} method...")
            X_train = np.array([self.create_combined_vector(row['#1 String'], row['#2 String'], method=method) for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc=f"Creating train vectors ({method})")])
            X_test = np.array([self.create_combined_vector(row['#1 String'], row['#2 String'], method=method) for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc=f"Creating test vectors ({method})")])
            y_train = train_data['Quality']
            y_test = test_data['Quality']
            
            results_df = self.train_and_evaluate_svm(X_train, y_train, X_test, y_test, tag=f'{self.prefix}_sbert_{method}')
            results.extend(results_df.to_dict('records'))
        
        return pd.DataFrame(results)

    def execute(self):
        sbert_results = self.run_sbert_pipeline()
        bow_results = self.run_bow_pipeline()

        combined_results = pd.concat([bow_results, sbert_results])
        combined_results.to_csv(os.path.join(self.results_dir, f'{self.prefix}_evaluation_results.csv'), index=False)
        
        print("Results:")
        for _, row in combined_results.iterrows():
            print(f"Method: {row['Method']}, Kernel: {row['Kernel']}, Accuracy: {row['Accuracy']}")
            
    def lite_execute(self, kernels=['rbf']):
        sbert_results = self.run_sbert_pipeline()
        
        print(f"Bag-of-World Running Lite execution with kernels: {kernels}...")
        self.kernels = kernels
        bow_results = self.run_bow_pipeline()
        
        combined_results = pd.concat([bow_results, sbert_results])
        combined_results.to_csv(os.path.join(self.results_dir, f'{self.prefix}_evaluation_results.csv'), index=False)
        
        print("Results:")
        for _, row in combined_results.iterrows():
            print(f"Method: {row['Method']}, Kernel: {row['Kernel']}, Accuracy: {row['Accuracy']}")
        
        print("Lite execution completed successfully.")