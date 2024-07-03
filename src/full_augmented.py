from pipeline import NLPPipeline
import os

def FA_MRPC():
    print('Full Augmented MRPC')
    
    train_dir = os.path.join('data', 'MRPC_data', 'full_augmented', 'FA_MRPC_train.txt')
    test_dir = os.path.join('data', 'MRPC_data', 'test_data', 'MRPC_test.txt')
    results_dir = os.path.join('results', 'MRPC_data', 'full_augmented')
    
    pipeline = NLPPipeline(train_dir, test_dir, results_dir, prefix='FA', save_models=False)
    pipeline.lite_execute()
    

def FA_Sample():
    print('Full Augmented Sample')
    
    train_dir = os.path.join('data', 'Sample_data', 'full_augmented', 'FA_Sample_train.txt')
    test_dir = os.path.join('data', 'Sample_data', 'test_data', 'Sample_test.txt')
    results_dir = os.path.join('results', 'Sample_data', 'full_augmented')
    
    pipeline = NLPPipeline(train_dir, test_dir, results_dir, prefix='FA', save_models=False)
    pipeline.lite_execute()
    
FA_Sample()