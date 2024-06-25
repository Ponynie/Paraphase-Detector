The data/ directory should be structured as follows:

data/
├── MRPC_data/
│   ├── original/
│   │   └── O_MRPC_train.txt
│   ├── augmented_1/
│   │   └── A1_MRPC_train.txt
│   │── ...
│   │
│   └── test_data/
│       └── MRPC_test.txt
└── Sample_data/
    ├── original/
    │   └── O_Sample_train.txt
    ├── augmented_1/
    │   └── A1_Sample_train.txt
    ├── ...
    │   
    └── test_data/
        └── Sample_test.txt

The results/ directory should be structured as follows:

results/
├── MRPC_data/
│   ├── original/
│   │
│   ├── augmented_1/
│   │
│   └── ...
└── Sample_data/
    ├── original/
    │
    ├── augmented_1/
    │
    └── ...