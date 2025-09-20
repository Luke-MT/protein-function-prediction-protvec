# protein-function-prediction-protvec
LSTM model for classifying protein function families from amino acid sequences using 3-gram embeddings. Achieves 97% accuracy and 98% F1-score.

# model
Architecture: Bidirectional LSTM

Features: ProtVec 3-gram embeddings (100D), focal loss for class imbalance, sequence padding at 95th percentile
