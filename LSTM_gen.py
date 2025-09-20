import numpy as np
import pandas as pd
from src.utils.config import CFG
from src.utils.dataset_utils import *

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from focal_loss import SparseCategoricalFocalLoss
import matplotlib.pyplot as plt
import seaborn as sns
import csv

np.random.seed(CFG['project']['seed'])
tf.random.set_seed(CFG['project']['seed'])


# Convert protein sequences to embedding vectors using 3-gram dictionary
def create_embeddings_sequences(df, df_3grams):
    """
    Creates embeddings sequences for each row of the dataset

    Parameters:
    df: dataframe
    df_3grams: dataframe with embeddings of 3grams

    Returns:
    df: dataframe with sequence of embeddings per row
    """

    # dict with the embedding of each 3gram
    #embedding_dict = {row['words']: row[1:].tolist() for _, row in df_3grams.iterrows()}
    embedding_dict = df_3grams
    # add the column with the list of the 3grams embeddings
    #df['Sequence_embeddings'] = None
    #for i in df:
    sequence_embedding = []
    protein_seq = df

    # Create the list of 3grams embeddings of the protein sequence
    for j in range(len(protein_seq) - 2):
        trigram = protein_seq[j:j + 3]

        if trigram in embedding_dict:
            embedding = embedding_dict[trigram]
        else:
            embedding = embedding_dict['<unk>']

        sequence_embedding.append(embedding)

    # Add the sequence to the dataframe
    #df.at[i, 'Sequence_embeddings'] = np.array(sequence_embedding)

    return np.array(sequence_embedding)


# Balance dataset
def balance_df(df):
    """
    Balance dataframe with the same number of 'other' classes as the number of classes to classify

    Parameters:
    df: dataframe to be balanced

    Returns:
    df: balanced dataframe
    """
    n = CFG['data']['num_classes']
    # df_final = preprocess_data(df_final, n)
    top_families = list(df['label'].value_counts()[:n + 1].index)
    top_families.remove('other')

    mask = df['label'].isin(top_families)
    df_topfamilies = df[mask]
    df_others = df[~mask]

    df_others = df_others.sample(n=len(df_topfamilies), random_state=CFG['project']['seed'])

    df = pd.concat([df_topfamilies, df_others], axis=0)
    df.reset_index(drop=True, inplace=True)

    return df


def protein_data_generator(sequences, labels, df_3grams, max_length=None, batch_size=32, ):
    """
    Generator that yields batches of protein embedding data instead of loading everything at once.
    This is like having a conveyor belt that brings you just the protein sequences you need,
    pads them to the right size, and packages them into neat batches.

    The beauty of this approach is that we never hold more than one batch worth of
    padded sequences in memory at any time.

    Parameters:
    sequences: list of embedding sequences (numpy arrays of varying lengths)
    labels: list of corresponding labels
    max_length: maximum sequence length for padding
    batch_size: number of sequences per batch

    Yields:
    batch_data: numpy array of shape (batch_size, max_length, embedding_dim)
    batch_labels: numpy array of shape (batch_size,)
    """

    embedding_dict = {row['words']: row[1:].tolist() for _, row in df_3grams.iterrows()}
    while True:  # Infinite loop for training - this allows multiple epochs
        # Shuffle the data indices for each epoch
        # This ensures we see data in different orders each time through
        indices = np.random.permutation(len(sequences))

        if max_length is None:
            # Calculate the sequence length at 95th percentile to avoid excessive padding
            lengths = [len(seq) for seq in sequences]
            max_length = int(np.percentile(lengths, 95))
            print(f"Using max_length of {max_length} (95th percentile of sequence lengths)")

        #print(sequences.head())
        #print(sequences.shape)
        #print(len(sequences))
        # Process data in batches
        for i in range(0, len(sequences), batch_size):
            #print(f"\nCalculating embeddings from {i} to {i + batch_size}")
            batch_indices = indices[i:i + batch_size]

            # Collect the sequences and labels for this batch
            batch_sequences = [create_embeddings_sequences(sequences.iloc[idx], embedding_dict) for idx in batch_indices]
            batch_labels = [labels[idx] for idx in batch_indices]

            # Pad each sequence in the batch individually
            # We only pad what we need right now
            batch_data = pad_embedding_sequences(batch_sequences, max_length)

            # Convert to numpy arrays and yield
            # The batch_data will be garbage collected after yielding, freeing memory
            #print(batch_data.shape, len(batch_labels))
            data = [{'sequence': x, 'label': y} for x, y in zip(sequences[i:i + batch_size], batch_labels)]

            csv_file_path = '..\\..\\data\\test_predictions.csv'
            fieldnames = ['sequence', 'label']

            # Check if file exists to decide whether to write header
            write_header = not os.path.exists(csv_file_path)

            with open(csv_file_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(data)

            yield np.array(batch_data), np.array(batch_labels)

def preprocess_newdataset(df_meta):
    df_meta = df_meta[['Sequence', 'Pfam']]

    top_families = list(df_meta['Pfam'].value_counts()[:10].index)

    mask = df_meta['Pfam'].isin(top_families)
    df_topfamilies = df_meta[mask]
    df_others = df_meta[~mask]

    df_others.loc[:, 'Pfam'] = 'other'
    df = pd.concat([df_topfamilies, df_others])
    return df

# Load protein data and prepare embedding sequences for LSTM training
def load_data(file_3grams):
    """
    Load protein sequences and metadata, convert to embeddings and encode labels.
    Now returns sequences and labels as lists instead of trying to store everything
    in memory-heavy structures.

    Parameters:
    file_3grams (str): Path to CSV file with 3-gram embeddings

    Returns:
    X_train, y_train, X_val, y_val, X_test, y_test_encoded: lists of sequences and labels
    label_encoder: LabelEncoder for converting family names to integers
    """
    make_test_train_folders()
    print("Reading train data...")
    df_train = pd.read_csv("..\\..\\data\\train\\train.csv")
    df_train = df_train[['Sequence', 'Pfam']]
    df_train.columns = ["sequence", "label"]
    print(df_train.head())
    print("Reading test data...")
    df_test = pd.read_csv("..\\..\\data\\test\\test20.csv")
    df_test = df_test[['Sequence', 'Pfam']]
    df_test.columns = ["sequence", "label"]
    print(df_test.head())

    print("Reading 3grams data...")
    file_3grams = os.path.join(CFG.data_dir, file_3grams)
    df_3grams = pd.read_csv(file_3grams, sep='\t')

    #df_train = balance_df(df_train)
    #df_test = balance_df(df_test)

    # Create embedding sequences - returns lists instead of DataFrame columns
    #X_train_val, y_train_val = create_embeddings_sequences(df_train, df_3grams)
    #X_test, y_test = create_embeddings_sequences(df_test, df_3grams)

    X_train_val = df_train['sequence']
    y_train_val = df_train['label']
    X_test = df_test['sequence']
    y_test = df_test['label']

    # Encode the family labels
    label_encoder = LabelEncoder()
    y_train_val_encoded = label_encoder.fit_transform(y_train_val)
    y_test_encoded = label_encoder.transform(y_test)

    # Split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val_encoded,
        test_size=CFG["data"]["validation_split"],
        random_state=CFG["project"]["seed"],
        stratify=y_train_val_encoded,
        shuffle=CFG["data"]["shuffle"]
    )

    print(f"Loaded {len(X_train_val) + len(X_test)} protein sequences")
    print(f"Number of unique families: {len(label_encoder.classes_)}")
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test_encoded, label_encoder


# Pad embedding sequences to uniform length for batch processing
def pad_embedding_sequences(X, max_length=None):
    """
    Pad or truncate sequences to consistent length using 95th percentile.

    Parameters:
    X (list): List of embedding matrices for each protein
    max_length (int, optional): Maximum sequence length. If None, calculated from data

    Returns:
    padded_X (array): Padded sequences with uniform length
    """
    #print(X[0])
    #print(f"Padding {len(X)} sequences")
    if max_length is None:
        # Calculate the sequence length at 95th percentile to avoid excessive padding
        lengths = [len(seq) for seq in X]
        max_length = int(np.percentile(lengths, 95))
        print(f"Using max_length of {max_length} (95th percentile of sequence lengths)")

    # Create a padding mask - sequences shorter than max_length will be padded with zeros
    # Sequences longer than max_length will be truncated
    padded_X = []
    for seq in X:
        if len(seq) > max_length:
            # Trim
            padded_X.append(seq[:max_length])
        else:
            # Pad with zeros
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            padded_X.append(np.vstack([seq, padding]))

    return np.array(padded_X)


# Build bidirectional LSTM model for protein family classification
def build_lstm_model(input_shape, num_classes):
    """
    Create sequential model with bidirectional LSTM layers and regularization.

    Parameters:
    input_shape (tuple): Shape of input data (sequence_length, embedding_dim)
    num_classes (int): Number of protein families to classify

    Returns:
    model: Compiled Keras model with focal loss and Adam optimizer
    """
    model = Sequential([
        # Bidirectional LSTM layer to capture context from both directions
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        # Second LSTM layer
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=SparseCategoricalFocalLoss(gamma=2.0),
        metrics=['accuracy']
    )

    return model


# Train LSTM model with callbacks for early stopping and learning rate reduction
def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train model with validation monitoring and automatic optimization callbacks.

    Parameters:
    model: Compiled Keras model to train
    X_train, y_train: Training data and labels
    X_val, y_val: Validation data and labels
    batch_size (int): Batch size for training
    epochs (int): Maximum number of epochs

    Returns:
    history: Training history with loss and accuracy metrics
    """
    # Stop training when val_loss does not improve after 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # Model checkpoint to save best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return history


# Train LSTM model with memory-efficient data generator
def train_model_with_generator(model, train_generator, val_generator,
                               steps_per_epoch, validation_steps, epochs=50):
    """
    Train model using data generators instead of loading all data into memory.
    This is the key difference - we pass generators instead of arrays to model.fit()

    Parameters:
    model: Compiled Keras model to train
    train_generator: generator yielding training batches
    val_generator: generator yielding validation batches
    steps_per_epoch: number of batches per epoch for training
    validation_steps: number of batches for validation
    epochs: maximum number of epochs

    Returns:
    history: Training history with loss and accuracy metrics
    """
    # Stop training when val_loss does not improve after 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # Model checkpoint to save best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Train using generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )

    return history


# Evaluate model performance using generator for memory efficiency
def evaluate_model_with_generator(model, test_generator, test_steps, y_test, label_encoder):
    """
    Test model accuracy using a generator to avoid loading all test data at once.
    We predict in batches and then combine the results.

    Parameters:
    model: Trained Keras model to evaluate
    test_generator: generator yielding test batches
    test_steps: number of batches in test set
    y_test: true test labels (for comparison)
    label_encoder: LabelEncoder used for converting labels to class names

    Returns:
    None (prints classification report and saves confusion matrix plot)
    """
    # Predict using generator - this processes test data in batches
    y_pred_proba = model.predict(test_generator, steps=test_steps)
    y_pred = np.argmax(y_pred_proba, axis=1)

    csv_file_path = '..\\..\\data\\test_predictions.csv'
    df = pd.read_csv(csv_file_path)
    df = df.iloc[:min(len(df),len(y_pred))]
    df['predicted_label'] = y_pred
    df.to_csv(csv_file_path, index=False)

    y_test=df['label']
    y_pred=df['predicted_label']

    """l = min(len(y_test), len(y_pred))
    if len(y_test) > l:
        y_test = y_test[:l]
    if len(y_pred) > l:
        y_pred = y_pred[:l]"""

    # Print classification report
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred, normalize="pred")
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('LSTM_confusion_matrix.png')
    plt.close()




def plot_training_history(history):
    """
    Plot accuracy and loss curves from training history
    """
    # Plot accuracy and loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.savefig('LSTM_training_history.png')
    plt.close()


if __name__ == '__main__':
    file_3grams = "protVec_100d_3grams.csv"

    # Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_data(file_3grams)

    # Determine suitable max_length for padding
    sequence_lengths = [len(seq) for seq in X_train]
    max_length = int(np.percentile(sequence_lengths, 95))  # 95th percentile
    print(f"Max sequence length (95th percentile): {max_length}")

    # Get model parameters
    embedding_dim = 100
    num_classes = len(np.unique(y_train))

    print(f"Embedding dimension: {embedding_dim}")
    print(f"Number of classes: {num_classes}")

    # Set up batch processing parameters
    batch_size = 32  # You can reduce this further if you still have memory issues
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    test_steps = len(X_test) // batch_size

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Test steps: {test_steps}")

    file_3grams = os.path.join(CFG.data_dir, file_3grams)
    df_3grams = pd.read_csv(file_3grams, sep='\t')

    # Create data generators - these will handle memory-efficient data loading
    train_generator = protein_data_generator(X_train, y_train, df_3grams, max_length, batch_size)
    val_generator = protein_data_generator(X_val, y_val, df_3grams, max_length, batch_size)
    test_generator = protein_data_generator(X_test, y_test, df_3grams, max_length, batch_size)

    model = build_lstm_model(
        input_shape=(max_length, embedding_dim),
        num_classes=num_classes
    )

    model.summary()

    """# Train the model using generators
    history = train_model_with_generator(
        model, train_generator, val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=50
    )"""

    # Evaluate the model using generator
    model.load_weights("best_model.keras")
    evaluate_model_with_generator(model, test_generator, test_steps, y_test, label_encoder)
    #plot_training_history(history)
