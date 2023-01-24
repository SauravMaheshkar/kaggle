

import os
import wandb
import random
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from wandb.keras import WandbCallback
from sklearn.model_selection import KFold
from transformers import TFAutoModelForSequenceClassification, TFAutoModel, AutoTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 42
seed_everything(seed)
warnings.filterwarnings('ignore')

train_filepath = '../data/train.csv'
test_filepath = '../data/test.csv'

train = pd.read_csv(train_filepath)
test = pd.read_csv(test_filepath)

# removing unused columns
train.drop(['url_legal', 'license'], axis=1, inplace=True)
test.drop(['url_legal', 'license'], axis=1, inplace=True)

api_key = ""
wandb.login(key=api_key);

# ## Device Configuration 🔌

DEVICE = 'GPU'

if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE == "GPU":
    n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", n_gpu)
    
    if n_gpu > 1:
        print("Using strategy for multiple GPU")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print('Standard strategy for GPU...')
        strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync

BATCH_SIZE = 8 * REPLICAS
LEARNING_RATE = 1e-3 * REPLICAS
EPOCHS = 15
N_FOLDS = 5
SEQ_LEN = 300
BASE_MODEL = ''
NEW_NAME = "John"
proper_names = ['fayre', 'roger', 'blaney']

# # Pre-Processing 👎🏻 -> 👍

import string

# must check this list
stop_words = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ] 

table = str.maketrans('', '', string.punctuation)

# Remove Punctuations
def remove_punctuation(orig_str):
    words = orig_str.split()
    filtered_sentence = ""
    
    for word in words:
        word = word.translate(table)
        filtered_sentence = filtered_sentence + word + " "
    
    return filtered_sentence

# Remove all Stopwords
def remove_stopwords(orig_str):
    filtered_sentence = ""
    
    words = orig_str.split(" ")
    
    for word in words:
        if word not in stop_words:
            filtered_sentence = filtered_sentence + word + " "
            
    return filtered_sentence

# Substitude proper names (all changed with John)
def change_proper_names(orig_str):
    filtered_sentence = ""
    
    words = orig_str.split(" ")
    
    for word in words:
        if word not in proper_names:
            filtered_sentence = filtered_sentence + word + " "
        else:
            filtered_sentence = filtered_sentence + NEW_NAME + " "
            
    return filtered_sentence

# A Custom Standardization Function
def custom_standardization(text):
    text = text.lower()
    text = text.strip()
    return text

# Sampling Function
def sample_target(features, target):
    mean, stddev = target
    sampled_target = tf.random.normal([], mean=tf.cast(mean, dtype=tf.float32), 
                                    stddev=tf.cast(stddev, dtype=tf.float32), dtype=tf.float32)
    
    return (features, sampled_target)
    

# Convert to tf.data.Dataset
def get_dataset(pandas_df, tokenizer, labeled=True, ordered=False, repeated=False, 
                is_sampled=False, batch_size=32, seq_len=128):
    
    pandas_df['excerpt'] = pandas_df['excerpt'].apply(remove_punctuation)
    pandas_df['excerpt'] = pandas_df['excerpt'].apply(remove_stopwords)
    pandas_df['excerpt'] = pandas_df['excerpt'].apply(change_proper_names)
    
    text = [custom_standardization(text) for text in pandas_df['excerpt']]
    
    # Tokenize inputs
    tokenized_inputs = tokenizer(text, max_length=seq_len, truncation=True, 
                                padding='max_length', return_tensors='tf')
    
    if labeled:
        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': tokenized_inputs['input_ids'], 
                                                    'attention_mask': tokenized_inputs['attention_mask']}, 
                                                    (pandas_df['target'], pandas_df['standard_error'])))
        if is_sampled:
            dataset = dataset.map(sample_target, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices({'input_ids': tokenized_inputs['input_ids'], 
                                                    'attention_mask': tokenized_inputs['attention_mask']})
        
    if repeated:
        dataset = dataset.repeat()
    if not ordered:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# # The Model 👷‍♀️

initial_learning_rate = 0.01

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True)

def get_model(encoder, seq_len=256):
    
    input_ids = tf.keras.layers.Input(shape=(seq_len), dtype=tf.int32, name='input_ids')
    
    input_attention_mask = tf.keras.layers.Input(shape=(seq_len,), dtype=tf.int32, name='attention_mask')
    
    output = encoder({'input_ids': input_ids, 
                    'attention_mask': input_attention_mask})
    
    model = tf.keras.Model(inputs = [input_ids, input_attention_mask], 
                        outputs = output, name = "CommonLit_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, 
                loss=tf.keras.losses.MeanSquaredError(), 
                metrics=['mse'])
    
    return model

# # Training 💪

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

history_list = []
test_pred = []

for fold,(idxT, idxV) in enumerate(skf.split(train)):
        

    # Create Model
    tf.keras.backend.clear_session()
    with strategy.scope():
        encoder = TFAutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
        model = get_model(encoder, SEQ_LEN)

    # Callbacks
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True)
    
    # Create a W&B run
    run = wandb.init(project='commonlit', entity='sauravmaheshkar', reinit=True, sync_tensorboard=True)

    # Train
    history = model.fit(x=get_dataset(train.loc[idxT], tokenizer, repeated=True, is_sampled=True, 
                                    batch_size=BATCH_SIZE, seq_len=SEQ_LEN), 
                        validation_data=get_dataset(train.loc[idxV], tokenizer, ordered=True, 
                                                    batch_size=BATCH_SIZE, seq_len=SEQ_LEN), 
                        steps_per_epoch=100, 
                        callbacks=[early_stopping_cb,WandbCallback(monitor='val_mse', mode='min', 
                                save_model=False)], 
                        epochs=EPOCHS,  
                        verbose=1).history
    
    run.finish()
    
    history_list.append(history)
    
    # Test predictions
    test_ds = get_dataset(test, tokenizer, labeled=False, ordered=True, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    x_test = test_ds.map(lambda sample: sample)
    test_pred.append(model.predict(x_test)['logits'])


# # Submission

submission = test[['id']]
submission['target'] = np.mean(test_pred, axis=0)
submission.to_csv('submission.csv', index=False)
