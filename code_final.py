# -*- coding: utf-8 -*-
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy.random import seed
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.utils import class_weight

from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import gensim
from gensim.models import Word2Vec

from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

from pattern.es import lemma


def vectorize_data(X_train, X_test, ngram):
    """
    This is the "bag of words" approach. Transforms training and testing data (=sentences) into numerical vectors
    by assigning the count of each word in the sentence.
    problem: longer documents will have higher average count values than shorter documents, even though they might talk about the same topics
    :param X_train: pd.Series: training data
    :param X_test: pd.Series: testing data
    :param ngram: bool: true if approach with ngrams, false if not
    :return: X_train (np.array), X_test (np.array): vectorized training and testing data
    """
    # with ngrams
    if ngram:
        vectorizer = CountVectorizer(min_df=0.05, max_df=0.95, ngram_range=(1,3))
    # without ngrams
    else:
        vectorizer = CountVectorizer(min_df=0.05, max_df=0.95)

    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test  = vectorizer.transform(X_test)
    vocab_size = len(vectorizer.get_feature_names())+1

    return X_train, X_test, vocab_size

def tf_idf_data(X_train, X_test, ngram):
    """
    This is the tf-idf approach. Transforms training and testing data (=sentences) into numerical vectors ().
    Also includes scaling the vectors via RobustScaler (this worked the best compared to MinMaxScaler and StandardScaler)
    Inuition: Downscale weights for words that occur in many documents in the corpus and are therefore less
    informative than those that occur only in a smaller portion of the corpus.
    :param X_train: pd.Series: training data
    :param X_test: pd.Series: testing data
    :param ngram: bool: true if approach with ngrams, false if not
    :return: X_train (np.array), X_test (np.array): vectorized and scaled training and testing data
    """
    # with ngrams
    if ngram:
        tfidfconverter = TfidfVectorizer(max_features=3000, ngram_range=(1,3), min_df=5, max_df=0.7)
    # without ngrams
    else:
        tfidfconverter = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.7)
        
    tfidfconverter.fit(X_train)
    X_train = tfidfconverter.transform(X_train)
    X_test  = tfidfconverter.transform(X_test)

    #sc = StandardScaler(with_mean=False)
    sc = RobustScaler(with_centering=False)
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    vocab_size = len(tfidfconverter.get_feature_names()) +1
    
    return X_train, X_test, vocab_size

def buildWordVector(text, size, model, google):
    """
    Helper method to build the vector per word. Solves the problem of not having a word in the vocabulary by keeping
    it a vector with 0s.
    :param text: str: sentence
    :param size: int: size of vector
    :param model: word2vec model for getting numeric repesentation of word
    :return: vec: word vector
    """
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            if not google:
                vec += model.wv.__getitem__(word).reshape((1, size))
            else:
                vec += model.__getitem__(word).reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def word_to_vec(X_train, X_test, model_w2v, google):
    """
    Transforms the words to vectors based on the word2vec model (which is built before)
    :param X_train: pd.Series: training data
    :param X_test: pd.Series: testing data
    :param model_w2v: model representation of word2vec model
    :return: X_train, X_test
    """
    n_dim = 300
    if not google:
        n_dim = 500
    X_train = np.concatenate([buildWordVector(z, n_dim, model_w2v, google) for z in X_train])
    X_test = np.concatenate([buildWordVector(z, n_dim, model_w2v, google) for z in X_test])
    #sc = StandardScaler(with_mean=False)
    sc = RobustScaler(with_centering=False)
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test =  sc.transform(X_test)

    return X_train, X_test

def word_indexing(X_train, X_test):
    """
    Use word indexing as transformation step, replacing each word by its word index in the vocubalary dictionary.
    Includes padding and scaling
    :param X_train: pd.Series: training data
    :param X_test: pd.Series: testing data
    :return: X_train (np.array), X_test (np.array): vectorized and scaled training and testing data
    """
    tokenizer = Tokenizer(num_words=5000)
    # create word->index dictionary
    tokenizer.fit_on_texts(X_train)

    # transform sentences to their index variable
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 100

    # ensure all sequences have the same length (some sentences might be longer than others)
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_test, vocab_size

def encode_y(y_train, y_test):
    """
    Label-encode and transform the the dependent variable to a vector representation of the classes
    so it can be used more easily by neural nets (and possibly other algorithms)
    :param y_train: pd.Series: training data
    :param y_test: pd.Series: testing data
    :return: y_train (np.array), y_test (np.array): vectorized and scaled training and testing y variable
    """
    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_test_score = y_test.copy()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return y_train, y_test, y_test_score


def build_nn(X_train):
    """
    Build an MLP neural network.
    :param X_train: pd.Series: training data
    :return: model: keras model of the specified neural network
    """
    input_dim = X_train.shape[1] 
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim))
    #model.add(layers.core.Dropout(0.3))
    model.add(layers.Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_embedding_nn(vocab_size, embedding_dim, maxlen, embedded, conv):
    """
    Build a neural network with embedding and pooling layer.
    Each index will get an embedding vector (=weights) which is randomly initialized and will be trained by the network.
    The size of the embedding has to be specified.
    :param vocab_size: int: number of distinct words in training set
    :param embedding_dim: int: size of the embedding vector (="features")
    :param maxlen: int: size of input sequence
    :return: model: keras model of the specified neural network
    """
    model = Sequential()
    if not embedded:
        model.add(layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   input_length=maxlen))

    if conv:
        model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.core.Dropout(0.5)) # for word indexing only
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def define_models():
    """
    Add all models to be tested to a list
    :return: models (list): all models to be tested as tuple
    """
    models = []
    models.append(('LR', LogisticRegression(class_weight='balanced', max_iter=2500)))
    models.append(('RF', RandomForestClassifier(class_weight='balanced',criterion='entropy')))
    models.append(('ET', ExtraTreesClassifier(class_weight='balanced',criterion='entropy')))
    models.append(('XG', xgb.XGBClassifier(class_weight='balanced',criterion='entropy')))
    models.append(('KNN', BalancedBaggingClassifier(base_estimator=KNeighborsClassifier(),
                                                    sampling_strategy='auto',
                                                    replacement=False)))
    models.append(('NB', MultinomialNB()))
    models.append(('MLP','mlp'))
    models.append(('EMB','emb'))
    models.append(('CON','con'))
    return models

def define_word2vec(X):
    """
    Create Word2Vec models based on given input X
    :param X: pd.Series: sentence as input
    :return: model: the built word2vec model
    """
    model_w2v = Word2Vec([word_tokenize(x) for x in X],
                         min_count=3,
                         size=500,
                         window=5,
                         iter=30)
    return model_w2v

def plot_learning_curves_nn(model):
    """
    Takes a Keras neural network model and plots its learning curves
    :param model: keras neural network model
    :return: None
    """
    history = model.history
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'],loc='upper right')
    plt.show()


# load data
data_sheet1 = pd.read_excel(r"C:\Users\Benjamin\Szenario Data Scientist\sentences_with_sentiment_final.xlsx")

# create one class column
data_sheet1['class'] = "-"
data_sheet1.loc[data_sheet1['Negative']==1, 'class']= "neg"
data_sheet1.loc[data_sheet1['Neutral']==1, 'class']= "neu"
data_sheet1.loc[data_sheet1['Positive']==1, 'class']= "pos"

# drop class columns
data_sheet1.drop(['Positive','Negative','Neutral'], axis=1, inplace=True)

# do some preparation which is required for all methods
data_sheet1['Sentence_prep'] = data_sheet1['Sentence'].str.lower()
data_sheet1['Sentence_prep_lemm'] = data_sheet1['Sentence_prep'].apply(lambda x:  ' '.join([lemma(word) for word in word_tokenize(x)
                                                                                  if (word not in stopwords.words('english'))
                                                                                  and (word.isalpha())]))

porter = PorterStemmer()
data_sheet1['Sentence_prep_stemm'] = data_sheet1['Sentence_prep'].apply(lambda x:  ' '.join([porter.stem(word) for word in word_tokenize(x)
                                                                                            if (word not in stopwords.words('english'))
                                                                                            and (word.isalpha())]))

models = define_models()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)

seed(77)
# bag of words (with and without ngrams), tf-idf (with and without ngrams), word embedding via pre-built model by google, word embedding via self-built model
strategies = ['bow-ngram','tfidf-ngram','bow-no','tfidf-no','bow-tf-idf-ngram', 'bow-tf-idf-no', 'word_indexing','word_to_vec_google','word_to_vec_self']

# load word2vec model from google, available at: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
model_google = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\Benjamin\Szenario Data Scientist\GoogleNews-vectors-negative300.bin', binary=True)

all_models_word2vec = [define_word2vec(data_sheet1['Sentence_prep_stemm']), define_word2vec(data_sheet1['Sentence_prep_lemm']),
                       define_word2vec(data_sheet1['Sentence'])]

embedding_dim = 300 #50
results = {}

Xs = [data_sheet1['Sentence_prep_stemm'], data_sheet1['Sentence_prep_lemm'], data_sheet1['Sentence']]
y = data_sheet1['class']

for ind_X, X in enumerate(Xs):
    model_w2v = all_models_word2vec[ind_X]
    for ind_strat, strat in enumerate(strategies):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%s'%strat)
        for name, model in models:
            print('++++++++++++++++++++++++++++++++++++++++++%s'%name)
            score_total = 0
            accuracy_score_list = []
            precision_score_list = []
            recall_score_list = []
            f1_score_list = []
            for train_index,test_index in skf.split(X,y):
                X_train, X_test = X[train_index],X[test_index]
                y_train, y_test = y[train_index],y[test_index]

                # possibility #1 - bag of words - term frequency
                if strat[:3] == 'bow':
                    ngram = True
                    if strat[:-2]=='no':
                        ngram = False
                    X_train, X_test, vocab_size = vectorize_data(X_train, X_test, ngram)
                # possibility #2 - tf-idf
                elif strat[:5] == 'tfidf':
                    ngram = True
                    if strat[:-2]=='no':
                        ngram = False
                    X_train, X_test, vocab_size = tf_idf_data(X_train, X_test, ngram)
                # possibility 3 - bag of words + tf-idf
                elif strat[0:5]=='bow-tf':
                    n_gram = True
                    if strat[:-2]=='no':
                        ngram = False
                    X_train, X_test, vocab_size = vectorize_data(X_train, X_test, ngram)
                    X_train, X_test, vocab_size = tf_idf_data(X_train, X_test, ngram)
                # possibility #3 - word indexing
                elif strat == 'word_indexing':
                    X_train, X_test, vocab_size = word_indexing(X_train, X_test)
                # possibility #4 - word2vec with google
                elif strat == 'word_to_vec_google':
                    X_train, X_test = word_to_vec(X_train, X_test, model_google, True)
                    vocab_size = len(model_google.vocab) + 1
                # possibility #5 - word2vec own model
                elif strat == 'word_to_vec_self':
                    X_train, X_test = word_to_vec(X_train, X_test, model_w2v, False)
                    vocab_size = len(model_w2v.wv.vocab) + 1


                if name == 'NB':
                    try:
                        X_train = X_train.toarray()
                        X_test = X_test.toarray()
                    except:
                        print("already array")
                        #continue

                if model != 'mlp' and model != 'emb' and model != 'con':
                    # Multinomial NB cnanot deal with negative values so they will be scaled if some are present
                    if (name == 'NB') and (len(X_train[X_train<0]) > 0):
                        mms = MinMaxScaler(feature_range=[0,1])
                        mms.fit(X_train)
                        X_train = mms.transform(X_train)
                        X_test = mms.transform(X_test)
                    model.fit(X_train, y_train)
                    pred_y = model.predict(X_test)
                    y_test_score = y_test.copy()

                else:
                    class_weights = class_weight.compute_class_weight('balanced',
                                                                     np.unique(y_train),
                                                                     y_train)
                    class_weights = dict(enumerate(class_weights))

                    y_train, y_test, y_test_score = encode_y(y_train, y_test)


                    if model == 'mlp':
                        model_nn = build_nn(X_train)
                    else:
                        embedded = False
                        conv = False
                        # this is not working due to some dimension problems. Could not fix it due to time limitations
                        if strat == 'word_to_vec_google' or strat == 'word_to_vec_self':
                            #embedded = True
                            accuracy_score_list.append(0)
                            precision_score_list.append(0)
                            recall_score_list.append(0)
                            f1_score_list.append(0)
                            continue

                        if model == 'con':
                            conv = True

                        model_nn = build_embedding_nn(vocab_size, embedding_dim, X_test.shape[1], embedded, conv)
                    #model_nn.summary()
                    model_nn.fit(X_train, y_train, epochs=50, verbose=False, class_weight=class_weights,
                                 validation_data=(X_test, y_test), batch_size=128)

                    # required otherwise scores cannot be calculated due to to_categorical encoding
                    pred_y = model_nn.predict_classes(X_test)
                    y_test = np.argmax(y_test, axis=1)

                    #plot_learning_curves_nn(model_nn)

                accuracy_score_list.append(accuracy_score(y_test_score,pred_y))
                precision_score_list.append(precision_score(y_test_score,pred_y,average='weighted', zero_division=0))
                recall_score_list.append(recall_score(y_test_score,pred_y,average='weighted'))
                f1_score_list.append(f1_score(y_test_score,pred_y,average='weighted'))

            results[str(ind_X)+'_'+strat+'_'+name] = {'acc':np.mean(accuracy_score_list),'prec':np.mean(precision_score_list),
                                       'recall':np.mean(recall_score_list),'f1':np.mean(f1_score_list)}


# STILL TO DO: HYPERPARAMETER OPTIMIZATION ESPECIALLY W.R.T OVERFITTING

df_res = pd.DataFrame(results)
df_res = df_res.transpose()

df_res.to_excel("res_nlp_all_final_all_2.xlsx")