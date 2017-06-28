import  numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

class Stemmer_Tokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer() #init stemer
    def __call__(self, doc):
        tokenizer = RegexpTokenizer(r'\w+') # init tokenizer
        tokens = tokenizer.tokenize(doc) #remove puntuation points and tokenize
        #stemming and delete stopwords
        return [self.wnl.stem(t) for t in tokens if not t in stopwords.words('english')]
'''
1003 0.907281303921
1503 0.955760998239
2003 0.979782826262
3003 0.998108346577
3503 1.0
'''
def preprocess():
    #load files
    documents = load_files(container_path = 'filesk', encoding = 'utf8')
    #binary bag of words
    bin_vect = CountVectorizer(tokenizer=Stemmer_Tokenizer(), strip_accents = 'ascii', lowercase = True, max_df = 4999, min_df = 2, binary = True)
    #term frequency bag of word
    frec_vect = TfidfVectorizer(tokenizer=Stemmer_Tokenizer(), strip_accents = 'ascii', lowercase = True, max_df = 4999, min_df = 2)
    #split documents for test and training
    x_train, x_test, y_train, y_test = train_test_split(documents.data, documents.target, test_size=1000)
    #compute binary bag of words
    x_bin_train = bin_vect.fit_transform(x_train)
    x_bin_test = bin_vect.transform(x_test)
    #compute term frequency bag of words
    x_frec_train = frec_vect.fit_transform(x_train)
    x_frec_test = frec_vect.transform(x_test)

    return x_bin_train, x_bin_test, x_frec_train, x_frec_test, y_train, y_test

def load_data():
    print "loading data"
    npzfile = np.load('data_preproc.npz')
    x_bin_train, x_bin_test = npzfile['arr_0'], npzfile['arr_1']
    x_frec_train, x_frec_test = npzfile['arr_2'], npzfile['arr_3']
    y_train, y_test = npzfile['arr_4'], npzfile['arr_5']
    return x_bin_train, x_bin_test, x_frec_train, x_frec_test, y_train, y_test

def evaluate_BernoulliNB(x_train, y_train, x_test, y_test):
    print "BernoulliNB"
    #params for gridSearch
    params = {'alpha' : [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
    #gridSearch with 3 folds
    gridSearch = GridSearchCV(BernoulliNB(), params, cv = 3)
    gridSearch.fit(x_train, y_train)
    #train classifier with best params
    clf = BernoulliNB(**gridSearch.best_params_).fit(x_train, y_train)
    print "Params", gridSearch.best_params_
    print "Accuracy", clf.score(x_test, y_test)

def evaluate_MultiNB(x_train, y_train, x_test, y_test):
    print "MultiNB"
    #params for gridSearch
    params = {'alpha' : [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
    #gridSearch with 3 folds
    gridSearch = GridSearchCV(MultinomialNB(), params, cv = 3)
    gridSearch.fit(x_train, y_train)
    #train classifier with best params
    clf = MultinomialNB(**gridSearch.best_params_).fit(x_train, y_train)
    print "Params", gridSearch.best_params_
    print "Accuracy", clf.score(x_test, y_test)

def rf_evaluation(x_train, y_train, x_test, y_test):
    print "Random Forest"
    #params for gridSearch
    params = {'n_estimators' : [100, 200, 400, 800], 'max_features': [50, 54, 60]}
    #gridSearch with 3 folds
    gridSearch = GridSearchCV(RandomForestClassifier(random_state = 1), params, cv = 3, n_jobs = -1)
    gridSearch.fit(x_train, y_train)
    #train classifier with best params
    clf = RandomForestClassifier(**gridSearch.best_params_).fit(x_train, y_train)
    print "Params", gridSearch.best_params_
    print "Accuracy", clf.score(x_test, y_test)

def gb_evaluation(x_train, y_train, x_test, y_test):
    print "Gradient Boosting Machine"
    #params for gridSearch
    params = {'n_estimators' : [30, 70, 100], 'learning_rate' : [0.1, 0.05], 'max_depth': [5]}
    #gridSearch with 3 folds
    gridSearch = GridSearchCV(GradientBoostingClassifier(random_state = 1), params, cv = 3, n_jobs = -1)
    gridSearch.fit(x_train, y_train)
    #train classifier with best params
    clf = GradientBoostingClassifier(**gridSearch.best_params_).fit(x_train, y_train)
    print "Params", gridSearch.best_params_
    print "Accuracy", clf.score(x_test, y_test)

def svm_evaluation(x_train, y_train, x_test, y_test):
    print "Support Vector Machine"
    #params for gridSearch
    params = {'C' : [2**(-5), 2**(0), 2**(5), 2**(10)], 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
    #gridSearch with 3 folds
    gridSearch = GridSearchCV(SVC(random_state = 1, kernel = 'rbf', decision_function_shape = 'ovr'), params, cv = 3, n_jobs = -1)
    gridSearch.fit(x_train, y_train)
    #train classifier with best params
    clf = SVC(**gridSearch.best_params_).fit(x_train, y_train)
    print "Params", gridSearch.best_params_
    print "Accuracy", clf.score(x_test, y_test)

def number_Component(x_frec_train):
    #compute pca with max number of components
    pca = TruncatedSVD(n_components = x_frec_train.shape[1] - 1)
    pca.fit(x_frec_train)
    variance = 0.0
    for i in range(len(pca.explained_variance_ratio_)):
        #accumulate the variance
        variance += pca.explained_variance_ratio_[i]
        if variance >= 0.99:
            print "pca result", i, variance
            #refit pca with the correct n_components
            pca = TruncatedSVD(n_components = i).fit(x_frec_train)
            print pca.explained_variance_ratio_.sum()
            return pca
    return pca

def best_classifier(x, y):
    print "Best classifer"
    #params for gridSearch
    params = {'C' : [2**(-5), 2**(0), 2**(5), 2**(10)], 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
    #gridSearch with 3-folds
    gridSearch = GridSearchCV(SVC(random_state = 1, kernel = 'rbf', decision_function_shape = 'ovr'), params, cv = 3, n_jobs = -1)
    gridSearch.fit(x, y)
    #print best results
    print "Best Params", gridSearch.best_params_
    print "Best Accuracy", gridSearch.best_score_
    #create final classifier with best params
    return SVC(**gridSearch.best_params_).fit(x, y)

if __name__ == '__main__':
    #preprocess data
    x_bin_train, x_bin_test, x_frec_train, x_frec_test, y_train, y_test = preprocess()

    #evaluate BernoulliNB
    evaluate_BernoulliNB(x_bin_train, y_train, x_bin_test, y_test)

    #evaluate MultiNB
    evaluate_MultiNB(x_frec_train, y_train, x_frec_test, y_test)

    #fit pca with the 0.99 of variance
    pca = number_Component(x_frec_train)

    #apply pca to data
    x_frec_train = pca.transform(x_frec_train)
    x_frec_test = pca.transform(x_frec_test)

    #evaluate svm
    svm_evaluation(x_frec_train, y_train, x_frec_test, y_test)

    #evalutate RandomForestClassifier
    rf_evaluation(x_frec_train, y_train, x_frec_test, y_test)

    #evaluate GradientBoostingClassifier
    gb_evaluation(x_frec_train, y_train, x_frec_test, y_test)

    #train best classifier
    best_classifier(np.vstack((x_frec_train, x_frec_test)), np.vstack((y_train.reshape((-1, 1)), y_test.reshape((-1, 1)) )).reshape(-1))
