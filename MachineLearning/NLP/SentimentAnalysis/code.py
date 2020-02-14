import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle as pkl

imdb = pd.read_csv('imdb_labelled.txt',sep='\t',header=None)
amazon = pd.read_csv('amazon_cells_labelled.txt',sep='\t',header=None)
yelp = pd.read_csv('yelp_labelled.txt',sep='\t',header=None)

dataset = pd.DataFrame()
dataset = dataset.append(imdb)
dataset = dataset.append(amazon)
dataset = dataset.append(yelp)

dataset.columns = ['Review','Sentiment']
print(dataset.head())

def textProcessing(dataset):
    tokens = []
    for i in range(len(dataset)):
        tokens.append(word_tokenize(dataset['Review'].iloc[i].lower()))
    eng_stopwords = stopwords.words('english')
    eng_stopwords.extend([',','?','\\','!','@','$','#','..','...','.','-','_'])
    words = []
    for i in range(len(tokens)):
        temp_list = []
        for token in tokens[i]:
            if token not in eng_stopwords:
                temp_list.append(token)
        words.append(temp_list)

    wnet = WordNetLemmatizer()
    for i in range(len(words)):
        for j in range(len(words[i])):
            words[i][j] = wnet.lemmatize(words[i][j],pos='v')

    for i in range(len(words)):
        words[i] = " ".join(words[i])

    return words

wordsList = textProcessing(dataset)

tfidf = TfidfVectorizer()
vect = tfidf.fit_transform(wordsList)
words_array = vect.toarray()

x_train,x_test,y_train,y_test = train_test_split(words_array,dataset['Sentiment'],test_size=0.25)

reg = LogisticRegression()
reg.fit(x_train,y_train)

y_pred = reg.predict(x_test)
reg_acc = accuracy_score(y_test,y_pred)
print("Accuracy of logistic regression is",reg_acc)

gauss = GaussianNB()
gauss.fit(x_train,y_train)

y_pred = gauss.predict(x_test)
reg_acc = accuracy_score(y_test,y_pred)
print("Accuracy of Gaussian NB is",reg_acc)

multinb = MultinomialNB()
multinb.fit(x_train,y_train)

y_pred = multinb.predict(x_test)
reg_acc = accuracy_score(y_test,y_pred)
print("Accuracy of Multinomial NB is",reg_acc)

###### Confusion Matrix #######
multi_cm = confusion_matrix(y_test,y_pred)
print(multi_cm)

coef = multinb.coef_

file = open('coef.pkl','wb')
pkl.dump(multinb,file)
file.close()

file = open('vectorizer.pkl','wb')
pkl.dump(tfidf,file)
file.close()
