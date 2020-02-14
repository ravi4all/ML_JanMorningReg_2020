import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle as pkl

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

def testData(review):
    dataset = pd.DataFrame({"Review":[review]})
    wordsList = textProcessing(dataset)
    file = open('vectorizer.pkl','rb')
    tfidf = pkl.load(file)
    vect = tfidf.transform(wordsList)
    words_array = vect.toarray()
    file = open('coef.pkl','rb')
    multiNb = pkl.load(file)
    prediction = multiNb.predict(words_array)
    return prediction
