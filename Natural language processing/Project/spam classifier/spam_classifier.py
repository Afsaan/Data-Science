import pandas as pd

df = pd.read_csv("SMSSpamCollection",sep='\t',names=['Labels','Messages'])

df.head()

from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lemma = WordNetLemmatizer()
new_message = []


for i in range(len(df)):
    step1 = re.sub("[^a-zA-Z]"," ",df['Messages'][i])
    step2 = step1.lower()
    step3 = nltk.word_tokenize(step2)
    step4_stem = [lemma.lemmatize(word) for word in step3 if word not in set(stopwords.words("english"))]
    step5 = " ".join(step4_stem)
    new_message.append(step5)
    

# converting into Document Matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
matrix = tf.fit_transform(new_message).toarray()

#traing our model
X=matrix
y = df.Labels
y = pd.get_dummies(y)
y = y.drop("spam",axis=1)
    
#splitting the model into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)


#traning the model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#checking the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)    

our_message = "you have won alotter and a free entry"
data = [our_message]
new_matrix = tf.transform(data).toarray()

y_new = model.predict(new_matrix)

if y_new==1:
    print("it is ham")
else:
    print("its is spam")


