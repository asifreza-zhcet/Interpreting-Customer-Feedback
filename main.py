import joblib
import warnings
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

warnings.filterwarnings('ignore')

def txt_clean(x):
    x = x.lower() #convert to lower case
    x = re.sub('\[.*?\]','', x) #remove [] and anything in between the brackets
    x = re.sub('www\S+|https?\S+', '', x) #remove links
    x = re.sub('\<.*?\>', '', x) #remove html tags
    x = re.sub(f'[{re.escape(punctuation)}]', '',x) #remove punctuations
    x = re.sub('\n','',x) #remove new line
    x = re.sub('\w*\d+\w*','',x) #remove and word containing numbers
    return x

def remove_stop(x):
    words = word_tokenize(x)
    words_no_stop = [word for word in words if word not in sw]
    return ' '.join(words_no_stop)

model_tfidf = joblib.load('./models/model_tfidf.joblib')
model_rand_fo = joblib.load('./models/model_rand_fo.joblib')
model_stem = joblib.load('./models/model_stem.joblib')
model_label = joblib.load('./models/model_label.joblib')

sw = stopwords.words('english')
sw_neg = ["not","don't","aren't","couldn't","didn't","doesn't","hadn't",
          "hasn't","haven't","isn't","mightn't","mustn't","shan't","shouldn't",
          "wasn't","weren't","won't","wouldn't"]
sw = set(sw) - set(sw_neg)
sw = {txt_clean(word) for word in sw}

def predict_senti(text):
    clean_text = txt_clean(text)
    clean_text = remove_stop(clean_text)
    clean_text = model_stem.stem(clean_text)
    X = model_tfidf.transform([clean_text])
    y = model_rand_fo.predict(X)
    z = model_label.inverse_transform(y)
    return z

