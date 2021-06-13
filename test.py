
import ktrain
import pandas as pd
import re

p1 = ktrain.load_predictor('model_BERT')

df = pd.read_csv('test.csv')

def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence
	
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
	
X = []
sentences = list(df['tweet'])
for sen in sentences:
    X.append(preprocess_text(sen))
	
label = p1.predict(X)
print("The text:",X,"   Result:",label[0])
