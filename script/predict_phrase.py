import numpy as np
import re
from keras.optimizers import Adam
from keras.models import model_from_json

class Predictor:
    def __init__(self):
        self.model_j = './model/er.json'
        self.model_wgt = './model/er.h5'
        self.max_len = 283

        #load the model
        self.json_file = open(self.model_j, 'r')
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.model = model_from_json(self.model_json)
        self.model.load_weights(self.model_wgt)

        print("Loaded model from disk")

        adam = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


    def clean_str(self,string, TREC=False):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Every dataset is lower cased except for TREC
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def predict_phrase(self,phrase):
       phrase_clean = self.clean_str(phrase)
       #load the dictionary
       char_dict = np.load('./model/char_dict.npy').item()
       phrase_clean = [char_dict[char] for char in phrase_clean]
       prediction = self.model.predict(np.concatenate((np.zeros((270-len(phrase_clean))), phrase_clean)).reshape(1,270))
       pred = np.argmax(prediction[0])
       return 'R' if pred == 0 else 'E'

    def predict_cln_phrs(self,clean_phrase):
       tags = []
       print clean_phrase
       for phr in clean_phrase:
            print phr
            tags.append(self.predict_phrase(phr))
       return tags

if __name__ == '__main__':
   #phrase = sys.argv[1:]
   p = Predictor()