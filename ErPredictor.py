#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-3-20 上午9:55
  @ Author   : Vodka
  @ File     : ErPredictor.py
  @ Software : PyCharm
"""
from keras.models import model_from_json
from keras.optimizers import Adam
import numpy as np


class ErPredictor:
    def __init__(self):
        print "Er Predictor Initializing"

        # Load the model parameters
        # Here we use the pre-trained model by AskNow
        try:
            model_json_file = 'model/er.json'
            model_weight = 'model/er.h5'
            json_file = open(model_json_file, 'r')
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights(model_weight)
            adam = Adam(lr=0.0001)
            self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        except Exception, e: 
            print e
            sys.exit(1)

        print "Er Predictor Initialized"

    def erpredict(self, chunk_with_position):
        """
        :param chunk_with_position:
        :return:
        """
        try:
            # Get the chunks such as 'Barack'+'Obama' combined
            combined_chunks = []
            for chunk in chunk_with_position:
                surface_start = chunk[0][2]
                end_pos = chunk[0][2]
                combined_phase = []
                for word in chunk:
                    combined_phase.append(word[0])
                    end_pos = word[2] + word[3]
                combined_phase = ' '.join(combined_phase)
                combined_chunks.append((combined_phase, surface_start, end_pos - surface_start))

            # Use the figure to represent the character,
            # thus the word becomes a series of figures,
            # and we use these figure sequence to predict the entity-relation.
            result = []
            for chunk in combined_chunks:
                character_chunk = chunk[0]
                char_dictionary = np.load('model/char_dict.npy').item()
                figure_chunk = [char_dictionary[i] for i in character_chunk]
                prediction = self.model.predict(
                    np.concatenate((np.zeros((270 - len(figure_chunk))), figure_chunk)).reshape(1, 270))
                output = np.argmax(prediction[0])
                if output == 1:
                    result.append(
                        {'chunk': chunk[0], 'surfacestart': chunk[1], 'surfacelength': chunk[2], 'class': 'entity'}
                    )
                else:
                    result.append(
                        {'chunk': chunk[0], 'surfacestart': chunk[1], 'surfacelength': chunk[2], 'class': 'relation'}
                    )
            return result
        except Exception, e:
            print e
            sys.exit(1)


if __name__ == '__main__':
    e = ErPredictor()
    print e.erpredict(
        [[['Who', 'S-NP', 0, 3]], [['the', 'B-NP', 7, 3], ['parent', 'I-NP', 11, 6], ['organisation', 'E-NP', 18, 12]],
         [['Barack', 'B-NP', 34, 6], ['Obama', 'E-NP', 41, 5]], [['is', 'S-VP', 4, 2]]])
