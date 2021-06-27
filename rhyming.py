import nltk
import json
import editdistance


class rhyming:
    def __init__(self):
        try:
            self.arpabet = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            self.arpabet = nltk.corpus.cmudict.dict()
        self.dictionary = self.build_dict('dictionary.json')

    def build_dict(self, path):
        f = open(path, 'r')
        dictionary = json.load(f)
        f.close
        return dictionary

    def check_rhyme(self, word):
        dictionary = self.dictionary
        # for word in dictionary.keys():
        #     word2 = dictionary[word]
            # print(word, word2)
        word2 = dictionary[word]
        # len1 = len(self.arpabet[word][0])
        # len2 = len(self.arpabet[word2][0])
        phoneme1 = self.arpabet[word][0]
        phoneme2 = self.arpabet[word2][0]
        # print(len1, len2)
        # if len1 != len2:
        #     to_append = abs(len2 - len1)
        #     ta = []
        #     for i in range(to_append):
        #         ta.append('X')
            
        #     if(len1 < len2):
        #         phoneme1 = ta + phoneme1
        #     else:
        #         phoneme2 = ta + phoneme2
        
        return phoneme1, phoneme2
            # print(phoneme1, phoneme2)

    def hamming_distance_end_rhymes(self, phoneme1, phoneme2):
        # use edit distance instead
        count = 0
        for (v_1, v_2) in zip(phoneme1, phoneme2):
            if v_1 == v_2:
                count = count + 1
            else:
                count = 0
        return count
    
    # def training():
    #     dictionary = self.dictionary
    #     for word in dictionary:
    #         phenome1, phenome2 = 

rhyming = rhyming()
phenome1, phenome2 = rhyming.check_rhyme('orange')
print(editdistance.eval(phenome1, phenome2))
print((phenome1, phenome2))
print(rhyming.hamming_distance_end_rhymes(phenome1, phenome2))

    # print(arpabet['conviction'])
    # print(arpabet['prediction'])