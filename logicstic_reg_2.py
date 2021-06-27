#%%
import nltk
import json
# import editdistance
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance as hammingdistance
from sklearn.metrics import jaccard_score

class logistic:
    def __init__(self):
        try:
            self.arpabet = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            self.arpabet = nltk.corpus.cmudict.dict()
        # self.dictionary_true = dictionary_true
        # self.dictionary_false = dictionary_false
        # self.dictionary_true, self.dictionary_false = self.build_dict('data/clean_dict_true_1.json', 'data/clean_dict_false_1.json')
        self.dictionary_true, self.dictionary_false = self.build_dict('data/med_true.json', 'data/med_false.json')
        # self.dictionary_true, self.dictionary_false = self.build_small_dict('data/clean_dict_true_1.json', 'data/clean_dict_false_1.json')
        self.ipa_us, self.ipa_uk = self.read_ipa_files()

    
    def build_dict(self, path_true, path_false):
        f = open(path_true, 'r')
        dictionary_true = json.load(f)
        f.close
        f = open(path_false, 'r')
        dictionary_false = json.load(f)
        f.close()
        return dictionary_true, dictionary_false

    def build_small_dict(self, path_true, path_false):
        dictionary_true ={}
        dictionary_false = {}
        f= open(path_true, 'r')
        true = json.load(f)
        f.close()
        for i in range(19000, 43750):
            print("True ", i, "/44750")
            key = list(true.keys())[i]
            # print(key)
            min_len = min(5, len(true[key]))
            dictionary_true[key] = random.sample(true[key], min_len)
            print("dictionary", dictionary_true[key])
        with open('data/test_true_24750.json', 'w') as fp:
            json.dump(dictionary_true, fp, indent=4)

        f = open(path_false, 'r')
        false = json.load(f)
        f.close()
        for i in range(250):
            print("False ", i, "/250")
            key = list(false.keys())[i]
            min_len = min(5, len(false[key]))
            dictionary_false[key] = random.sample(false[key], min_len)
            
        with open('data/test_false_250.json', 'w') as fp:
            json.dump(dictionary_false, fp, indent=4)
        # print(dictionary_true)
        # return dictionary_true, dictionary_false


    def read_ipa_files(self):
        file_us = "ipa_en_US.csv"
        file_uk = "ipa_en_UK.csv"
        ipa_us = pd.read_csv(file_us)
        ipa_uk = pd.read_csv(file_uk)
        return ipa_us, ipa_uk

    def inputs_outputs(self):
        # med dict: 248905 pairs
        dictionary_true, dictionary_false = self.dictionary_true, self.dictionary_false
        inputs = []
        outputs = []
        count = 0
        # phonemes_half = {}
        for word1 in dictionary_true.keys():
            for word2 in dictionary_true[word1]:
                inputs.append((word1, word2))
                outputs.append("True")
                # phoneme1, phoneme2, cmu_ipa = self.get_phonemes(word1, word2)
                # phonemes_half[(word1, word2)] = [phoneme1, phoneme2, cmu_ipa]
                # print(phonemes_full[(word1, word2)])
        for word1 in dictionary_false.keys():
            for word2 in dictionary_false[word1]:
                inputs.append((word1, word2))
                outputs.append("False")
                # phoneme1, phoneme2, cmu_ipa = self.get_phonemes(word1, word2)
                # phonemes_half[(word1, word2)] = [phoneme1, phoneme2, cmu_ipa]
        
        # print(phonemes_full)
        print(len(outputs))
        return inputs, outputs

    
    def inputs_outputs_test_set(self, dictionary_true, dictionary_false):
        # med dict: 248905 pairs
        inputs = []
        outputs = []
        count = 0

        for word1 in dictionary_true.keys():
            for word2 in dictionary_true[word1]:
                inputs.append((word1, word2))
                outputs.append("True")
                
        for word1 in dictionary_false.keys():
            for word2 in dictionary_false[word1]:
                inputs.append((word1, word2))
                outputs.append("False")
                
        print(len(outputs))
        return inputs, outputs


    def get_phonemes(self, word1, word2):
        phoneme1 = []
        phoneme2 = []
        arpabet = self.arpabet
        # try:
        #     arpabet = nltk.corpus.cmudict.dict()
        # except LookupError:
        #     nltk.download('cmudict')
        #     arpabet = nltk.corpus.cmudict.dict()
        # print("here")
        if word1 in arpabet.keys() and word2 in arpabet.keys():
           
            phoneme1 = arpabet[word1][0]
            phoneme2 = arpabet[word2][0]
            index1 = self.get_stress_cmu(phoneme1)
            index2 = self.get_stress_cmu(phoneme2)
            phon1 = phoneme1[index1:]
            phon2 = phoneme2[index2:]
            # print("cmu", phon1, phon2)
            # half phoneme
            return phon1, phon2, "cmu"
            # full phoneme
            # print(word1, word2)
            # print(phoneme1, phoneme2)
            # return phoneme1, phoneme2, "cmu"
            

        else:
        
            phoneme1, phoneme2 = self.get_phonemes_ipa(word1, word2)
            if(phoneme1 and phoneme2):
                index1 = self.get_stress_ipa(phoneme1)
                index2 = self.get_stress_ipa(phoneme2)
                phon1 = phoneme1[0][index1:]
                phon2 = phoneme2[0][index2:]
                # print("ipa", list(phon1), list(phon2))
                # half phoneme
                return list(phon1), list(phon2), "ipa"
                # full phoneme
                # return list(phoneme1[0]), list(phoneme2[0]), "ipa"
            
        print("none", word1, word2)
        return [], []

    def get_phonemes_ipa(self, word1, word2):
        ipa_us = self.ipa_us
        ipa_uk = self.ipa_uk
        phoneme1 = []
        phoneme2 = []

        if not ipa_us[ipa_us['word']==word1].empty and not ipa_us[ipa_us['word']==word2].empty:
            phoneme_us = ipa_us.loc[ipa_us['word']== word1]['ipa'].to_string()
            strip = phoneme_us.split()
            phoneme1.append(strip[1])
            # if not ipa_us[ipa_us['word']==word2].empty:
            phoneme_us = ipa_us.loc[ipa_us['word']== word2]['ipa'].to_string()
            strip = phoneme_us.split()
            phoneme2.append(strip[1])
        
        # if not(phoneme1 and phoneme2):
        # elif not ipa_uk[ipa_uk['word']==word1].empty and not ipa_uk[ipa_uk['word']==word2].empty:
        #     phoneme_uk = ipa_uk.loc[ipa_uk['word']== word1]['ipa'].to_string()
        #     strip = phoneme_uk.split()
        #     phoneme1.append(strip[1])

        #     # if not ipa_uk[ipa_uk['word']==word2].empty:
        #     phoneme_uk = ipa_uk.loc[ipa_uk['word']== word2]['ipa'].to_string()
        #     strip = phoneme_uk.split()
        #     phoneme2.append(strip[1])
        # if not ipa_us[ipa_us['word']==word2].empty:
        #     phoneme_us = ipa_us.loc[ipa_us['word']== word2]['ipa'].to_string()
        #     strip = phoneme_us.split()
        #     phoneme2.append(strip[1])
        # elif not ipa_uk[ipa_uk['word']==word2].empty:
        #     phoneme_uk = ipa_uk.loc[ipa_uk['word']== word2]['ipa'].to_string()
        #     strip = phoneme_uk.split()
        #     phoneme2.append(strip[1])

        return phoneme1, phoneme2

        
    
    def get_stress_cmu(self, phoneme):
        i = len(phoneme) - 1
        # print(phoneme)
        stress = False
        secondary_stress = False
        secondary_stress_index = 0
        # print(i)
        while i>=0 and stress == False:
            if('1' in phoneme[i]):
                stress = True
                return i
            elif('2' in phoneme[i]):
                if not secondary_stress:
                    # (print("here"))
                    secondary_stress = True
                    secondary_stress_index = i
        
            i = i-1
        # print(secondary_stress_index)
        return secondary_stress_index
        

    def get_stress_ipa(self, phoneme):
        i = len(phoneme[0]) - 1
        stress = False
        secondary_stress = False
        secondary_stress_index = 0
        while i>=0 and stress == False:
            if("ˈ" in phoneme[0][i]):
                stress = True
                return i
            elif("ˌ" in phoneme[0][i]):
                if not secondary_stress:
                    secondary_stress = True
                    secondary_stress_index = i
        
            i = i-1


        return secondary_stress_index



    # def hamming_distance(self, phoneme1, phoneme2):
    #     # Hamming distance: pad the beginning of array so that they're the same length
    #     #   Calc hamming distance (from where they differ)

    #     # lower score means they're more similar
    #     phon1, phon2 = self.pad_phoneme(phoneme1, phoneme2)
    #     # print(phoneme1, phoneme2)
    #     hd = hammingdistance.hamming(phon1, phon2)
    #     return hd
    
    # def pad_phoneme(self, phoneme1, phoneme2):
    #     len1 = len(phoneme1)
    #     len2 = len(phoneme2)
        
    #     phon1 = phoneme1
    #     phon2 = phoneme2
    #     if len1 > len2:
            
    #         to_append = len1 - len2
    #         prev = []
    #         for i in range(to_append):
    #             prev.append('X')
            
    #         phon2[:0] = prev
            
    #     elif len2 > len1:
    #         to_append = len2 - len1
    #         prev = []
    #         for i in range(to_append):
    #             prev.append('X')

    #         phon1[:0] = prev

    #     return phon1, phon2

    # def edit_distance(self, phoneme1, phoneme2):
    #     # Lower score means they're more similar
    #     maxi = max(len(phoneme1), len(phoneme2))
    #     return editdistance.eval(phoneme1, phoneme2)/maxi

    # def jaccard_similarity(self, phoneme1, phoneme2):
    #     phon1, phon2 = self.pad_phoneme(phoneme1, phoneme2)
        
    #     # print(phoneme1, phoneme2)
    #     # Here the higher score means they're more similar
    #     return jaccard_score(phon1, phon2, average = 'weighted')

    # def longest_common_substring(self, phoneme1, phoneme2):

    #     #https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
    #     answer = []
    #     len1, len2 = len(phoneme1), len(phoneme2)
    #     for i in range(len1):
    #         for j in range(len2):
    #             lcs_temp=0
    #             match=[]
    #             while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and phoneme1[i+lcs_temp] == phoneme2[j+lcs_temp]):
    #                 match.append(phoneme2[j+lcs_temp])
    #                 lcs_temp+=1
    #             if (len(match) > len(answer)):
    #                 answer = match
    #     # print(answer)
    #     maxi = max(len1, len2)
    #     return len(answer)/maxi

    # def vowel_constant_match_cmu(self, phoneme1, phoneme2):
    #     scores = {'n': 0, 'nv':0.2, 'nc':0.4, '-yv': 0.5, 'yv':0.6, 'yc':0.8, '*yv':1}
    #     len1, len2 = len(phoneme1)-1, len(phoneme2)-1
    #     vowels = ['A', 'E', 'I', 'O', 'U']
    #     score = 0
    #     while len1 >= 0 and len2>=0:
    #         phon1 = phoneme1[len1]
    #         phon2 = phoneme2[len2]

    #         if any(p in phon1 for p in vowels):
    #             if any(p in phon2 for p in vowels):
    #                 vow1 = phon1[:len(phon1)-1]
    #                 vow2 = phon2[:len(phon2)-1]
    #                 if(phon1 == phon2):
    #                     if '1' in phon1 or '2' in phon1:
    #                         score += scores['*yv']
    #                     else:
    #                         score += scores['yv']
    #                 elif(vow1 == vow2):
    #                     score += scores['-yv']
    #                 else:
    #                     # print("here")
    #                     score += scores['nv']
            
    #         elif not any(p in phon2 for p in vowels):
    #             if(phon1 == phon2):
    #                 score += scores['yc']
    #             else:
    #                 score += scores['nc']
            
    #         # print(score)
    #         len1 -= 1
    #         len2 -= 1
    #     # print(score)
    #     maxi = max(len(phoneme1), len(phoneme2))
    #     return score/maxi
    
    # def vowel_constant_match_ipa(self, phoneme1, phoneme2):
    #     scores = {'n': 0, 'nv':0.2, 'nc':0.4, 'yv':0.6, 'yc':0.8, '*yv':1}
    #     len1, len2 = len(phoneme1)-1, len(phoneme2)-1
    #     vowels = ['i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ʌ', 'ə', 'u', 'ʊ', 'o', 'ɔ']
    #     score = 0
    #     stress = ['ˈ', 'ˌ']
        
    #     stressed1 = False
    #     stressed2 = False
        
    #     while len1>=0 and len2>=0:
    #         phon1 = phoneme1[len1]
    #         phon2 = phoneme2[len2]

    #         if phon1 in stress or phon2 in stress:
    #             if phon1 in stress and stressed1:
    #                 score += 0.4
    #             if phon2 in stress and stressed2:
    #                 score +=0.4
    #             break

    #         if phon1 in vowels:
    #             if phon2 in vowels:
    #                 if(phon1 == phon2):
    #                     stressed1 = True
    #                     stressed2 = True
    #                     score += scores['yv']

    #                 else:
    #                     score += scores['nv']
    #                     stressed1 = False
    #                     stressed2 = False

    #         elif not phon2 in vowels:
    #             if(phon1 == phon2):
    #                     score += scores['yc']
    #             else:
    #                     score += scores['nc']
    #         # print(phon1, phon2, score)
    #         len1 -= 1
    #         len2 -= 1
            
    #     max_len = max(len1+1, len2+1)
    #     #normalizing it
    #     return score/max_len

# dictionary_true, dictionary_false = build_dict('data/med_true.json', 'data/med_false.json')
log = logistic()
# log.build_small_dict('data/clean_dict_true_1.json', 'data/clean_dict_false_1.json')
# inputs, outputs = log.inputs_outputs()
# for input in inputs:
#     log.get_phonemes(input[0], input[1] )
# log.get_phonemes('zucchinis', 'payees')
# log.inputs_outputs()
# musket, musketeer = log.get_phonemes_ipa('musket', 'mustachioed')
# print(musket, musketeer)
# i = log.get_stress_ipa(musket)
# j = log.get_stress_ipa(musketeer)
# print(musket[0][i:])
# print(musketeer[0][j:])
# self.arpabet['lord']
# phon1, phon2, x = log.get_phonemes('jellied', 'bleed')
# print(log.vowel_constant_match_cmu(['G', 'EH1', 'R', 'IY0', 'AH0', 'S'], ['EH2', 'AO1', 'R', 'IY0', 'AH1', 'S']))
# phon1,phon2 = list('abcdefg'), list('bcacdefg')
# print(log.longest_common_substring(phon1, phon2))
# phon1, phon2, type_phoneme = log.get_phonemes('conquer', 'blabber')
# log.vowel_constant_match_ipa(list('ˈɡɫɛɹiəs_'), list('ˈfɛɹiəs_'))
# print(log.longest_common_substring(phon1, phon2))
# log.get_stress_cmu(['N', 'AH0', 'F', 'EH2', 'R', 'IY0', 'AH0', 'S'])
# log.get_false()
# log.longestSubstringFinder("abcdefg", "bcacdefg")
# log.hamming_distance(['L', 'O', 'P', 'E', 'S'], ['E', 'E'])
# %%
