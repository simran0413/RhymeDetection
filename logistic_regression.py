#%%
import nltk
import json
import editdistance
import random
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


class logistics:
    def __init__(self):
        try:
            self.arpabet = nltk.corpus.cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            self.arpabet = nltk.corpus.cmudict.dict()
        self.dictionary = self.build_dict('dictionary.json')
        self.clf = LogisticRegression()
        self.linear = LinearRegression()
        shuffled = self.shuffle()
        inputs, outputs = self.inputs_outputs(shuffled)
        data = self.get_phonemes(inputs, outputs)
        data_using_alphabets = self.using_alphabets(inputs, outputs)
        # print("Using alphabets)
        # self.train(data_using_alphabets)

        print("Using IPA")
        self.ipa_us, self.ipa_uk = self.read_ipa_files()
        data_using_IPA = self.get_phonemes_ipa(inputs, outputs)
        self.train(data_using_IPA)



        # print("Using phonemes")
        # self.train(data)


    def build_dict(self, path):
        f = open(path, 'r')
        dictionary = json.load(f)
        f.close
        # print(dictionary)
        return dictionary
    
    def read_ipa_files(self):
        file_us = "ipa_en_US.csv"
        file_uk = "ipa_en_UK.csv"
        ipa_us = pd.read_csv(file_us)
        ipa_uk = pd.read_csv(file_uk)
        return ipa_us, ipa_uk

        # result = ipa_us.loc[ipa_us['word']=='me']
        # print(type(result['ipa']))

    def get_phonemes_ipa(self, inputs, outputs):
        ipa_us = self.ipa_us
        ipa_uk = self.ipa_uk
        data = {'word1': [], 'word2':[], 'max_length':[], 'edit_distance':[], 'phoneme1':[], 'phoneme2':[], 'output':[]}
        for (tups, out) in zip(inputs, outputs):
        # for i in range(10):
            # tups = inputs[i]
            # out = outputs[i]
            phoneme1 = []
            phoneme2 = []
            try:
                phoneme_us = ipa_us.loc[ipa_us['word']== tups[0]]['ipa'].to_string()
                # print(phoneme_us['ipa'].to_string())
                strip = phoneme_us.split()
                print(strip)
                phoneme1.append(strip[1])
            except:
                print("US phoneme not found", tups[0])
            try:
                phoneme_uk = ipa_uk.loc[ipa_uk['word']== tups[0]]['ipa'].to_string()
                strip = phoneme_uk.split()
                phoneme1.append(strip[1])
            except:
                print("UK phoneme not found", tups[0])

            try:
                phoneme_us = ipa_us.loc[ipa_us['word']== tups[1]]['ipa'].to_string()
                strip = phoneme_us.split()
                phoneme2.append(strip[1])
            except:
                print("US phoneme not found", tups[1])
            try:
                phoneme_uk = ipa_uk.loc[ipa_uk['word']== tups[1]]['ipa'].to_string()
                strip = phoneme_uk.split()
                phoneme2.append(strip[1])
            except:
                print("UK phoneme not found", tups[1])
            
            # print(phoneme1)
            if(phoneme1 and phoneme2):
                min_ed = editdistance.eval(phoneme1[0], phoneme2[0])
                p1 = phoneme1[0]
                p2 = phoneme2[0]
                for phon1 in phoneme1:
                    for phon2 in phoneme2:
                        ed = editdistance.eval(phon1, phon2)
                        if ed < min_ed:
                            min_ed = ed
                            p1 = phon1
                            p2 = phon2
                
                max_len = max(len(p1), len(p2))
                # print(tups[0], p1, tups[1], p2, min_ed, max_len)
                # print(tups[0], tups[1], p1, p2, min_ed, max_len, out)
                # check all pronunciations and keep the one that has the smallest distance. change max length accordingly
                data['phoneme1'].append(p1)
                data['phoneme2'].append(p2)
                data['max_length'].append(max_len)
                data['edit_distance'].append(min_ed)
                data['output'].append(out)
                data['word1'].append(tups[0])
                data['word2'].append(tups[1])
            # else:
                # print(tups[0], tups[1])
            # print(data)
        return data

        # print(ipa_us.loc['word']=='me')
        # ipa_us_dict = ipa_us.to_dict('records')
        # ipa_uk_dict = ipa_uk.to_dict('records')
        # # print(type(ipa_us_dict))
        # ipa_dict_empty = {}
        # ipa_dict_us = self.to_dict(ipa_dict_empty, ipa_us_dict)
        # ipa_dict_both = self.to_dict(ipa_dict_us, ipa_uk_dict)
        # # with open('isa_dict.json', 'w') as fp:
        # #     json.dump(ipa_dict_both, fp)
        # # print(ipa_us_dict[0])
        # j = json.dumps(ipa_dict_both, indent=4)
        # f = open('ipa_dict.json', 'w')
        # print >> f, j
        # f.close()
    
    # def to_dict(self, ipa_dict, ipa_list):
    #     for elem in ipa_list:
    #         word = elem['word']
    #         isa = elem['ipa']
    #         if word not in ipa_dict.keys():
    #             ipa_dict[word] = []
    #         ipa_dict[word].append(isa)
    #     return ipa_dict
    
    def shuffle(self):
        dic = self.dictionary
        inputs = []
        # outputs = []
        for out in dic.keys():
            for word1 in dic[out]:
                word2 = dic[out][word1]
                tups = (word1, word2, out)
                inputs.append(tups)
                # outputs.append(out)
        # print(inputs)
        # random.shuffle(inputs)
        # print(shuffled)
        return inputs

    def inputs_outputs(self, shuffled):
        inputs = []
        outputs = []
        for tups in shuffled:
            inputs.append((tups[0], tups[1]))
            outputs.append(tups[2])
        
        # print(inputs, outputs)
        return inputs, outputs
    
    def get_phonemes(self, inputs,outputs):
        # phonemes = []
        # max_length = []
        # edit_distance = []
        data = {'word1': [], 'word2':[], 'max_length':[], 'edit_distance':[], 'phoneme1':[], 'phoneme2':[], 'output':[]}
        for (tups, out) in zip(inputs, outputs):
            phoneme1 = self.arpabet[tups[0]]
            phoneme2 = self.arpabet[tups[1]]
            min_ed = editdistance.eval(phoneme1[0], phoneme2[0])
            p1 = phoneme1[0]
            p2 = phoneme2[0]
            for phon1 in phoneme1:
                for phon2 in phoneme2:
                    ed = editdistance.eval(phon1, phon2)
                    if ed < min_ed:
                        min_ed = ed
                        p1 = phon1
                        p2 = phon2

            max_len = max(len(p1), len(p2))
            # print(tups[0], tups[1], p1, p2, min_ed, max_len, out)
             # check all pronunciations and keep the one that has the smallest distance. change max length accordingly
            data['phoneme1'].append(p1)
            data['phoneme2'].append(p2)
            data['max_length'].append(max_len)
            data['edit_distance'].append(min_ed)
            data['output'].append(out)
            data['word1'].append(tups[0])
            data['word2'].append(tups[1])

        # print(data)
            # print(phoneme1, phoneme2)
            # phonemes.append((phoneme1, phoneme2))
            # max_length.append(max_len)
            # edit_distance.append(ed)
        # inputs = (max_length, edit_distance)
        return data

    def using_alphabets(self, inputs, outputs):
        data = {'word1': [], 'word2':[], 'max_length':[], 'edit_distance':[], 'output':[]}
        for (tups, out) in zip(inputs, outputs):
            edit_dist = editdistance.eval(tups[0], tups[1])
            max_length = max(len(tups[0]), len(tups[1]))
            data['word1'].append(tups[0])
            data['word2'].append(tups[1])
            data['max_length'].append(max_length)
            data['edit_distance'].append(edit_dist)
            data['output'].append(out)
        
        # print(data)
        return data

    def train(self, data):
        df = pd.DataFrame(data)
        input = np.asarray(df[['edit_distance', 'max_length']])
        # input = np.asarray(df[['phoneme1', 'phoneme2', 'max_length', 'edit_distance']])
        output = np.asarray(df['output'])
        colors = {'True': 'red', 'False': 'blue'}
        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(4,4))
        plt.xlabel("Edit Distance")
        plt.ylabel("Max Length")
        ax = fig.add_subplot(111)
        ax.scatter(df['edit_distance'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
        plt.show()
        fig.savefig("test_scatter.png")


        scores = []
        coefs = []
        ## Get average over 1000 rounds
        
        for x in range(1000):
            input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = 0.2, shuffle = True)      
            clf = self.clf
            clf.fit(input_train, output_train)
            output_results = clf.predict(input_test)
            score = clf.score(input_test, output_test)
            coef = clf.coef_
            scores.append(score)
            coefs.append(coef)
        
        # print(output_test)
        print("Logistic Regression")
        # print(scores)
        print(np.mean(scores))
        print(np.mean(coefs, axis = 0))


        ## Plotting graph
        input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = 0.2, shuffle = True)      
        clf = self.clf
        clf.fit(input_train, output_train)
        output_results = clf.predict(input_test)
        score = clf.score(input_test, output_test)
        coef = clf.coef_
        scores.append(score)
        coefs.append(coef)
        print('log weights: ', clf.coef_)
        print('log accuracy with test', clf.score(input_test, output_test))
        plt.plot(output_results)
        plt.plot(output_test)

        ## Linear regression
        # linear = self.linear
        # linear.fit(input_train, output_train)
        # output_results = linear.predict(input_val)
        # score = linear.score(input_val, output_val)
        # print('log weights: ', linear.coef_)
        # print('log accuracy with val: ', score)
        # print('log accuracy with test', linear.score(input_test, output_test))

    def to_use(self, word1, word2):
        phoneme1 = self.arpabet[word1]
        phoneme2 = self.arpabet[word2]
        min_ed = editdistance.eval(phoneme1[0], phoneme2[0])
        p1 = phoneme1[0]
        p2 = phoneme2[0]
        for phon1 in phoneme1:
            for phon2 in phoneme2:
                ed = editdistance.eval(phon1, phon2)
                if ed < min_ed:
                    min_ed = ed
                    p1 = phon1
                    p2 = phon2

        max_len = max(len(p1), len(p2))
        input = np.array([min_ed, max_len]).reshape(1, -1)

        clf = self.clf
        predicted_output = clf.predict(input)
        print(predicted_output)


log = logistic()
# log.to_use('dog', 'bottle')
# log.read_isa_files()




# %%
