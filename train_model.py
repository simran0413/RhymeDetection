#%%
from nltk.grammar import sdg_demo
import nltk
from pandas.core.frame import DataFrame
from sklearn.utils import shuffle
from logicstic_reg_2 import logistic
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import json
from scipy.special import expit
from metrics import hamming_distance, edit_distance, jaccard_similarity, longest_common_substring, vowel_constant_match_cmu, vowel_constant_match_ipa



class train:
    def __init__(self):
        
        # dictionary_true, dictionary_false = self.build_dict('data/test_true.json', 'data/test_false.json')
        # self.dictionary_true = dictionary_true
        # self.dictionary_false = dictionary_false
        self.log = logistic()
        print("getting inputs and outputs")
        self.inputs, self.outputs = self.log.inputs_outputs()
        # dictionary_true, dictionary_false = self.build_dict('data/test_true_24750.json', 'data/test_false_250.json')
        # self.test_inputs, self.test_outputs = self.log.inputs_outputs_test_set(dictionary_true, dictionary_false)
        print("got inputs and outputs")
        self.clf = LogisticRegression(penalty='l1', solver= 'liblinear')

    def build_dict(self, path_true, path_false):
        f = open(path_true, 'r')
        dictionary_true = json.load(f)
        f.close
        f = open(path_false, 'r')
        dictionary_false = json.load(f)
        f.close()
        return dictionary_true, dictionary_false

    def data(self):
        # Get the words, phonemes, all distances, and outputs
        log = self.log
        inputs = self.inputs
        outputs = self.outputs
        # inputs = self.test_inputs
        # outputs = self.test_outputs
        # phonemes_half = self.phoneme_half
        # print(phonemes_full)
        # dictionary_true = self.dictionary_true
        # dictionary_false = self.dictionary_false
        # arpabet = self.arpabet
        print("arranging data in df")
        data = {'word1': [], 'word2':[], 'phoneme1':[], 'phoneme2':[], 'edit_distance':[], 'hamming_distance':[], 'jaccard_similarity':[], 'longest_common_substring':[], 'weighed_phonemes':[], 'max_length': [],'output':[]}
        i =0
        for (tups, out) in zip(inputs, outputs):
            # print(i)
            i += 1
            # elem = phonemes_half[(tups[0], tups[1])]
            # phoneme1 = elem[0]
            # phoneme2 = elem[1]
            # cmu_ipa = elem[2]
            phoneme1, phoneme2, cmu_ipa = log.get_phonemes(tups[0], tups[1])
            print(tups[0], tups[1])
            print(phoneme1, phoneme2)
            hd = hamming_distance(phoneme1, phoneme2)
            # hd = 1.0
            ed = edit_distance(phoneme1, phoneme2)
            js = jaccard_similarity(phoneme1, phoneme2)
            # js = 1.0
            lcs = longest_common_substring(phoneme1, phoneme2)
            if cmu_ipa == "cmu":
                vcm = vowel_constant_match_cmu(phoneme1, phoneme2)
            elif cmu_ipa == "ipa":
                vcm = vowel_constant_match_ipa(phoneme1, phoneme2)
            max_len = max(len(phoneme1), len(phoneme2))
            data['word1'].append(tups[0])
            data['word2'].append(tups[1])
            data['phoneme1'].append(phoneme1)
            data['phoneme2'].append(phoneme2)
            data['edit_distance'].append(ed)
            data['hamming_distance'].append(hd)
            data['jaccard_similarity'].append(js)
            data['longest_common_substring'].append(lcs)
            data['weighed_phonemes'].append(vcm)
            data['max_length'].append(max_len)
            data['output'].append(out)
        # print(data)
        print("data arranged")
        df = pd.DataFrame(data)
        df.to_csv('data/med_half_new.csv')
        return data



    def train(self, data):
        # Train model with all similarity metrics
        # df = pd.DataFrame(data)
        df = pd.read_csv('data/med_data_half.csv')
        input = np.asarray(df[['edit_distance', 'hamming_distance', 'jaccard_similarity', 'longest_common_substring', 'weighed_phonemes']])
        output = np.asarray(df['output'])
        colors = {'True': 'red', 'False': 'blue'}
        fig = plt.figure(figsize=(4,4))
        plt.xlabel("hamming")
        plt.ylabel("jaccard")
        ax = fig.add_subplot(111)
        ax.scatter(df['hamming_distance'], df['jaccard_similarity'], c=df['output'].map(colors), alpha = 0.2)
        # ax.scatter(df['edit_distance'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
        # ax.scatter(df['hamming_distance'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
        # ax.scatter(df['jaccard_similarity'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
        # ax.scatter(df['weighed_phonemes'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
        # ax.scatter(df['longest_common_substring'], df['max_length'], c=df['output'].map(colors), alpha = 0.2)
        
        plt.show()
        fig.savefig("scatter_hamming_jaccard_half_phonemes.png")

        # ## Plotting graph
        
        # input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = 0.2, shuffle = True)      
        # clf = self.clf
        # clf.fit(input_train, output_train)
        # output_results_proba = clf.predict_proba(input_test)
        # output_results= clf.predict(input_test)
        # score = clf.score(input_test, output_test)
        # coef = clf.coef_
        # # best_params = clf.best_params_
        # # scores.append(score)
        # # coefs.append(coef)
        # print('log weights: ',coef)
        # print('log accuracy with test', score)
        # # plt.plot(output_results)
        # # print(output_results)
        # results = {'type':"all metrics, l1, full phoneme", 
        # 'log_weights': coef.tolist(), 
        # 'accuracy': score, 
        # # 'predicted_results': output_results.tolist(), 
        # # 'actual_results': output_test.tolist(), 
        # # 'predict_proba': output_results_proba.tolist()
        # }
     
        # # print(best_params)
        # plt.plot(output_test)
        # with open('results.json', 'r+') as file:
        #     data = json.load(file)
        #     temp = data['results']
        #     # data.update(results)
        #     temp.append(results)
        #     # file.seek(0)
        # self.write_json(data, 'results.json')

    def write_json(self, data, file):
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    
    def train_sgd(self):
        print("Reading data")
        df = pd.read_csv('data/med_half_new.csv')
        # df_test = pd.read_csv('data/test_data_one_percent_false_full.csv')
        # df = pd.DataFrame(data_)
        print("Finished reading data")
        metrics = ['edit_distance', 'hamming_distance', 'jaccard_similarity', 'longest_common_substring', 'weighed_phonemes']
        inputs = np.asarray(df[metrics])
        output = np.asarray(df['output'])
        # test_inputs = np.asarray(df_test[metrics])
        # test_output = np.asarray(df_test[['output']])
        # print(test_output)
        # k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
        # scores = cross_val_score(sgdc, input_train, output_train, scoring='accuracy', cv= k_fold)
        # cv_score = (np.mean(scores), np.std(scores))
        # cv_test_score = sgdc.score(input_test, output_test)
        testing_scores = []
        # testing_scores = []
        coefs = []
        print("setting up classifier")
        sgdc = SGDClassifier(loss='log', penalty='l1', alpha= 0.001, shuffle=True, random_state=42)
        print("splitting data")
        # input_train, input_test, output_train, output_test = train_test_split(inputs, output, test_size = 0.2, shuffle = True)
        
        for i in range(10):
            print(i)
            input_train, input_test, output_train, output_test = train_test_split(inputs, output, test_size = 0.2, shuffle = True)
            # x_train, x_val, y_train, y_val = train_test_split(input_train, output_train, test_size= 0.2, shuffle = True)
            # x_train, x_val, y_train, y_val = train_test_split(inputs, output, test_size= 0.2, shuffle = True)
            # split training into training and validation
            # sgdc.fit(x_train, y_train)
            sgdc.fit(input_train, output_train)
            testing_score = sgdc.score(input_test, output_test)
            # add scores and coefs to lists, get mean and std
            print("Testing score: ", testing_score)
            testing_scores.append(testing_score)
            coef = sgdc.coef_
            coefs.append(coef.tolist())
            print(sgdc.coef_)
        # testing_score = sgdc.score(input_test, output_test)
        # testing_score = sgdc.score(test_inputs, test_output)
        # print("Testing score: ", testing_score)
        output_predictions = sgdc.predict(input_test)
        # output_predictions = sgdc.predict(test_inputs)
        output_predicted_proba = sgdc.predict_proba(input_test)
        # output_predicted_proba = sgdc.predict_proba(test_inputs)
        # print(output_predicted_proba)
        lr_proba = output_predicted_proba[:, 1]
        fpr, tpr, threshold = roc_curve(output_test, lr_proba)
        # print(fpr, tpr)
        # plt.plot(fpr, tpr)
        # plt.show()
        # plt.savefig("results/roc.png", bbox_inches = 'tight')

        tn, fp, fn, tp = confusion_matrix(output_test, output_predictions).ravel()
        # tn, fp, fn, tp = confusion_matrix(test_output, output_predictions).ravel()
        cm = [int(tn), int(fp), int(fn), int(tp)]
        # acc_score = accuracy_score(output_test, output_predictions)
        log_l = log_loss(output_test, output_predicted_proba)
        # log_l = log_loss(test_output, output_predicted_proba)
        # print(tn, fp, fn, tp)
        mean_score_testing = np.mean(testing_scores)
        # mean_score_testing = np.mean(testing_scores)
        std_testing = np.std(testing_scores)
        # std_testing = np.std(testing_scores)
        print("Mean: ", mean_score_testing)

        
        # print("Coef: ", coef)
        results = {'type':"jaccard and lcs, l1, full phoneme, using sgd, alpha= 0.001, shuffle=True, random_state= 42, 10-fold, imbalanced dataset one percent false", 
        'cross_val': True,
        'log_weights': coefs, 
        # 'cv_score': cv_score
        # 'cv_training_score': cv_training_score,
        # 'cv_testing_score': cv_test_score,
        # 'ave_training_score': np.mean(scores),
        # 'std_training_score': np.std(scores),
        'training_score': testing_scores,
        # 'test_score': testing_score,
        'mean_training': mean_score_testing,
        # 'mean_testing': mean_score_testing,
        'std_val': std_testing,
        # 'std_testing': std_testing,
        'confusion_matrix': cm,
        # 'accuracy_score': acc_score,
        'log_loss': log_l

        }

        # results = {'type':"jaccard and lcs, half phoneme, balanced", 
        # 'fpr': fpr.tolist(),
        # 'tpr': tpr.tolist(),
        # 'theshold': threshold.tolist()
        # }
        # print(results)

        with open('results/roc_results_half_balanced.json', 'r+') as file:
            data = json.load(file)
            temp = data['results']
            # data.update(results)
            temp.append(results)
            # file.seek(0)
        # self.write_json(data, 'results/roc_results_half_balanced.json')

        # fig = plt.figure(figsize=(4,4))
        # # plt.title("Confusion Matrix, half phonemes, all metrics")
        # plt.xlabel("Actual output")
        # plt.ylabel("Predicted output")
        # plot_confusion_matrix(sgdc, test_inputs, test_output)
        # plt.show()
        # imbalanced_numoftrue_num0ffalse
        # plt.savefig("results/confusion_matrix/imbalanced_one_percent_false/full_phoneme_jaccard_lcs.png")

tm = train()
# data = tm.data()
# tm.train(data)
tm.train_sgd()
# %%
