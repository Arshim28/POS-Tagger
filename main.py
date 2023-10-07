import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score as ps
np.random.seed(42)

import nltk
nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown


class PosHMM:
    def __init__(self):
        self.word_tag = None
        self.trans = None
    
    def training(self, train):
        word_tag = {}
        for i in train:
            for j in i:
                if j[1] not in word_tag:
                    word_tag[j[1]] = {}
                if j[0].lower() not in word_tag[j[1]]:
                    word_tag[j[1]][j[0].lower()] = 0
                word_tag[j[1]][j[0].lower()]+=1
        for x in word_tag:
            tot = 0
            for j in word_tag[x]:
                tot += word_tag[x][j]
            word_tag[x]['Total'] = tot
        
        
        trans = {}
        trans['^'] = {}
        word_tag['^'] = {}
        word_tag['^']['Total'] = len(train)
        
        for i in train:
            if i[0][1] not in trans['^']:
                trans['^'][i[0][1]] = 0
            trans['^'][i[0][1]]+=1
            if len(i) <= 1:
                continue
            for x in range(len(i)-1):
                if i[x][1] not in trans:
                    trans[i[x][1]] = {}
                if i[x+1][1] not in trans[i[x][1]]:
                    trans[i[x][1]][i[x+1][1]] = 0
                trans[i[x][1]][i[x+1][1]]+=1
        
        self.word_tag = word_tag
        self.trans = trans
    
    def lex_prob(self, word, tag):
        if word in self.word_tag[tag]:
            return self.word_tag[tag][word]/self.word_tag[tag]['Total']
        else:
            return 1e-9

    def trans_prob(self, curr_tag, prev_tag, total):
        if curr_tag not in self.trans[prev_tag]:
            return 1e-9
        else:
            return self.trans[prev_tag][curr_tag] / total
    
    def predict(self, sentence = None, sentence_train = None):
        if sentence_train is not None:
            n = 0
            for i in sentence_train:
                n += 1
            sentence = sentence_train
        else:
            words = str.split(sentence, ' ')
            sentence = []
            for x in words:
                sentence.append((x, 0))
            n = len(sentence)
        
        curr_tag = [['^', 0]]
        new_tag = []
        #curr_max = -1e10000
        max_entries = 4
        curr_entries = 1
        
        possible_tags = ['NOUN', 'VERB','ADV', 'ADJ', 'PRT', 'DET', 'CONJ', '.', 'ADP', 'PRON', 'X', 'NUM']
        ans_tag = []
        
        for i in range(n):
            word = sentence[i][0].lower()
            for poss in curr_tag:
                for j in possible_tags:
                    prob = poss[-1]
                    prev_tag = poss[-2]
                    copy_poss = poss.copy()
                    
                    prob += math.log(self.lex_prob(word, j)) + math.log(self.trans_prob(j, prev_tag, self.word_tag[j]['Total']))
                    copy_poss[-1] = j
                    copy_poss.append(prob)
                    new_tag.append(copy_poss)
            curr_tag = sorted(new_tag, key = lambda r:r[-1], reverse = True)[:max_entries]
            #print(word, '->', curr_tag)
            new_tag = []
        final_tag = curr_tag[0][:-1]
        final_tag = final_tag[1:]
        #print(len(final_tag))
        return final_tag
    
    def get_confusion_matrix(self, all_preds, all_vals):
        possible_tags = ['NOUN', 'VERB','ADV', 'ADJ', 'PRT', 'DET', 'CONJ', '.', 'ADP', 'PRON', 'X', 'NUM']
        cm = confusion_matrix(all_vals, all_preds, labels = possible_tags)
        return cm

    def know_vals(self, d):
        l = []
        for i in d:
            l.append(i[1])
        return l

    def get_scores(self, confusion_matrix):
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        possible_tags = ['NOUN', 'VERB','ADV', 'ADJ', 'PRT', 'DET', 'CONJ', '.', 'ADP', 'PRON', 'X', 'NUM']
        
        for i in range(12):
            correct = confusion_matrix[i, i]
            false_pos = np.sum(confusion_matrix[:, i]) - correct
            false_neg = np.sum(confusion_matrix[i, :]) - correct
            precision = correct / (correct + false_pos)
            recall = correct / (correct + false_neg)
            tag = possible_tags[i]
            precision_scores[tag] = precision
            recall_scores[tag] = recall

            # Calculate F1 score for each class
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores[tag] = f1
        
        return precision_scores, recall_scores, f1_scores
    
    def train(self):
        brown_corpus = brown.tagged_sents(tagset='universal')
        brown_corpus = np.array(brown_corpus, dtype = object)
        possible_tags = ['NOUN', 'VERB','ADV', 'ADJ', 'PRT', 'DET', 'CONJ', '.', 'ADP', 'PRON', 'X', 'NUM']

        kf = KFold(n_splits=5, shuffle = True)
        split = kf.split(brown_corpus)

        ind_prec = [0,0,0,0,0]
        ind_recall = [0,0,0,0,0]
        ind_f1score = [0,0,0,0,0]
        cnt = 0
        cm = [0,0,0,0,0]

        f1_score = f2_score = prec = recall = fhalf_score = 0

        for train, test in split:
            #train_data, test_data = brown_corpus[train], brown_corpus[test]
            train_data, test_data = brown_corpus[train], brown_corpus[test]
            self.training(train_data)
            all_vals = []
            all_preds = []
            suc = 0
            tot = 0
            for d in test_data:
                vals = self.know_vals(d)
                pred = self.predict(sentence_train = d)
                if(vals == pred):
                    suc += 1
                tot += 1
                #print(len(vals), len(pred))
                all_vals.extend(vals)
                all_preds.extend(pred)
            print(suc/tot)
            cm[cnt] = self.get_confusion_matrix(all_preds, all_vals)
            f1_score += fbeta_score(all_vals, all_preds, beta = 1, average = 'weighted')
            fhalf_score += fbeta_score(all_vals, all_preds, beta = 0.5, average = 'weighted')
            f2_score += fbeta_score(all_vals, all_preds, beta = 2, average = 'weighted')
            prec += ps(all_vals, all_preds, average = 'weighted')
            recall += recall_score(all_vals, all_preds, average = 'weighted')
            print(cm[cnt])
            ind_prec[cnt], ind_recall[cnt], ind_f1score[cnt] = self.get_scores(cm[cnt])
            cnt += 1

        f1_score /= 5.0
        fhalf_score /= 5.0
        prec /= 5.0
        recall /= 5.0
        f2_score /= 5.0

        print('Overall scores : ')
        print(f'F1 score - {f1_score}')
        print(f'F2 score - {f2_score}')
        print(f'F0.5 score - {fhalf_score}')
        print(f'Precision score - {prec}')
        print(f'Recall score - {recall}')

        print()
        print('Overall Individual Scores:')
        iprec = {}
        irecall = {}
        if1score = {}
        for i in possible_tags:
            iprec[i] = irecall[i] = if1score[i] = 0
            for j in range(5):
                iprec[i] += ind_prec[j][i]
                irecall[i] += ind_recall[j][i]
                if1score[i] += ind_f1score[j][i]
            iprec[i]/=5.0
            if1score[i]/=5.0
            irecall[i] /= 5.0

        print(iprec)
        print(irecall)
        print(if1score)

        return cm, possible_tags
    
    def plot_matrix(self, cm, possible_tags):
        confusion_matrix = (cm[0] + cm[1] + cm[2] + cm[3] + cm[4])//5

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')

        # Set axis labels and title
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        plt.title('Confusion Matrix Heatmap')
        ax.set_xticklabels(possible_tags)
        ax.set_yticklabels(possible_tags)

        # Show the heatmap
        plt.show()

        confusion_matrix = np.array(confusion_matrix, dtype = 'float64')
        s = np.sum(confusion_matrix, axis = 1)
        s *= 1.0
        for i in range(12):
            for j in range(12):
                confusion_matrix[i][j] = round(confusion_matrix[i][j]/s[i], 2)
        print(confusion_matrix)

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

        # Set axis labels and title
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        plt.title('Confusion Matrix Heatmap')
        ax.set_xticklabels(possible_tags)
        ax.set_yticklabels(possible_tags)

        # Show the heatmap
        plt.show()
    



if __name__=="__main__":
    pos = PosHMM()
    cm, possible_tags = pos.train()
    pos.plot_matrix(cm, possible_tags)
