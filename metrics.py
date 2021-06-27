import editdistance
from scipy.spatial import distance as hammingdistance
from sklearn.metrics import jaccard_score




def hamming_distance(phoneme1, phoneme2):
    # Hamming distance: pad the beginning of array so that they're the same length
    #   Calc hamming distance (from where they differ)

    # lower score means they're more similar
    phon1 = phoneme1
    phon2 = phoneme2
    phon1, phon2 = pad_phoneme(phon1, phon2)
    # print(phoneme1, phoneme2)
    hd = hammingdistance.hamming(phon1, phon2)
    # print(hd)
    return hd

def pad_phoneme(phoneme1, phoneme2):
    len1 = len(phoneme1)
    len2 = len(phoneme2)
    
    phon1 = phoneme1.copy()
    phon2 = phoneme2.copy()
    if len1 > len2:
        
        to_append = len1 - len2
        prev = []
        for i in range(to_append):
            prev.append('X')
        
        phon2[:0] = prev
        
    elif len2 > len1:
        to_append = len2 - len1
        prev = []
        for i in range(to_append):
            prev.append('X')

        phon1[:0] = prev

    return phon1, phon2

def edit_distance(phoneme1, phoneme2):
    # Lower score means they're more similar
    maxi = max(len(phoneme1), len(phoneme2))
    return editdistance.eval(phoneme1, phoneme2)/maxi

def jaccard_similarity(phoneme1, phoneme2):
    phon1 = phoneme1
    phon2 = phoneme2
    phon1, phon2 = pad_phoneme(phon1, phon2)
    
    # print(phoneme1, phoneme2)
    # Here the higher score means they're more similar
    return jaccard_score(phon1, phon2, average = 'weighted')

def longest_common_substring(phoneme1, phoneme2):

    #https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
    answer = []
    len1, len2 = len(phoneme1), len(phoneme2)
    for i in range(len1):
        for j in range(len2):
            lcs_temp=0
            match=[]
            while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and phoneme1[i+lcs_temp] == phoneme2[j+lcs_temp]):
                match.append(phoneme2[j+lcs_temp])
                lcs_temp+=1
            if (len(match) > len(answer)):
                answer = match
    # print(answer)
    maxi = max(len1, len2)
    return len(answer)/maxi

def vowel_constant_match_cmu(phoneme1, phoneme2):
    scores = {'n': 0, 'nv':0.2, 'nc':0.4, '-yv': 0.5, 'yv':0.6, 'yc':0.8, '*yv':1}
    len1, len2 = len(phoneme1)-1, len(phoneme2)-1
    vowels = ['A', 'E', 'I', 'O', 'U']
    score = 0
    while len1 >= 0 and len2>=0:
        phon1 = phoneme1[len1]
        phon2 = phoneme2[len2]

        if any(p in phon1 for p in vowels):
            if any(p in phon2 for p in vowels):
                vow1 = phon1[:len(phon1)-1]
                vow2 = phon2[:len(phon2)-1]
                if(phon1 == phon2):
                    if '1' in phon1 or '2' in phon1:
                        score += scores['*yv']
                    else:
                        score += scores['yv']
                elif(vow1 == vow2):
                    score += scores['-yv']
                else:
                    # print("here")
                    score += scores['nv']
        
        elif not any(p in phon2 for p in vowels):
            if(phon1 == phon2):
                score += scores['yc']
            else:
                score += scores['nc']
        
        # print(score)
        len1 -= 1
        len2 -= 1
    # print(score)
    maxi = max(len(phoneme1), len(phoneme2))
    return score/maxi

def vowel_constant_match_ipa(phoneme1, phoneme2):
    scores = {'n': 0, 'nv':0.2, 'nc':0.4, 'yv':0.6, 'yc':0.8, '*yv':1}
    len1, len2 = len(phoneme1)-1, len(phoneme2)-1
    vowels = ['i', 'ɪ', 'e', 'ɛ', 'æ', 'a', 'ʌ', 'ə', 'u', 'ʊ', 'o', 'ɔ']
    score = 0
    stress = ['ˈ', 'ˌ']
    
    stressed1 = False
    stressed2 = False
    
    while len1>=0 and len2>=0:
        phon1 = phoneme1[len1]
        phon2 = phoneme2[len2]

        if phon1 in stress or phon2 in stress:
            if phon1 in stress and stressed1:
                score += 0.4
            if phon2 in stress and stressed2:
                score +=0.4
            break

        if phon1 in vowels:
            if phon2 in vowels:
                if(phon1 == phon2):
                    stressed1 = True
                    stressed2 = True
                    score += scores['yv']

                else:
                    score += scores['nv']
                    stressed1 = False
                    stressed2 = False

        elif not phon2 in vowels:
            if(phon1 == phon2):
                    score += scores['yc']
            else:
                    score += scores['nc']
        # print(phon1, phon2, score)
        len1 -= 1
        len2 -= 1
        
    max_len = max(len(phoneme1), len(phoneme2))
    #normalizing it
    return score/max_len

    