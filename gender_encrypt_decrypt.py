import re
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single
import flair
assert flair.__version__=='0.4.2'
from keras.models import load_model
import spacy
import neuralcoref


male_pronouns   = ["he",  "him", "his$", "his", "himself"]
female_pronouns = ["she", "her", "her$", "hers", "herself"]
neutral_pronouns= ["zie", "zim", "zir", "zis", "zieself"]
merged_pronouns = ["he/she", "him/her", "his/her", "his/hers", "himself/herself"]

gender_pronouns_dict = {}
gender_honorific_dict = {}

for (g1,g2,g3,g4) in zip(male_pronouns, female_pronouns, neutral_pronouns, merged_pronouns):
    element = {"male": g1.replace("$",""), "female":g2.replace("$",""), "neutral":g3, "merged":g4}
    gender_pronouns_dict[g1] = gender_pronouns_dict[g2] = gender_pronouns_dict[g3] = gender_pronouns_dict[g4] =element

male_hons   =  ["Mr.", "Mr", "Md.", "Md", "Sir", "Lord", "Mister"]
female_hons =  ["Ms.", "Ms", "Mst.", "Mst", "Madam", "Lady", "Miss"]
neutral_hons = ["Mx.", "Mx", "Mx.", "Mx", "Sir/Madam", "Lord/Lady", "Mister/Miss"]
married_hons = ["Mrs.", "Mrs", "Mst.", "Mst", "Madam", "Lady", "Mis'ess"]
merged_hons =  ["Mr./Ms.", "Mr/Ms", "Md./Mst.", "Md/Mst", "Sir/Madam", "Lord/Lady", "Mister/Miss"]

for (h1,h2,h3,h4,h5) in zip(male_hons, female_hons, neutral_hons, married_hons, merged_hons):
    element = {"male": h1, "female":h2, "neutral":h3, "married_fem":h4, "merged":h5}
    gender_honorific_dict[h1] = gender_honorific_dict[h2] = gender_honorific_dict[h3] = \
    gender_honorific_dict[h4] = gender_honorific_dict[h5] = element


import json

char2idx = 'gender-resources/char2idx.json'
idx2char = 'gender-resources/idx2char.json'
with open(char2idx, 'r') as fp:
    char2idx = json.load(fp)
    
with open(idx2char, 'r') as fp:
    idx2char = json.load(fp)

model_name = 'gender-resources/char_rnn_hsc_model_0.h5'
model = load_model(model_name)

import en_core_web_lg
nlp = en_core_web_lg.load()
neuralcoref.add_to_pipe(nlp)

tagger_ner = SequenceTagger.load('ner')
tagger_pos = SequenceTagger.load('pos')

import numpy as np
from keras.preprocessing import sequence

# Converts a name into vector
def name2vectorTest(name):
    name = name.lower()
    new_name = ""
    for char in name:
      if char in char2idx:
        new_name += char
    chars = list(new_name)
    vector = [ char2idx[c] for c in chars ]
    return np.array(vector)

# Converts names to fixed size tensor
def names2tensorTest(names, maxlen=25):
    namelist = [name2vectorTest(name) for name in names]
    return sequence.pad_sequences(np.array(namelist), maxlen=maxlen)  # root of all troubles

def name2gender(name):
  result = model.predict_classes(np.array(names2tensorTest([name.lower()])))[0][0]
  if result:
    return "male"
  else:
    return "female"
  
def isMale(name):
  result = model.predict_classes(np.array(names2tensorTest([name.lower()])))[0][0]
  return result


def store(name, name_found, Name2Key, Key2Name, num_keys): 
    
    if name_found not in Name2Key:
        #global num_keys
        num_keys+=1
        key = "PER_"+str(num_keys)
        gender = name2gender(name_found)
        alias = None
        element = {"name": name, "key": key, "gender":gender, "alias":alias, "is_alias": False, "alias_to": None}
        Name2Key[name_found] = element
        Key2Name[key] = element
    
    if name not in Name2Key:
        element_alias = {"name": name, "is_alias": True, "alias_to": name_found}
        Name2Key[name] = element_alias
        Name2Key[name_found]["alias"] = name
        
    return Name2Key[name_found]["key"], num_keys

def gender_encrypt(s):
  s = ' '.join(s.split())
  doc = nlp(s)
  tokenized_text = ' '.join([token.text for token in doc])
  oracle = []
  coref2name = {}

  # POS TAG
  sent = Sentence(tokenized_text)
  tagger_ner.predict(sent)
  tagger_pos.predict(sent)
  tagged_list = sent.to_tagged_string().split()
  tokens_pos = []
  pos = []
  count = 0
  for i in range(0,len(tagged_list),2):
      tokens_pos.append(tagged_list[i])
      count = count+1
      pos.append(tagged_list[i+1])
      
      if tagged_list[i].lower() in male_pronouns+female_pronouns:
          oracle.append(2)
      elif tagged_list[i+1] in ['<B-PER/NNP>', '<I-PER/NNP>', '<E-PER/NNP>', '<S-PER/NNP>']:
          oracle.append(4)
      else:
          oracle.append(0)

  # COREFERENCE RESOLUTION 
  if len(doc)!=len(tokens_pos):
    tokens_doc = [token.text for token in doc]
    print("doc:", tokens_doc)
    print("pos:", tokens_pos)
    return None, None
  
  coref_stack = []
  name_stack = []
  for i in range(len(doc)):
      token = doc[i]
      if token._.in_coref:
          coref_stack.append(tokens_pos[i])
          if oracle[i] == 4:
              name_stack.append(tokens_pos[i])
          oracle[i] += 1
      else:
          if len(name_stack) > 0:
              name = ' '.join(name_stack)
              coref = ' '.join(coref_stack)
              #name2coref[name] = coref
              coref2name[coref] = name
              name_stack.clear()
          coref_stack.clear()

  # IF THE SENTENCE DOES NOT END WITH A PERIOD OR SPECIAL CHARACTER
  if len(name_stack) > 0:
      name = ' '.join(name_stack)
      name2coref[name] = ' '.join(coref_stack)
      name_stack.clear()
  coref_stack.clear()

  Name2Key = {}
  Key2Name = {}
  encrypted = []
  num_keys = 0
  i = 0
  while i<len(tokens_pos): 
      #print("Oracle ", i, tokens_pos[i])
      if oracle[i] == 2:
          pronoun = tokens_pos[i].lower()
          if pos[i] == '<PRP$>':
              pronoun+="$"
          encrypted.append(gender_pronouns_dict[pronoun]["merged"])
      elif oracle[i] == 3:
          coref = doc[i]._.coref_clusters[0][0].text
          pronoun = tokens_pos[i].lower()
          if pos[i] == '<PRP$>':
              pronoun+="$"
          if coref in coref2name:
            name_found = coref2name[coref]
            key = Name2Key[name_found]["key"]
            encrypted.append("<|coref|>")
            encrypted.append(gender_pronouns_dict[pronoun]["merged"])
            encrypted.append(key)
          else:
            encrypted.append(gender_pronouns_dict[pronoun]["merged"])
          
      elif oracle[i] in [4,5]:
          if i > 0 and tokens_pos[i-1] in gender_honorific_dict:
            hons = encrypted.pop()
            encrypted.append("<|hons|>")
            encrypted.append(gender_honorific_dict[hons]["merged"])
          if pos[i] == '<S-PER/NNP>':
              name = tokens_pos[i]
          elif pos[i] == '<B-PER/NNP>':
              name = ""
              while True:
                  #print(i, oracle[i])
                  name += tokens_pos[i]
                  if pos[i] == '<E-PER/NNP>':
                      break
                  name += " "
                  i+=1
          
          if oracle[i] == 4:
              key, num_keys = store(name, name, Name2Key, Key2Name, num_keys)
              encrypted.append(key)
          else:
              coref = doc[i]._.coref_clusters[0][0].text
              name_found = coref2name[coref]
              if name == name_found:
                  key, num_keys = store(name, name_found, Name2Key, Key2Name, num_keys)
                  encrypted.append(key)
              else:
                  key, num_keys = store(name, name_found, Name2Key, Key2Name, num_keys)
                  encrypted.append("<|alias|>")
                  encrypted.append(key)
      else:
        encrypted.append(tokens_pos[i])
      i+=1
      
  encrypted_text = ' '.join(encrypted)

  return tokenized_text, Key2Name, encrypted_text


def gender_decrypt(encrypted_text, Key2Name):
  encrypted = encrypted_text.split()
  decrypted = []
  i = 0
  while i != len(encrypted):
    token = encrypted[i]
    if token in Key2Name:
      decrypted.append(Key2Name[token]["name"])
    elif token == '<|alias|>':
      i+=1
      key = encrypted[i]
      if Key2Name[key]["alias"]:
        decrypted.append(Key2Name[key]["alias"])
      else:
        decrypted.append(Key2Name[key]["name"])
    elif token == '<|coref|>':
      startOfSent = (i==0 or encrypted[i-1] == ".")
      pronoun =  encrypted[i+1]
      key = encrypted[i+2]
      gender = Key2Name[key]["gender"].lower()
      decrypted_pronoun = gender_pronouns_dict[pronoun][gender]
      decrypted_pronoun = decrypted_pronoun[0].upper()+decrypted_pronoun[1:] if startOfSent else decrypted_pronoun
      decrypted.append(decrypted_pronoun)
      i+=2
    elif token == "<|hons|>":
      hons = encrypted[i+1]
      key = encrypted[i+2] # NEED CHECK FOR HONS IN ALIAS
      gender = Key2Name[key]["gender"].lower()
      decrypted.append(gender_honorific_dict[hons][gender])
      i+=1
    elif token == "he/she":
        if i==0 or encrypted[i-1] == ".":
            decrypted.append("He/She")
        else:
            decrypted.append("he/she")
    else:
      decrypted.append(token)
    i+=1

  decrypted_text = ' '.join(decrypted)
  #decrypted_text = decrypted_text.replace(" .", ".")
  return decrypted_text
