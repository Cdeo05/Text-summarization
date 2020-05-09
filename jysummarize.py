#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import nltk
from keywords import keyword
import numpy as np

import spacy

word_embeddings = {}
f = open('glove.6B/glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()


string="""Students in Kashmir are keeping up with their academics by taking to the online mode of education amid the coronavirus-induced lockdown, even as the clamour for the restoration of high-speed internet services has grown in view of the difficulties faced by them to access the study material.
However, to salvage the academic session, schools in the valley took to various means, including online classes, to impart education to the students. 

Teachers from both the government as well as private schools have started delivering online lectures through various platforms like WhatsApp, Zoom and Google Classroom among others.

Irfan Ahmad, a government school teacher in Ganderbal district in central Kashmir, uses WhatsApp and Zoom applications daily to connect with and deliver lessons to his students.

"It is the only option we have to replace the physical classes. I use these applications on a daily basis to deliver my lectures to the students and to try to answer their questions," Ahmad said. He said the students have already been affected due to the situation since August last year and there was a need to salvage the academic session so that they do not lose much.

Touqeer Javaid, a student of Class 7 at a private school in Ganderbal, logs on to Zoom application at 11 am daily to connect with his classmates and teachers. "We start the class at 11 am. A few of my classmates, who have smartphones, are there and our teacher comes online to deliver lessons," Javaid said. He said while it does not feel like an actual class, it was important to keep the education going. "Education is important and in these times of competition, we can hardly afford to miss our lessons. At least, we are getting something," he added.

Sehar Jan, a student of class 6 at a private school in Srinagar, said apart from the video-conferencing application Zoom, her school also uses WhatsApp to deliver lessons and assignments to the students. "We get regular assignments from the school which we work on and then submit back to the school. It keeps us involved at home as well as helping in our education," she said.

However, some parents complain that the schools were putting the students under tremendous pressure by way of too many assignments. Recently, an audio clip went viral on social media where a parent complained that the schools were putting the students under too much pressure.

"This is just a formality by the schools. They send too many assignments on a daily basis. It is not humane. They are putting too much pressure on the children as well as their parents. They are giving them more work than they actually do at the schools," the parent complained.

Assignments are not the only issue of complaint by the students or the parents.

Many of them lament the slow network speed which hamper the online classes and delivery of assignments. "The internet speed at 2G is very low and it becomes very difficult to attend the online classes. Sometimes, we do not even see the teacher clearly and often we miss whatever he says," Irtiqa, a student, said.

A private school owner in the city here acknowledged the difficulties faced by the students as well as teachers and went on to say that the online classes were just for the sake of formality. The problems are more in far flung or rural areas where broadband internet penetration is very low and the people have to rely on erratic 2G mobile internet.

"I have a broadband connection at home which makes my life a little easier while delivering lectures. However, many students in my school face problems due to slow network speed. Most of the times, I do not understand what they say," Ahmad, the government school teacher in Ganderbal district, said.

In a tweet recently, principal secretary School Education department, government of Jammu and Kashmir, Asgar Samoon acknowledged that low internet speed was hampering online education as the department released funds for distributing books among the students of Classes 1 to 8 in the union territory.

"Kids have no ipads/no access to desktops; cant download text books as internet is slow; Rs 2030.87 lac released for 1st-8th classes; DSEJ Rs 1105.44 lac & DSEk Rs. 925.43 lac; kmr div/ distributed books; remaining Jmu districts-dsej/ ceos/zeos to expedite in 2 wks (sic)," Mr Samoon said.

Private Schools Association Jammu and Kashmir (PSAJK) has filed a petition in the Supreme Court contending that the lack of 4G connectivity for internet in Jammu and Kashmir is infringing the fundamental right to education.

PSAJK president G N Var said while about 70 per cent of 2,675 private schools in the valley have started online classes for their students, they were facing problems due to connectivity issues.

To deal with the problems due to internet speeds, Directorate of School Education, Kashmir, is starting educational broadcast (audio classes) for the students of the valley, the officials said. They said the broadcast is being organized in collaboration with All India Radio, Srinagar.

Director, School Education, Kashmir, Mohammad Younis Malik said these platforms have been kept available for the children so that they do not feel isolated and can continue with their studies while staying at their home due to the lock down."""
df = pd.DataFrame([string],columns=['string_values'])

# print(df)
# print(df['string_values'][0])




from nltk.tokenize import sent_tokenize
sentences = []
for s in df['string_values']:
  sentences.append(sent_tokenize(s))
# for x in sentences:
#   print(x)
sentences = [y for x in sentences for y in x]
# for x in sentences:
#   print('/n')
#   print(x)




clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
# print(pd.Series(sentences))

clean_sentences = [s.lower() for s in clean_sentences]

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new



clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]




sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

sim_mat = np.zeros([len(sentences), len(sentences)])



from sklearn.metrics.pairwise import cosine_similarity
for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)


ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# print(ranked_sentences)
s=""

for i in range(5):
  # print(ranked_sentences[i][1])
  # print()
  # l.append(ranked_sentences[i][1])
  # "".join()
  s+=ranked_sentences[i][1]+" "

print(s)
# nlp=spacy.load('en_core_web_sm')
# doc=nlp(s)


# sen = [sent.string.strip() for sent in doc.sents]
# x=""
# for i in sen:
#   x+=i+' '
# print(x)
keyword.keywords(s)

