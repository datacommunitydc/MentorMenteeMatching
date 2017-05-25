import os, operator
import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import matplotlib.pyplot as plt

data_dir = "./data/"
cmap = {
  0:"Timestamp",
  1:"Email Address",
  2:"First Name",
  3:"Last Name",
  4:"Mentor or Mentee?",
  5:"Which part of the Full Stack would you like to focus on?",
  6:"Please describe your proficiencies or your needs in 2-4 sentences (no tl;dr please)"}

def readData(data_dir):
  files = [os.path.abspath(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]
  files = [f for f in files if os.path.isfile(f) and f.endswith('.csv')]
  return pd.read_csv(os.path.join(data_dir, files[0]))

def extractMentorsMentees(data):
  # mentors = pd.DataFrame([row for row in data.iterrows() if (fuzz.ratio(row[1][cmap[4]], "Mentor")>90)])
  # mentees = pd.DataFrame([row for row in data.iterrows() if (fuzz.ratio(row[1][cmap[4]], "Mentee")>90)])
  mentors = data[data[cmap[4]] == "Mentor"]
  mentees = data[data[cmap[4]] == "Mentee"]
  mentors['xx'] = list(range(len(mentors)))
  mentees['xx'] = list(range(len(mentees)))
  return mentors, mentees

def readStack(data):
  stack_rough = list(data[cmap[5]].apply(str))
  stack_list = [val for sl in stack_rough for val in sl.split(',')]
  stack = set(stack_list)
  return stack

def matchMM(mentors,mentees,stack):
  num_mentors = len(mentors)
  num_mentees = len(mentees)
  MM = np.zeros(num_mentors*num_mentees).reshape(num_mentors,num_mentees)
  for mr,mentor in mentors.iterrows():
    for me,mentee in mentees.iterrows():
      MM[mentor['xx'],mentee['xx']] = fuzz.ratio(mentor[cmap[5]],mentee[cmap[5]])
  return MM

#-----------
data = readData(data_dir)
mentors, mentees = extractMentorsMentees(data)
stack = readStack(data)
MM = matchMM(mentors,mentees,stack)
plt.imshow(MM)