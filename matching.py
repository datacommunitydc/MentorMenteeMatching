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
  return MM.astype(int)

def prioritizeMM(MM,mentors,mentees):
  def testMatches(MT):
    MT = MT.astype(int)
    mentee_matches = np.sum(MT,axis=1)
    if (0 in mentee_matches) | (1 in mentee_matches): # Every Mentee gets two options.
      return False
    else:
      return True

  def tupleMatches(MM,MT):
    mentor_assignments = []
    mentee_assignments = []
    for mr in range(MT.shape[0]):
      sorted = np.argsort(MM[mr][MT[mr]])[::-1]
      mentor_assignments += [(mr,sorted[0:3])]
    ME = np.transpose(MT)
    MM = np.transpose(MM)
    for me in range(ME.shape[0]):
      sorted = np.argsort(MM[me][ME[me]])[::-1]
      mentee_assignments += [(me,sorted[0:2])]
    return mentor_assignments,mentee_assignments

  maxMM = np.amax(MM)
  minMM = np.amin(MM)
  step = 5
  thresholds = list(reversed(range(minMM,maxMM,step))) # Decrease in 5% increments
  for T in thresholds:
    # Find the indices of the matrix that are above this threshold, and test the results
    MT = MM > T
    if testMatches(MT):
      break
  return tupleMatches(MM,MT)

#-----------
data = readData(data_dir)
mentors, mentees = extractMentorsMentees(data)
stack = readStack(data)
MM = matchMM(mentors,mentees,stack)
mentor_matches,mentee_matches = prioritizeMM(MM,mentors,mentees)
plt.imshow(MM)
mentor_assignments, mentee_assignments = prioritizeMM(MM,mentors,mentees)
blah = 9
