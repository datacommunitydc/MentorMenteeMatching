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

def readData(data_dir,filename=None):
  # files = [os.path.abspath(os.path.join(data_dir, f)) for f in os.listdir(data_dir)]
  # files = [f for f in files if os.path.isfile(f) and f.endswith('.csv')]
  # files = os.listdir(data_dir)
  for file in os.listdir(data_dir):
    full_name = os.path.abspath(os.path.join(data_dir, file))
    if os.path.isfile(full_name) and file.endswith('.csv') and (filename.lower() in file.lower()):
      return pd.read_csv(full_name)
  return None

def testMentorMenteeCol(column):
  for val in column:
    if ('mentor' in val) or ('mentee' in val):
      return True
  return False

def discoverMentorMenteeCol(data):
  data_columns = list(data)
  columns_str = '; '.join(data_columns).lower()
  if ('mentor' in columns_str) or ('mentee' in columns_str):
    for c, col in enumerate(data_columns):
      if ('mentor' in col) or ('mentee' in col):
        dcol = (c,col); break
  else: # find a good column
    for c,col in enumerate(data_columns):
      if testMentorMenteeCol(data[col].as_matrix()):
        dcol = (c,col); break
  return dcol

def testFullStackCol(column):
  for val in column:
    if ('blah' in val.lower()): # Test against the full stack dictionary
      return True
  return False

def discoverFullStackCol(data):
  data['xx'] = list(range(len(data)))
  data_columns = list(data)
  columns_str = '; '.join(data_columns).lower()
  if ('full stack' in columns_str):
    for c, col in enumerate(data_columns):
      if ('full stack' in col.lower()):
        dcol = (c,col,data); break
  else: # find a good column
    for c,col in enumerate(data_columns):
      if testFullStackCol(data[col].as_matrix()):
        dcol = (c,col,data); break
  return dcol

def extractMentorsMentees(data):
  # mentors = pd.DataFrame([row for row in data.iterrows() if (fuzz.ratio(row[1][cmap[4]], "Mentor")>90)])
  # mentees = pd.DataFrame([row for row in data.iterrows() if (fuzz.ratio(row[1][cmap[4]], "Mentee")>90)])
  mentors = data[data[cmap[4]] == "Mentor"]
  mentees = data[data[cmap[4]] == "Mentee"]
  mentors['xx'] = list(range(len(mentors)))
  mentees['xx'] = list(range(len(mentees)))
  return mentors, mentees

def readFullStack(data,col_name=None):
  if col_name is None:
    col_name = cmap[5]
  stack_rough = list(data[col_name].apply(str))
  stack_list = [val for sl in stack_rough for val in sl.split(',')]
  stack = set(stack_list)
  return stack

def scoreTheMatch(peer1,peer2,field_name):
  return fuzz.ratio(peer1[field_name], peer2[field_name])

def matchMM(mentors,mentees,stack):
  num_mentors = len(mentors)
  num_mentees = len(mentees)
  MM = np.zeros(num_mentors*num_mentees).reshape(num_mentors,num_mentees)
  for mr,mentor in mentors.iterrows():
    for me,mentee in mentees.iterrows():
      MM[mentor['xx'],mentee['xx']] = scoreTheMatch(mentor,mentee,cmap[5])
  return MM.astype(int)

def matchPeers(peers,col_name=None):
  if col_name is None:
    col_name = cmap[5]
  num_peers = len(peers)
  PM = np.zeros(num_peers**2).reshape(num_peers,num_peers)
  for r,peer1 in peers.iterrows():
    for c,peer2 in peers.iterrows():
      PM[peer1['xx'], peer2['xx']] = scoreTheMatch(peer1,peer2,col_name)
  mask = np.zeros(PM.shape,dtype=bool)
  np.fill_diagonal(mask,1)
  PM[mask] = 0
  return PM.astype(int)

def prioritizeMM(MM):
  """ Optimize Mentees for Mentors, and vice versa. """
  # Assume asymmetric score matrix (i.e. mentors and not mentees)
  def testMatches(MT):
    MT = MT.astype(int)
    mentee_matches = np.sum(MT,axis=1)
    if (0 in mentee_matches) | (1 in mentee_matches): # Every Mentee gets two options.
      return False
    else:
      return True

  def tupleMatches(MM,MT):
    """Optimize the matches both ways"""
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

def prioritizeStackInterests(MM,num=3):
  """Assumes a square matrix"""
  assignments = []
  for r in range(MM.shape[0]):
    assignments += [(r,np.argsort(MM[r])[::-1][0:num])]
  return assignments

def full_name(names,xx):
  individual = names[names['xx'] == xx].iloc[0]
  return individual[cmap[2]] + '_' + individual[cmap[3]]

def nameMatches(assignments,owners,assigned):
  Names = []
  for k,assign in enumerate(assignments):
    Names += [[full_name(owners,assign[0]),
              ', '.join([full_name(assigned,ax) for ax in assign[1]])]]
  return Names

def writeAssignmentsToCSV(assignments,column_names=['Your Name','Colleagues'],filename='assignments.csv'):
  acsv = pd.DataFrame(assignments)
  acsv.columns = column_names
  acsv.to_csv(filename)

# def writeAssignmentsToCSV(mentor_assignments,mentee_assignments):
#   mentor_csv = pd.DataFrame(mentor_assignments)
#   mentor_csv.columns = ["Mentors", "Mentees"]
#   mentor_csv.to_csv('mentor_assignments.csv')
#
#   mentee_csv = pd.DataFrame(mentee_assignments)
#   mentee_csv.columns = ["Mentees", "Mentors"]
#   mentee_csv.to_csv('mentee_assignments.csv')

#-----------
def match(filename,mentor_match):
  data = readData(data_dir,filename=filename)

  if mentor_match:
    col,name = discoverMentorMenteeCol(data)
    mentors, mentees = extractMentorsMentees(data)
    stack = readFullStack(data, col_name=name)
    MM = matchMM(mentors, mentees, stack)
    mentor_assignments, mentee_assignments = prioritizeMM(MM)
    mentor_assignments = nameMatches(mentor_assignments,mentors,mentees)
    mentee_assignments = nameMatches(mentee_assignments,mentees,mentors)
    writeAssignmentsToCSV(mentor_assignments,column_names=["Mentors", "Mentees"],filename='mentor_assignments.csv')
    writeAssignmentsToCSV(mentor_assignments,column_names=["Mentees", "Mentors"],filename='mentee_assignments.csv')

  else: # no hierarchy, all are equal
    col,name,data = discoverFullStackCol(data)
    # stack = readFullStack(data,col_name=name)
    MM = matchPeers(data,col_name=name)
    assignments = prioritizeStackInterests(MM,num=3)
    assignments = nameMatches(assignments, data, data)
    writeAssignmentsToCSV(assignments)

  plt.imshow(MM)
  blah = 9
