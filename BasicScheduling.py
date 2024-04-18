# BasicScheduling.py is meant to parse the applicant data into easily usable components
# Input csv file: 
#   - Name
#   - Major needs to be standardized
#   - Interview times cannot have commas in the name

# To run this program, you have to have python installed on your computer. 
# You may have to install some other packages (pandas, random)
# You run the command:
#       python3 .\BasicScheduling.py "<csv file>"
# Example:
#       python3 .\BasicScheduling.py ".\Applications.csv" 

import sys
import pandas as pd
import random
import numpy as np
from pulp import *

num_interviews = 27
# This is the name of the column that contains availability in the csv file.
name_col = "Applicant"
availability_col = "Interviews availability"
major_col = "Major"
# These are the dates and times from the csv file. Make sure they don't have a comma!
day1 = [
        "Monday April 1st - 6:00-7:00 pm", 
        "Monday April 1st - 7:00-8:00 pm", 
        "Monday April 1st - 8:00-9:00 pm"]
day2 = [
        "Tuesday April 2nd - 6:00-7:00 pm", 
        "Tuesday April 2nd - 7:00-8:00 pm", 
        "Tuesday April 2nd - 8:00-9:00 pm"]
day3 = [
        "Wednesday April 3rd - 6:00-7:00 pm",
        "Wednesday April 3rd - 7:00-8:00 pm",
        "Wednesday April 3rd - 8:00-9:00 pm"]
days = [day1, day2, day3]

# Input (should be cvs file)
apps_file = sys.argv[1]
data = pd.read_csv(apps_file)

# Grab 27 applicants from the file
apps = data.head(n=num_interviews)

# This will randomize the applicants grabbed.
# apps = data.sample(n=num_interviews)
initial_num_cols = apps.shape[1]
for day in days:
    for time in day:
        col = apps.shape[1]
        apps.insert(col, time, 0)

# Go through the applicants and extract times
for index, applicant in apps.iterrows():
    availability = applicant[availability_col].split(", ")
    for time in availability:
        apps.loc[index, time] = 1

# print(apps.head(n=10))

# When turning this into arrays, we can make multiple different arrays
A = apps.iloc[:, initial_num_cols:].to_numpy()
# print(A)

# Go through the applicants and extract majors
# TODO: fix the funky .loc warning
# interview_num_cols = apps.shape[1]
# for index, applicant in apps.iterrows():
#     majors = applicant[major_col].split(", ")
#     for major in majors:
#         apps.loc[index, major] = 1
# M = np.nan_to_num(apps.iloc[:, interview_num_cols:].to_numpy())
# print(M)

model = LpProblem("Scheduling_Problem", LpMaximize)

# Variables
x = LpVariable.dicts("Interview", [(i,j) for i in range(len(A)) for j in range(len(A[0]))], cat='Binary')

# Objective Function
model += lpSum([x[i,j]*A[i,j] for i in range(len(A)) for j in range(len(A[0]))])

# Constraints
for i in range(len(A)):
        model += lpSum([x[i,j]*A[i,j] for j in range(len(A[0]))]) <= 1
for j in range(len(A[0])):
        model += lpSum([x[i,j]*A[i,j] for i in range(len(A))]) <= 3

# Solve the problem
model.solve()

# Print the results
print("Status:", LpStatus[model.status])
schedule = np.zeros((len(A), len(A[0])))
for i in range(len(A)):
     for j in range(len(A[0])):
          if (x[i,j].varValue == 1): schedule[i, j] = 1
print(schedule)   

# for var in model.variables():
#     if var.varValue == 1:
#         print(var.name, " = ", var.varValue)
print("Number of interviews scheduled: ", value(model.objective))

def printSchedule(schedule):
     '''This will print the schedule with the times and the names of each applicant.'''
     timeSlots = np.array(days).flatten()
     for slot in range(len(schedule[0])):
         interviewees = np.where(schedule[:, slot] == 1)[0]
         names = [apps.loc[i, name_col] for i in interviewees]
         print(timeSlots[slot], "-", names) 

printSchedule(schedule)
