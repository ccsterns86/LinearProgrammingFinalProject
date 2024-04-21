# ScheduleMax.py is meant to parse the applicant data into easily usable components
# Input csv file: 
#   - Name
#   - Major needs to be standardized
#   - Interview times cannot have commas in the name

# To run this program, you have to have python installed on your computer. 
# You may have to install some other packages (pandas, random)
# You run the command:
#       python3 .\ScheduleMin.py "<csv file>"
# Example:
#       python3 .\ScheduleMin.py ".\Applications.csv" 

import sys
import pandas as pd
import random
import numpy as np
from pulp import *

num_interviews = 8
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

def printSchedule(schedule):
     '''This will print the schedule with the times and the names of each applicant.'''
     timeSlots = np.array(days).flatten()
     for slot in range(len(schedule[0])):
         interviewees = np.where(schedule[:, slot] == 1)[0]
         names = [apps.loc[i, name_col] for i in interviewees]
         print(timeSlots[slot], "-", names) 

# Input (should be cvs file)
apps_file = sys.argv[1]
data = pd.read_csv(apps_file)

# Grab 27 applicants from the file
apps = data.head(n=num_interviews)

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

# A is the availability of the applicants
A = apps.iloc[:, initial_num_cols:].to_numpy()

# Go through the applicants and extract majors
interview_num_cols = apps.shape[1]
for index, applicant in apps.iterrows():
    majors = applicant[major_col].split(", ")
    for major in majors:
        if major in apps.columns:
                apps.loc[index, major] = 1
        else:
                col = apps.shape[1]
                apps.insert(col, major, 0)
                apps.loc[index, major] = 1
M = np.nan_to_num(apps.iloc[:, interview_num_cols:].to_numpy())
# print(M)

################################################### ILP

model = LpProblem("Scheduling_Problem", LpMinimize)

# Variables
x = LpVariable.dicts("Interview", [(i,j) for i in range(len(A)) for j in range(len(A[0]))], cat='Binary')
majorDay = LpVariable.dicts("MajorOnDay", [(m,d) for m in range(len(M[0])) for d in range(len(days))], cat='Integer')
majorDayIndicator = LpVariable.dicts("MaxCountIndicator", [(m,d) for m in range(len(M[0])) for d in range(len(days))], cat='Binary')
bigM = 99999

# Objective Function
model += lpSum(majorDayIndicator[m, d] for m in range(len(M[0])) for d in range(len(days)))

# Constraints
# Make sure that everyone is scheduled
model += lpSum(x[i,j]*A[i,j] for j in range(len(A[0])) for i in range(len(A))) == num_interviews

for i in range(len(A)): # Interveiwee can only be scheduled once
        model += lpSum([x[i,j] for j in range(len(A[0]))]) <= 1
for j in range(len(A[0])): # 3 interviewees can be scheduled per hour
        model += lpSum([x[i,j] for i in range(len(A))]) <= 3

for major in range(len(M[0])): # For summing the number of applicants for a major scheduled on a day.
      day_index = 0
      for day in range(len(days)):
            dayRange = range(day_index, day_index+len(days[day]))
            model += lpSum(x[i,j]*M[i,major] for j in dayRange for i in range(len(A))) == majorDay[major, day]
            day_index += len(days[day])

for major in range(len(M[0])): # Indicator var if applicants of specific major scheduled on that day.
      for day in range(len(days)):
            model += majorDay[major, day] <= M*majorDayIndicator[major, day]

# Solve the problem
model.solve()

#######################################################

# Print the results
print("Status:", LpStatus[model.status])
schedule = np.zeros((len(A), len(A[0])))
for i in range(len(A)): # Get the schedule
     for j in range(len(A[0])):
          if (x[i,j].varValue == 1): schedule[i, j] = 1
print("Schedule: \n", schedule) 
print("Availability: \n", A)  

# Get the variables of if a major is scheduled on a day
majorScheduled = np.zeros((len(M[0]), len(days)))
majorIndicator = np.zeros((len(M[0]), len(days)))
for m in range(len(M[0])):
        for i in range(len(days)):
            if (majorDay[m,i].varValue > 0): majorScheduled[m,i] = majorDay[m,i].varValue
            if (majorDayIndicator[m,i].varValue > 0): majorIndicator[m,i] = majorDayIndicator[m,i].varValue
print("Major: \n", M)
print("Major Scheduled: \n", majorScheduled)
print("Major indicator: \n", majorIndicator)

print("Optimized value:", value(model.objective))

printSchedule(schedule)
