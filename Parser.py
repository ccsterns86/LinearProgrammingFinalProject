# Parser.py is meant to parse the applicant data into easily usable components
# Input csv file: 
#   - Name
#   - Major needs to be standardized
#   - Interview times cannot have commas in the name

# To run this program, you have to have python installed on your computer. 
# You may have to install some other packages (pandas, random)
# You run the command:
#       python3 .\Parser.py "<csv file>"
# Example:
#       python3 .\Parser.py ".\Applications.csv" 

import sys
import pandas as pd
import random
import numpy as np
import pulp

num_interviews = 27
# This is the name of the column that contains availability in the csv file.
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

print(apps.head(n=10))

#When turning this into arrays, we can make multiple different arrays
A = apps.iloc[:, initial_num_cols:].to_numpy()
print(A)

interview_num_cols = apps.shape[1]

# Go through the applicants and extract majors
# TODO: fix the funky .loc warning
for index, applicant in apps.iterrows():
    majors = applicant[major_col].split(", ")
    for major in majors:
        apps.loc[index, major] = 1

M = np.nan_to_num(apps.iloc[:, interview_num_cols:].to_numpy())
print(M)