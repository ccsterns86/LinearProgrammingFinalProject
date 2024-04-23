import numpy as np
import cvxopt
from cvxopt import lapack, solvers, matrix, glpk
import pandas as pd

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

def printSchedule(schedule):
     '''This will print the schedule with the times and the names of each applicant.'''
     timeSlots = np.array(days).flatten()
     for slot in range(len(schedule[0])):
         interviewees = np.where(schedule[:, slot] == 1)[0]
         names = [apps.loc[i, name_col] for i in interviewees]
         print(timeSlots[slot], "-", names) 

apps_file = "Applications.csv"
data = pd.read_csv(apps_file)

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
A = apps.iloc[:, initial_num_cols:].to_numpy()

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
M_dict = {}
for i in range(0, M.shape[1]):
    M_dict[i] = list(np.where(M.T[i] == 1)[0])

PPT = 3 #People per timeslot
CATEGORIES_DICT = M_dict
SCHEDULE = A
HOURS = np.array([1,2,3,25,26,27,49,50,51])
CATEGORIES = len(CATEGORIES_DICT)
APPLICANTS = SCHEDULE.shape[0]
TIME_SLOTS = SCHEDULE.shape[1]
assert len(HOURS) == TIME_SLOTS

def arr(start, length, total_length):
    assert start >= 1
    assert length >= 1
    assert total_length >= start + length - 1
    first_half = np.concatenate((np.zeros(start-1), np.ones(length)), axis=0)
    return np.concatenate((first_half, np.zeros(total_length-start-length+1)), axis=0)

def arr_mid(start, length, total_length):
    assert start >= 1
    assert length >= 2
    assert total_length >= start+length -1
    middle = np.hstack([[-1.0], np.ones(length-2), [-1.0]])
    first_half = np.concatenate((np.zeros(start-1), middle), axis=0)
    return np.concatenate((first_half, np.zeros(total_length-start-length+1)), axis=0)

def arr_with_columns(column, rows, columns):
    one_row = arr(column,1,columns)
    return np.reshape(np.vstack([one_row for x in range(rows)]), (rows*columns,))

#LP
#nonzero
G_nonzero = np.vstack([
    -1.0*np.eye(TIME_SLOTS*APPLICANTS+2*CATEGORIES),
    np.eye(TIME_SLOTS*APPLICANTS+2*CATEGORIES)
])
h_nonzero = np.hstack([
    np.zeros(TIME_SLOTS*APPLICANTS), max(HOURS)*np.ones(2*CATEGORIES),
    np.ones(TIME_SLOTS*APPLICANTS), max(HOURS)*np.ones(2*CATEGORIES)
])
c = np.hstack([np.zeros(TIME_SLOTS*APPLICANTS), np.ones(2*CATEGORIES)])
#each applicant is scheduled once
G_timeslots = np.vstack([
np.vstack([
    np.hstack([
        np.zeros(TIME_SLOTS*x),
        -1.0*np.ones(TIME_SLOTS),
        np.zeros(APPLICANTS*TIME_SLOTS-(x+1)*TIME_SLOTS),
        np.zeros(2*CATEGORIES)
    ]) for x in range(0,APPLICANTS)
]),
np.vstack([
    np.hstack([
        np.zeros(TIME_SLOTS*x),
        np.ones(TIME_SLOTS),
        np.zeros(APPLICANTS*TIME_SLOTS-(x+1)*TIME_SLOTS),
        np.zeros(2*CATEGORIES)
    ]) for x in range(0,APPLICANTS)
])
])
h_timeslots = np.hstack([-1.0*np.ones(APPLICANTS), np.ones(APPLICANTS)])
#total of 3 scheduled per hour
G_ppt = np.hstack([
    np.vstack([
        arr_with_columns(x,APPLICANTS,TIME_SLOTS) for x in range(1,TIME_SLOTS+1)
    ]), 
    np.zeros((TIME_SLOTS,2*CATEGORIES))
])
h_ppt = np.ones(TIME_SLOTS)*PPT
A_schedule = np.vstack([
    np.hstack([
        np.zeros(TIME_SLOTS*x), 
        SCHEDULE[x], 
        np.zeros(TIME_SLOTS*(APPLICANTS-x-1)), 
        np.zeros(2*CATEGORIES)
    ]) for x in range(0,APPLICANTS)
])
b_schedule = np.ones(APPLICANTS)
#scheduling categories
G_add = np.vstack([G_timeslots, G_ppt])
h_add = np.hstack([h_timeslots, h_ppt])
for key in CATEGORIES_DICT:
    G_max = np.vstack([
        np.hstack([
            np.zeros(TIME_SLOTS*x), 
            HOURS, 
            np.zeros(APPLICANTS*TIME_SLOTS-(x+1)*TIME_SLOTS), 
            np.zeros(2*key),
            [-1], 
            np.zeros(2*CATEGORIES-1-2*key)
        ]) for x in CATEGORIES_DICT[key]
    ])
    G_min = np.vstack([
        np.hstack([
            np.zeros(TIME_SLOTS*x), 
            -1.0*HOURS, 
            np.zeros(APPLICANTS*TIME_SLOTS-(x+1)*TIME_SLOTS), 
            np.zeros(2*key+1), 
            [-1], 
            np.zeros(2*CATEGORIES-2-2*key)
        ]) for x in CATEGORIES_DICT[key]
    ])
    h_max = np.zeros(len(CATEGORIES_DICT[key]))
    h_min = np.zeros(len(CATEGORIES_DICT[key]))
    G_add = np.vstack([G_add, G_max, G_min])
    h_add = np.hstack([h_add, h_max, h_min])
#
final_G = np.vstack([G_nonzero, G_add])#, G_max_3, G_max2_3])
final_h = np.hstack([h_nonzero, h_add])#, h_max_3, h_max2_3])
final_c = c
final_A = A_schedule
final_b = b_schedule
print(final_G.shape, final_h.shape, final_c.shape, final_A.shape, final_b.shape)
final_G = cvxopt.matrix(final_G)
final_h = cvxopt.matrix(final_h)
final_c = cvxopt.matrix(final_c)
final_A = cvxopt.matrix(final_A)
final_b = cvxopt.matrix(final_b)
(status, x) = cvxopt.glpk.ilp(final_c, final_G, final_h, final_A, final_b, I=set(range(TIME_SLOTS*APPLICANTS+2*CATEGORIES)))

schedule = np.array(x)[:APPLICANTS*TIME_SLOTS].reshape(APPLICANTS,TIME_SLOTS)
objective = sum(np.array(x)[APPLICANTS*TIME_SLOTS:])[0]

print(f"Schedule:\n", schedule)
print(f"Objective: {objective}")