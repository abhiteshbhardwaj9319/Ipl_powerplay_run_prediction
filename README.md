# Ipl_powerplay_run_prediction
For upcoming T20 matches, participants should submit their code to predict the score at the end of 6 overs in each of the innings.
This academic competition has been designed for beginners to learn and compete as well as professionals to demonstrate their capabilities. Even those who are new to data science, but have the interest to learn can go through the resources available online and try participating by using simple mathematical models.
Participation can be as an individual or as a team with a maximum of 4 members.
The individual/team with the lowest error in their prediction over best 35 matches matches will be declared as the winner of the IIT Madras BS Degree’s Cricket Hackathon 2023.

WHAT WILL BE SHARED
Ball-by-ball data of the past T-20 matches
Sample input file which the code has to accept and process

OUTPUT EXPECTED FROM THE CODE
For each innings of each match, the output of the code must be a single number, which is the score at the end of the 6th over.

TELL ME MORE!
Step1: You need to register as an individual or as a team (maximum 4 members).
Step 2: Go through the data sets given and start formulating the algorithm to predict the score.
You can submit and change your code everyday which will be used for evaluation by the IITM BS team to get the score prediction.
Check out the next page for technical details of the languages and packages that can be used.

INSTRUCTIONS FOR THE CODING CONTEST
The actual contest will be run for best 35 matches T20 cricket matches (70 innings) between 15 April and 21 May. Actual dates will be intimated to participants soon.

Each team will submit the code and a writeup of the logic/algorithm used, along with the score predicted by the code (when run on their local system prior to the match)
During the competition, the teams can update and resubmit their code to improve their predictions at regular intervals.
The cut-off time for updated submissions is 12:30 PM (IST) for a 3:30 match and 4:30 PM (IST) for a 7:30 match
The participant should submit the code through their login on the site. More details will be shared with registered participants.
After the match is over everyday, IITM BS team will prepare the input file for each of the innings in the format shared on the website and execute the participant’s latest code against it. The points won by the participant will depend on how close the score predicted by the code matches the actual score.
Every day, the results of the match of the previous day will be released
The leaderboard will display the winners of the day and winners based on cumulative points accrued.

THOSE WHO CANNOT CODE BUT WISH TO PARTICIPATE
If you are a keen follower of cricket and have a fair understanding of the game but are not a programmer or a data scientist, we have something for you too!
Register on the form and choose the option “I am planning to predict the score using intuition”.
Login everyday and predict the score at the end of 6 overs for each team playing that day and win T-shirts from us.
DISCLAIMER:
This is an academic project and all mention of predictions and guessing of the match scores are in the context of the academic interest. Any score predictions will be disclosed the day after the match day.


Problem Statement:

Given certain input parameters regarding the innings of an IPL T20 cricket match, predict the total runs scored by the batting team at the end of 6 overs.
Programming Language to be used: only Python 3.9
Maximum size of zip file allowed: 5MB.
Maximum permitted execution time: 20secs.
We will be executing your code on a VM with the same docker image having a single threaded processor.
Data:

Training data: You will be provided two files for training data, they are as following:
IPL_Ball_by_Ball_2008_2022.csv: This file contains a ball by ball record of each IPL match since 2008 to 2022. The column names are self explanatory. This file is already present in the docker image.
IPL_Matches_Result_2008_2022.csv: This file contains results of each IPL match since 2008 to 2022. The column names are self explanatory. This file is already present in the docker image.
Sample test data: Each row represents a test sample.
test_file.csv: The column names are self explanatory.
Sample submission file: This file contains predicted runs by your model corresponding to each sample in test data.
sample_submission.csv: The column `id` contains the innings for which the prediction is made and the column `predicted_run` contains your prediction for the respective innings
IITM BS IPL Contest 2023 Shared files - Google Drive contains above files for your reference
Note: The test file for an innings may have discrepancies that your model/code will have to take care of, e.g. new players making their debut in IPL 2023, change in team name, change in stadium name etc.
