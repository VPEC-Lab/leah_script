'''

Created on Tue Dec 28  

@author: gorkemer

Demo Experiment for Python W22 course.

This program draws a Gabor grating that could vary in orientation and location at each trial. There will be 200 trials (Tim: right now trial n (nTrial) is set to 200, 
look for nTrial and change it if you want to test it with lower n, also stimDur is set to 0.15 seconds, in your matlab study, it is set to 0.2 seconds, feel free to test other values for this one too)

Program then fit a logistic function and plots the results.

'''

#===============================================
# Import modules
#===============================================

from psychopy import core, visual, event, monitors, info, gui
import numpy as np
import random as rd
import csv
import pprint
from psychopy.tools.monitorunittools import posToPix
import pandas as pd
import pylab
import scipy as sy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

#==============================================
# Key Functions - Experiment
#==============================================

def showIntroText():
    introText.size = 0.8
    introText.draw()
    win.flip()
    event.waitKeys(keyList= 'space')
    dlg = gui.DlgFromDict(dictionary=exp_info, title=exp_name)
    print(exp_info)

def showDebriefText():
    debrief.draw()
    win.flip()
    event.waitKeys(keyList = ['space'])

def runExperimentLoop():
    #===========================================
    # Initialize the experiment session loop
    #===========================================
    for trials in range(nTrial):
        rt_clock.reset() 
        gabor.ori = featureList_ori[trials]
        gabor.pos = coordList[trials]
        fixation.draw()
        win.flip()
        core.wait(1) #1 second wait-time after a response
        #posPix = posToPix(gabor)
        for frameN in range(stimDur):
            fixation.draw()
            gabor.draw()
            win.flip()
        fixation.draw()
        win.flip() #need to flip back again to show a gray screen
        
        rt_clock.reset() # reset the time after the stimulus was shown
        while rt_clock.getTime() <= 10:     # For the duration of 'responseWindow', listen for a left or right press
            #keys = event.getKeys(keyList=['left','right'])
            keys = event.getKeys(keyList=['left', 'right'], timeStamped=rt_clock )
            for key in keys:
                if key[0]=="left":
                    key_ID = "left" #could be 1 
                else:
                    key_ID = "right" # 2
                responses.append(key_ID) #key[1] is the timestamp
                responseTime.append(key[1])
                print(key_ID)
            if keys:
                break
            if event.getKeys(keyList=['escape', 'q']): # listen for a escape or q press, it will run after the above left-right press is progressed.
                win.close()
                core.quit()

#==============================================
# Key Functions - Analyses
#==============================================

def saveData():
    #===========================================
    # Save Data
    #===========================================
    header = ["trialIndex","key", "time", "tilt", "coordinate", "coordName","cardinalType"]
    rows = zip(trialIndex, responses, responseTime, featureList_ori, coordList, coordNames, cardType)
    with open(dir_path+fileName, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        print(rows)
        for row in rows:
            writer.writerow(row)
        for key, value in exp_info.items():
            writer.writerow([key, value])



def sigmoid(x, x0, k):
     y = 1 / (1 + np.exp(-k*(x-x0)))
     return y

def plotEm(xdata, ydata, condLabel, RT):
     xdata = xdata.get('xx') #here we used get function to get the values of a dictionary - # getting the value of a dictionary given a key: xdata.keys() 
     ydata = ydata.get('yy')
     #labelCond =  xdata.get('cond', "defaultReturned" ) # getting the conditon name to to-be-plotted-data 
     par, pcov = curve_fit(sigmoid, xdata, ydata) 
     print(""+ condLabel + "parameters: (threshold, slope):", par)
     x = np.linspace(-20, 20, 50) #define the range of the x-axis ticks
     y = sigmoid(x, *par)
     plotCurves(xdata, ydata, x, y, condLabel, RT)

def plotCurves(xdata, ydata, x, y, condLabel, RT):
     # plot the results
     pylab.plot(xdata, ydata, 'o', label=condLabel + " data")
     pylab.plot(x,y, label= condLabel + " fit")
     pylab.ylim(0, 1.05)
     pylab.legend(loc='best')

     
def plotRegLine(d, RT):
     #plot tilt x RT regression
     #plt.clf()
     tilt = abs(d['tilt']) #got the abs value
     plt.plot(tilt, RT, 'o', color="black") #create scatter plot
     m, b = np.polyfit(tilt, RT, 1) #m = slope, b=intercept
     plt.plot(tilt, m*tilt + b, color="blue")
     plt.ylabel('RT')
     plt.xlabel('Tilt (abs value)')
     plt.show()

def plotRTHist(RT):
     #plot RT histogram

     plt.hist(RT, density=True, facecolor='g', alpha=0.75, histtype='bar', ec='black')  # density=False would make counts
     plt.ylabel('Number of Trials')
     plt.xlabel('RT')
     #plt.figure(dpi=300)
     plt.show()

def analyseData():
     # get the data 
     d = pd.read_csv(dir_path+fileName)
     #find total observations in dataset
     len(d.index)
     d['key2'] = np.where(d['key']== 'right', 1, 0) #convert 'left' & 'right' to 0 and 1
     #X = d['tilt'].values.reshape(-1,1) #these are scipy logistic regression, if you like
     #y = d['key2'].values.reshape(-1,1)

     #subset the data based on the cardinal condition
     d_c1 = d.loc[d['cardinalType'] == 1]
     d_c0 = d.loc[d['cardinalType'] == 0]

     #define x and y values for the plot (and for curve fitting)
     xdata_c1 = {'xx': d_c1['tilt'].values, 'cond': "cardinal"}
     ydata_c1 = {'yy': d_c1['key2'].values, 'cond': "cardinal"}
     #xdata_c1 = d_c1['tilt'].values
     #ydata_c1 = d_c1['key2'].values

     xdata_c2 = {'xx': d_c0['tilt'].values, 'cond': "non-cardinal"}
     ydata_c2 = {'yy': d_c0['key2'].values, 'cond': "non-cardinal"}
     #xdata_c2 = d_c0['tilt'].values
     #ydata_c2 = d_c0['key2'].values   

     RT_c1 = d_c1['time'].values
     RT_c2 = d_c0['time'].values
     RT = d['time'] #overall reaction time
     
     condLabel = "cardinal"
     plotEm(xdata_c1, ydata_c1, condLabel, RT_c1)
     condLabel = "non-cardinal"
     
     plotEm(xdata_c2, ydata_c2, condLabel, RT_c2) #this two-function operation should be written more efficiently. 
     pylab.show()
     #pylab.close()
     #core.wait(3)
     plotRTHist(RT)
     plotRegLine(d, RT)
     

#==============================================
# Settings that we might want to tweak later on
#==============================================

scrsize = (600,400)                # screen size in pixels
stimDur = int(60 * 0.15) #.20 seconds
respWindow = 60 * 10 # 50 seconds to respond
trialClock = core.Clock()
nDur = 60 * 60 #frame rate multiplied by 3 will equal to 3 seconds
nTrial = 15 #200
pressed = False
responses = []
responseTime = []
cardType = [] #cardinal type
coordNames = []
dir_path = os.path.dirname(os.path.realpath(__file__)) #get the full path to the directorrry a python File is contained in
print(dir_path)
fileName = "testDemoRun.csv"
#data_path = subj_id + "_cond_" + cond_num + "_rep_" + rep_num + ".tsv"

#==============================================
# dialog box
#==============================================
exp_name = 'Orientation Discrimination'
#below object is called dictionary. It can store 1) keys and 2) values.

exp_info = {
            'Observer': '', 
            'Whats your participant number?': '',
            'How old are you in people years?': '',
            'How would you describe your gender?': '',#('male', 'female'), 
            'What about your ethnicity?':'', 
            'left-handed':False 
            }

#this is the bit that brings up a dialog box and populates the box with the 
#values obtained from the dictionary. 
#gui is a module that we import this.

#=========================================
# Creation of window
#=========================================
mon = monitors.Monitor('SonyG55')#fetch the most recent calib for this monitor
mon.setDistance(73)#further away than normal?

for mon in monitors.getAllMonitors():
    print(mon, monitors.Monitor(mon).getSizePix())

win = visual.Window(size=(1440,800), monitor = mon, fullscr=False, units="norm")

rt_clock = core.Clock()
rt_clock.reset()  # set rt clock to 0

#===========================================
# Creation of stimuli
#===========================================

gabor = visual.GratingStim(win, tex="sin", mask="gauss", texRes=256, 
        size=[0.25, 0.25], sf=[5, 0], ori = 0, name='gabor1', contrast = 0.05)  
introText = visual.TextStim(win, text = "Lets get started. I'm going to ask you some questions now. I will know if you are lying, so you better tell the truth. Press 'space to continue.\
                            \
                            You will see a black-white grated (bars?) circle and you are asked to judge the orientation of the black-white bars. If you think the bars rotated towards the right, press right-arrow-key, and for left, please press left-arrow-key. Press 'escape' to quit the experiment any time.",
                            )
debrief = visual.TextStim(win, pos = (0.0, 0.0), text = "The purpose of the experiment is to test if discriminating orientation is sensitive to the location of the objects. \
                          ... some other debrief stuff.. Press 'space' to see your plotted data, and close the plot to close this experiment window.")
message2 = visual.TextStim(win, pos = (0.0, 0.0), text = "Something is wrong. Check the console") # to change size message2.size = [0.2, 0.2]

errorMessage = visual.TextStim(win, pos = (0.0, 0.0), text = "Some debug text") # when needed
message2.size = [0.2, 0.2]
fixation = visual.Rect(win, width=0.01, height=0.01, fillColor='black', fillColorSpace=None, pos=(0, 0)) 
 
#=================================================
# Store feature lists to be tested on each trial
# 1) orientation of the Gabor, and,
# 2) location of the Gabor 
#=================================================

distToCenter = 0.5
#  footnote 1. # Other not-really-efficienct option would be define all posibilities and create list of that, e.g., coordList = [(0,distToCenter), (distToCenter,distToCenter), (distToCenter,0), (distToCenter,-distToCenter), (0,-distToCenter), (-distToCenter,-distToCenter), (-distToCenter,0), (-distToCenter,distToCenter)], then do: coordList = coordList * int(nTrial/len(coordList))
oriList = list(range(-20,20, 1)) # Orientation convention is like a clock: 0 is vertical, creates a list (can be said array because the elements had the same data type) starting from 0 to 360, with intervals of 5 (e.g. 0, 5, 10, 15..), # 0: 3pm, 180: 9pm, 90: noon, 270: 6pm
featureLists_coor = [(0,distToCenter), (distToCenter,distToCenter), (distToCenter,0), (distToCenter,-distToCenter), (0,-distToCenter), (-distToCenter,-distToCenter), (-distToCenter,0), (-distToCenter,distToCenter)]
featureList_ori = [] #for the oriList, we might need to use a for loop to determine feature lists
coordList = []
# alterniatvely, we can expand the n of this list to a trial number and shuffle it, just like what we did for the coordList rd.shuffle(oriList), see footnote 1. 
for t in range(nTrial):
    ''' determining feature lists'''
    ori = rd.choice(oriList)
    coor = rd.choice(featureLists_coor)
    featureList_ori.append(ori) #append it to our feature list
    coordList.append(coor)


len(coordList) #check the number of elements in the list
rd.shuffle(coordList)

trialIndex = list(range(1,nTrial, 1)) #to name our rows in the final data sheet. Note that we start from 1 to make it more intutive. 


for coord in coordList: #  #there is no replace method for the lists in python, (for student) so here we do things manually, a challenge for you would be to reduce this operation to few lines) (for student)
    if coord == (0,distToCenter):
        coordNames.append("noon")
    if coord == (distToCenter,distToCenter):
        coordNames.append("1:30")
    if coord == (distToCenter,0):
        coordNames.append("3")
    if coord == (distToCenter,-distToCenter):
        coordNames.append("4:30")
    if coord == (0,-distToCenter):
        coordNames.append("6")
    if coord == (-distToCenter,-distToCenter):
        coordNames.append("7:30")
    if coord == (-distToCenter,0):
        coordNames.append("9")
    if coord == (-distToCenter,distToCenter):
        coordNames.append("10:30")
    if coord in [(0,distToCenter), (distToCenter,0), (0,-distToCenter), (-distToCenter,distToCenter)]:
        cardType.append(1) #cardinal
    else:
        cardType.append(0) #non-cardinal

np.unique(cardType, return_counts=True) # get the freq stats of a numpy array to confirm



showIntroText()

runExperimentLoop()
showDebriefText()

saveData()
analyseData()


# Quit the experiment
win.close()
core.quit()

'''
things to note here:

- notice that functions are located at the top of the program. I notice many programmers like to seperate functions to the top. 


'''

