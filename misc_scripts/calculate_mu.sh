#!/bin/bash

echo Doing word count
mongo "C:\Users\Kevin\Desktop\GitHub\Research\Webscraper\serverscripts\mutualInformation\extractWordCount.js" 

echo Doing Genre count
mongo "C:\Users\Kevin\Desktop\GitHub\Research\Webscraper\serverscripts\mutualInformation\extractGenreCount.js"  

echo Doing Word and Genre joint count
mongo "C:\Users\Kevin\Desktop\GitHub\Research\Webscraper\serverscripts\mutualInformation\extractGenreWordJointCount.js"

wait
echo Calculating MutualInformation and storing in db
mongo "C:\Users\Kevin\Desktop\GitHub\Research\Webscraper\serverscripts\mutualInformation\mutualInformation.js"

