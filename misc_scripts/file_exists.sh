#!/bin/bash 
file=$1 
 
if [ -e "$file" ] 
then 
	echo "EXISTS" 
else 
	echo "NOTEXISTS" 
fi 
 
