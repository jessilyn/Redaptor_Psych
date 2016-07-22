# Redaptor_Psych
Shared repo for predicting JPsych badges


Learn about badges and see if we can automate this process (since its currently based on reviewer manual curation). 

Here (https://osf.io/8kt4b/wiki/home/) is the page describing How they scored the methods sections for badges in the psychology journals, with the detailed flowchart here (https://osf.io/hwtb6/) and list of rules here (https://osf.io/4rf3v/). 

We can refer to these rules for our scoring system and automated badge calling.


use featurize.py - it will assign a label based on yuor input documents and labels

inputs: 
1) inputdir <- point to your directory of clean documents

2)loadlabels in <- replace with your dictionary where it assigns a true label based on the docID


For all 40 possible labels, replace the dictionary and re-run 
