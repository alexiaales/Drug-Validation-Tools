# GUI for gene target validation 
A tool for classification and clustering has been developed in the lab. The initial scripts can be found in the following link(https://github.com/alexiaales/drug-target-validation-/blob/master/READ_ME.docx). Since however these tools maybe used in the future from people who might not be familiar with programming a graphical interface was done in order to facilitate these users. 

In this github you can find the following :
-- a GUI script for linux and for windows.
-- a toy-file with data you can use in order to classify or cluster genes.(.csv files are compatible for linux while .xlsx for windows)
-- a toy-training file in case you want to performe classification(once again .csv files are compatible for linux while .xlsx for windows)


Once the user runs the script a window will appear letting you decide whether you want to proceed with a classification or clustering analysis. If you have selected to do a classification analysis you will be asked to insert a training data set and later on, the data you wish to classify your genes into. Once you provide these files, a window with the accuracy of each classifier will pop up on the right corner on your screen letting you know how each one out of these 10 proposed classifiers performed on your training set. You can then decide with which classifier you would like to continue your analysis with. In the end of the classification process a window will appear letting you know that the classification is completed. You can then visit your working directory to see the output files.

If on the other hand you have selected to do a clustering analysis, then you will be asked to provide the data you want to perform the clusering on and of course the number of clusters you are interested in clustering you data into. Once these parameters have been set, then once again you can visit your working directory to check out your results.

>For more information concerning the implementation of each tool respectively you can visit this page :
>https://github.com/alexiaales/drug-target-validation-/blob/master/READ_ME.docx
>For additional questions please do not hesiatate to contact us.
