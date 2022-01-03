The model_assessment.sas code was provided by Justyna Czaja, an Analytical Consultant on the SAS Analytics Practice Team.  
It can be inserted in a SAS Code node in Model Studio after a supervised learning modeling node (for binary classification) to generate additional assessment plots, in particular plots of various metrics (Sensitivity, Specificity, Accuracy, FPR) at different probability cutoff values. No changes are necessary - it is written in a generic fashion to work as-is. You can modify and extend this code to generate different types of plots you may need.

The precision_recall.sas code was provided by Tamara Fischer for plotting precision-recall curves in a SAS Code node following a supervised learning modeling node (for binary classification).

For both of these examples, the code is entered into the Training Code pane of the SAS Code node.



