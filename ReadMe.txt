
Bank Loan Prediction with the available data in csv file bank-loan.csv

Projects done in R and Python are self explanatory with heading above each code.

both R and Python can be executed with CMD 

Rscript done on R 4.0 in RStudio and python coding on Python 3.7 in Jupyter

.........................Python................................

Python is also saved as 'Loan_Prediction_pickle' in pickle file 

and can be called by....

with open('Loan_prediction_pickle','rb') as f:
	model=pickle.load(f)


.........................RScript................................


Rscript is saved as "loan-predicion.RDS" and can also be called by.....

model <- readRDS("Loan-prediction.RDS")