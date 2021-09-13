This assignment contains two parts: Linear Regression and Logistic Regression.

# Linear Regression

## Problem Description

#### In this problem, we use Linear Regression to predict the Total Costs incurred by a hospital patient. The training dataset used is the **SPARCS Hospital dataset**, with the target variable as the last column.

### Part 1

Solves the linear regression optimisation problem using the Moore-Penrose pseudo-inverse formula for calculating weights.

### Part 2

Introduces regularization to the model and implements **Ridge Regression**, where the regularization parameter, $\lambda$, is found using **k-folds cross-validation**.

### Part 3

This part is specific to the **SPARCS Hospital dataset**, and implements feature engineering and feature selection using **Lasso Regression** to optimize the loss function. More details can be found in the [report](https://github.com/VaibhavVerma16113108/COL341-Machine-Learning/blob/main/Assignment_1/report.pdf).

## Running Instructions

The code is run as: <br/>
`python3 linear.py Mode Parameters`
<br/>
<br/>
The mode corresponds to part [a,b,c] of the assignment.
The parameters are dependant on the mode: <br/>
a.] `python3 linear.py a trainfile.csv testfile.csv outputfile.txt weightfile.txt` <br/>

### Inputs: </br>

`trainfile.csv`: Training data <br/>`testfile.csv`: Testing data <br/>

### Outputs: <br/>

`outputfile.txt`: Contains line-aligned predictions made on the test dataset <br/>
`weightfile.txt`: Contains the weights found (including the intercept term in the first line)
<br/><br/>
b.] `python3 linear.py b trainfile.csv testfile.csv regularization.txt outputfile.txt weightfile.txt bestparameter.txt` <br/>

### Inputs: </br>

`trainfile.csv`: Training data <br/>`testfile.csv`: Testing data <br/>
`regularization.txt`: Comma-separated list of regularization parameters used for cross-validation

### Outputs: <br/>

`outputfile.txt`: Contains line-aligned predictions made on the test dataset <br/>
`weightfile.txt`: Contains the weights found (including the intercept term in the first line) <br/>
`bestparameter.txt`: Most optimal regularization parameter found after cross-validation. <br/>

c.] `python3 linear.py c trainfile.csv testfile.csv outputfile.txt`

### Inputs: </br>

`trainfile.csv`: Training data <br/>`testfile.csv`: Testing data <br/>

### Outputs: <br/>

`outputfile.txt`: Contains line-aligned predictions made on the test dataset <br/>
