from linreg import PolynomialLinearRegression
from linreg import LinearRegression

# Example of polynomial linear regression
print('-'*100)
plr = PolynomialLinearRegression([1, 2, 3, 4, 5],  # X values
                                 [14, 64, 182, 398, 742],  # Y values
                                 3)  # polynomial degree of the resultant equation

print(plr)
print(f'  --> The Y of the equation is: {plr.Y_proc}')
print(f'  --> The R2 of the equation is: {plr.R2}')
print('-'*100)

# Example of Linear regression with multiple variables
lr = LinearRegression([6, 25, 74, 165, 310],  # Y values
                      [[1, 2, 3, 4, 5],  # X1 values
                       [1, 4, 9, 16, 25],  # X2 values
                       [1, 8, 27, 64, 125]])  # X3 values (you can add more X lists)

print(lr)
print(f'  --> The Y of the equation is: {lr.Y_proc}')
print(f'  --> The R2 of the equation is: {lr.R2}')
print('-'*100)
