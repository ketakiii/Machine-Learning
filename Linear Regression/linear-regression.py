import numpy as np 

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 2.9, 3.7, 4.5, 5.1])

def linear_regression(X, y):
    """
    Perform linear regression using the OLS method.
    Args:
        X (np.darray): input feature
        y (np.darray): target variable
    Returns:
        beta0, beta1: coefficients of the model.
    """
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    #Calculate beta1 (slope)
    numerator = np.sum((X - X_mean) * (y - y_mean))
    demoninator = np.sum((X - X_mean) ** 2)
    beta1 = numerator / demoninator

    # Calculate beta0 (intercept)
    beta0 = y_mean - beta1 * X_mean

    return beta0, beta1

def predict(X, beta0, beta1):
    """
    Predicts the target variable using linear regression using coefficients.
    Args:
        X (np.darray): input feature
        beta0 (float): intercept
        beta1 (float): slope
    Returns:
        y_pred (np.darray): predicted target variable
        """
    y_pred = beta0 + beta1 * X
    return y_pred

beta0, beta1 = linear_regression(X, y)
y_pred = predict(X, beta0, beta1)

print(f'Coefficients, intercept: {beta0}, slope: {beta1}')
print(f'Prediction for x = 6 is {predict(6, beta0, beta1)}')