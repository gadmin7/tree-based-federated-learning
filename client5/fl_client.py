import socket
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def regression():
    df = pd.read_csv(r"C:\Users\GAURI TOSHNIWAL\Documents\nitt_academics\sem2\FL\FL_regression\client1\dataset_client1.csv")
    print("Dataset Size",df.shape)
    print("Dataset Columns",df.columns)
    print("Summary Statistisc:\n", df.describe())

    X = df[["Temperature"]]
    y = df[["Ice Cream Profits"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("R2 Score: ", r2_score(y_test, y_pred)*100)
    print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))


    return [model.coef_[0][0],model.intercept_[0]],X_test,y_test

def evaluateModel(model, X, y_test):
    
    y_predict = model.predict(X)
    
    MAE = mean_absolute_error(y_test, y_predict)
    MSE = mean_squared_error(y_test, y_predict)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_predict)

    print(f"Mean Absolute Error    : {MAE}")
    print(f"Mean Squared Error     : {MSE}")
    print(f"Root Mean Squared Error: {RMSE}")
    print(f"\nR2 Score: {R2}")

def model_training():
    data = pd.read_csv("walmart_weekly_sales_5.csv")
    data['Date'] = pd.to_datetime(data['Date'], format = "%d-%m-%Y")
    data['Month_Name'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.week
    data.drop('Date',axis=1,inplace=True)

    X = data.drop(['Weekly_Sales'], axis = 1)
    y = data['Weekly_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)

    print("\nEvaluation for Decision Tree Regressor\n")
    evaluateModel(reg, X_train, y_train)
    evaluateModel(reg, X_test, y_test)
    
    return reg,X_test,y_test,X_train,y_train

def main():
    host = '127.0.0.1'
    port = 12345

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    local_params,X_test,y_test,X_train,y_train = model_training()  # Local model parameters (slope, intercept)

    client_socket.send(pickle.dumps(local_params))  # Send local parameters to server

    global_params = pickle.loads(client_socket.recv(2000000))
    print("Updated global parameters:", global_params)

    client_socket.close()

    global_params.fit(X_train,y_train)
    evaluateModel(global_params,X_test,y_test)

    # y_pred = global_params[0]*X_test + global_params[1]
    # print("Accuracy with Golbal Parameters:\n")
    # print("R2 Score: ", r2_score(y_test, y_pred)*100)
    # print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
    # print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

if __name__ == "__main__":
    main()
