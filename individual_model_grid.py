class Poly():

    def __init__(self,X,y,high_range=3):

        self.X = X
        self.y = y
        self.high_range = high_range+1

        self.models = []

        self.train_rmse_errors = []
        self.test_rmse_errors = []
        self.cross_poly = []

    def df(self,cv=10,scoring='neg_mean_squared_error'):

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error,mean_squared_error

        for d in range(1,self.high_range):
            
            polynomial_converter = PolynomialFeatures(degree=d,include_bias=False)
            poly_features = polynomial_converter.fit_transform(self.X)
            X_train, X_test, y_train, y_test = train_test_split(poly_features, self.y, test_size=0.3)

            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            model = LinearRegression(fit_intercept=True)
            model.fit(X_train,y_train)
            self.models.append(model.best_estimator_)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))
            test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))
            self.train_rmse_errors.append(train_RMSE)
            self.test_rmse_errors.append(test_RMSE)

            scores_poly = cross_val_score(model, poly_features, self.y, cv=cv, scoring=scoring)
            self.cross_poly.append(np.sqrt(-scores_poly).mean())

        df = pd.DataFrame({'train_rmse_errors':self.train_rmse_errors,
        'test_rmse_errors':self.test_rmse_errors,
        'cross':self.cross_poly},
        index = range(1,len(self.train_rmse_errors)+1))

        return df

    def graph(self):

        import matplotlib.pyplot as plt
        import pandas as pd

        plt.plot(range(1,self.high_range),self.train_rmse_errors[:high_range-1],label='TRAIN')
        plt.plot(range(1,self.high_range),self.test_rmse_errors[:high_range-1],label='TEST')
        plt.xlabel("Polynomial Complexity")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show()


class KNN():

    def __init__(self,X,y,k=30):

        self.X = X
        self.y = y
        self.k = k+1

        self.test_error_rates = []
        self.models = []

        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
        scaler = StandardScaler()
        self.scaled_X_train = scaler.fit_transform(self.X_train)
        self.scaled_X_test = scaler.transform(self.X_test)

    def graph(self):

        import matplotlib.pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score

        for i in range(1,self.k):
            knn_model = KNeighborsClassifier(n_neighbors=i)
            knn_model.fit(self.scaled_X_train,self.y_train) 
            y_pred_test = knn_model.predict(self.scaled_X_test)
            test_error = 1 - accuracy_score(self.y_test,y_pred_test)
            self.test_error_rates.append(test_error)

        plt.plot(range(1,31),self.test_error_rates,label='Test Error')
        plt.legend()
        plt.ylabel('Error Rate')
        plt.xlabel("K Value")
        plt.show()

    def df(self,cv=5):

        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KNeighborsClassifier

        model = GridSearchCV(KNeighborsClassifier(),param_grid={'n_neighbors':list(range(1,self.k)),
            'weights':['uniform', 'distance'],'algorithm':['ball_tree', 'kd_tree', 'brute'],
            'p':[1,2,3,4,5,6,7,8,9,10]},cv=cv,scoring='accuracy')
        model.fit(self.X_train,self.y_train)
        self.models.append(model.best_estimator_)

        return pd.DataFrame(model.best_estimator_.get_params(),index=[1])


def forest(X,y,min=64,max=128):
    
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    errors = []

    for n in range(min,max):
        rfc = RandomForestClassifier(n_estimators=n)
        rfc.fit(X,y)
        preds = rfc.predict(X)
        err = 1 - accuracy_score(preds,y)
        errors.append(err)

    plt.plot(range(min,max),errors)