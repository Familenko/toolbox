class Data():
    
    def __init__(self,problem,target,df,procent=1.0):

        self.problem = problem
        
        self.df = df
        self.target = target

        self.standard_mark = 'off'
        self.minmax_mark = 'off'
        self.pca_mark = 'off'
        
        self.procented_features = df.sample(frac=1).iloc[:int(len(df.index)*procent)]
        
        self.X = self.procented_features.drop({self.target},axis=1)
        self.y = self.procented_features[self.target]

    def split_data(self,valid = 0.2,test=0.1):

        from sklearn.model_selection import train_test_split
        self.X, self.X_check, self.y, self.y_check = train_test_split(self.X, self.y, test_size=test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=valid)

        return self.X_train,self.y_train,self.X_test,self.y_test,self.X_check,self.y_check

    def preprocessing(self,mode='minmax',n_components=2,alpha=0.5):

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if mode=='standard':

            self.standard_mark = 'on'
  
            self.build_scaler = StandardScaler()
            self.X_train = self.build_scaler.fit_transform(self.X_train)
            self.X_test = self.build_scaler.transform(self.X_test)
            self.X_check = self.build_scaler.transform(self.X_check)

        if mode=='minmax':
  
            self.minmax_mark = 'on'

            self.build_scaler = MinMaxScaler()
            self.X_train = self.build_scaler.fit_transform(self.X_train)
            self.X_test = self.build_scaler.transform(self.X_test)
            self.X_check = self.build_scaler.transform(self.X_check)

        if mode=='pca':

            self.pca_mark = 'on'

            self.build_scaler = StandardScaler()
            self.X_train = self.build_scaler.fit_transform(self.X_train)
            self.X_test = self.build_scaler.transform(self.X_test)
            self.X_check = self.build_scaler.transform(self.X_check)

            self.pca = PCA(n_components=n_components)
            principal_components = self.pca.fit_transform(self.X_train)

            self.X_train = pd.DataFrame(principal_components)
            self.X_test = pca.transform(self.X_test)
            self.X_check = pca.transform(self.X_check)

        if mode =='pca' or mode == 'minmax' or mode == 'standard':
            return self.X_train, self.X_test, self.y_train, self.y_test


        if mode=='analyze' and self.problem == 'classification':

            # correlation plot
            corr_df = pd.DataFrame(self.procented_features).corr()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)

            sns.barplot(x=corr_df[self.target].sort_values().iloc[1:-1].index, 
                        y=corr_df[self.target].sort_values().iloc[1:-1].values, ax=ax1)
            ax1.set_title(f"Feature Correlation to {self.target}")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

            # distribution plot
            if len(pd.DataFrame(self.procented_features)[f'{self.target}'].value_counts()) < 10:
                cluster_counts = pd.DataFrame(self.procented_features)[f'{self.target}'].value_counts()
                ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
            else:
                sns.histplot(data=pd.DataFrame(self.procented_features), x=f'{self.target}', kde=True, color='green', bins=20, ax=ax2)

            ax2.set_title(f"{self.target} Distribution")
            ax2.set_xlabel(f"{self.target}")
            ax2.set_ylabel("Count")

            plt.show()

            # PCA plot
            scaler = StandardScaler()
            X_train = scaler.fit_transform(self.X)

            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_train)

            X_train_pca = pd.DataFrame(principal_components)

            print(f'pca.explained_variance_ratio_ = {pca.explained_variance_ratio_}')
            print(f'np.sum(pca.explained_variance_ratio_ = {np.sum(pca.explained_variance_ratio_)}')

            plt.figure(figsize=(12,6))
            sns.scatterplot(x=X_train_pca[0],y=X_train_pca[1],data=pd.DataFrame(self.X),hue=self.y,alpha=alpha)
            plt.xlabel('First principal component')
            plt.ylabel('Second Principal Component')
            plt.show()

        if mode=='analyze' and self.problem == 'regression':

            scaler = StandardScaler()
            X_train = scaler.fit_transform(self.X)

            pca_2 = PCA(n_components=2)
            principal_components_2 = pca_2.fit_transform(X_train)
            d2 = pd.DataFrame(principal_components_2)

            pca_1 = PCA(n_components=1)
            principal_components_1 = pca_1.fit_transform(X_train)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            sns.regplot(data=self.df, x=principal_components_1, y=self.y, ax=ax1)
            ax1.set_xlabel('Principal component')
            ax1.set_ylabel('Target variable')
            ax1.set_title(f'Explained variance ratio: {np.sum(pca_1.explained_variance_ratio_):.2f}')

            sns.scatterplot(x=d2[0], y=d2[1], data=self.df, hue=self.y, alpha=0.5, ax=ax2)
            ax2.set_xlabel('First principal component')
            ax2.set_ylabel('Second principal component')
            ax2.set_title(f'Explained variance ratio: {pca_2.explained_variance_ratio_.sum():.2f}')

    def check_model(self,model,mode='valid'):   
        
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        import pandas as pd
        import scipy as sp

        if mode == 'valid':

            X = self.X_test
            y = self.y_test
            label = 'valid'

        if mode == 'test':

            X = self.X_check
            y = self.y_check
            label = 'test'

        if self.problem == 'classification':

            y_pred = model.predict(X).squeeze()
            cm = confusion_matrix(np.round(y), np.round(y_pred))
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            ax = sns.heatmap(cm, annot=True,cmap='Greens')
            ax.set(xlabel='Predict', ylabel='Actual')

            print(classification_report(np.round(y), np.round(y_pred)))

        if self.problem == 'regression':

            # Отримання PCA-зображень
            pca = PCA(n_components=1)
            X_train = pca.fit_transform(self.X_train)

            X_pca = pca.transform(X)

            # Отримання передбачень моделі
            y_pred = model.predict(X).squeeze()

            # Побудова графіків
            fig, axs = plt.subplots(3, figsize=(10, 15))

            # Перший графік - графік PCA
            axs[0].set_title(f'Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}')
            axs[0].scatter(X_train, self.y_train, c='g')
            axs[0].scatter(X_pca, y, c='g')
            axs[0].scatter(X_pca, y_pred.squeeze(), c='b', label=label)
            axs[0].legend()

            # Другий графік - графік структури
            predictions = model.predict(X).squeeze()
            test_res = y - predictions
            sns.scatterplot(x=y,y=test_res, ax=axs[1])
            axs[1].axhline(y=0, color='r', linestyle='--')

            # Третій графік - графік вірогідності
            _ = sp.stats.probplot(test_res,plot=axs[2])

            plt.show()

    def get_build(self,model):

        from sklearn.pipeline import make_pipeline

        if self.standard_mark == 'on' or self.minmax_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,model)

        if self.pca_mark == 'on':
            self.build_pipe = make_pipeline(self.build_scaler,self.pca,model)

        return self.build_pipe

    def pca_choose(self,min_n=1,max_n=10):

        from sklearn.decomposition import PCA
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
            
        scaler = StandardScaler()
        pca_X = scaler.fit_transform(self.X)

        explained_variance = []

        for n in range(min_n,max_n):
            pca = PCA(n_components=n)
            pca.fit(pca_X)
            
            explained_variance.append(np.sum(pca.explained_variance_ratio_))

        plt.plot(range(min_n,max_n),explained_variance)
        plt.xlabel("Number of Components")
        plt.ylabel("Variance Explained")
        plt.grid(alpha=0.2);

    def target_corr(self):

        import seaborn as sns
        import matplotlib.pyplot as plt

        corr_df = self.df.corr()

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6), dpi=200)

        sns.barplot(x=corr_df[self.target].sort_values().iloc[1:-1].index, y=corr_df[self.target].sort_values().iloc[1:-1].values, ax=ax1)
        ax1.set_title("Feature Correlation to Cluster")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

        if len(self.df[self.target].value_counts()) < 10:
            cluster_counts = self.df[self.target].value_counts()
            ax2.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
        else:
            sns.displot(data=self.df, x=self.target, kde=True, color='green', bins=20, ax=ax2)

        plt.show()