def plot_mat(model,X_test,y_test):   
    
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    classes = model.classes_
    ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes,cmap='Greens')
    ax.set(xlabel='Predict', ylabel='Actual')
    
    print(classification_report(y_test,y_pred))