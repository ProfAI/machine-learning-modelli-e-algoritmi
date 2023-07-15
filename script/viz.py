import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_decision_boundary(model, train_set, test_set, sv=None):
        
    #plt.figure(figsize=figsize)
        
    if(model):
        X_train, Y_train = train_set
        X_test, Y_test = test_set
        X = np.vstack([X_train, X_test])
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                             np.arange(y_min, y_max, .02))

        if hasattr(model, "predict_proba"):
            Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        else:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=.8)

    plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
    plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, alpha=0.6)
    
    if sv is not None:
      plt.scatter(sv[:, 0], sv[:, 1], facecolors="none", edgecolor='white', s=100)

    plt.show()
