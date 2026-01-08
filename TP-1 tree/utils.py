from matplotlib.colors import ListedColormap
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def plot_2d_scatter(X, y, feature_names, title=""):
    plt.figure(figsize=(5,4))
    for label, marker, name in [(0, "o", "setosa"), (1, "s", "versicolor"), (2, "^", "virginica")]:
        idx = (y == label)
        plt.scatter(X[idx, 0], X[idx, 1], label=name, marker=marker, alpha=0.8)
    plt.xlabel(feature_names[0]); plt.ylabel(feature_names[1])
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

def plot_linear_boundary_2d(model, X, y, feature_names, title="Decision boundary"):
    # Works for linear SVMs (SVC with linear kernel or LinearSVC) in 2D
    plt.figure(figsize=(5,4))
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid).reshape(xx.shape)
    # margin lines at -1, 0, 1
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--','-','--'])
    # data
    for label, marker, name in [(0, "o", "setosa"), (1, "s", "versicolor"), (2, "^", "virginica")]:
        idx = (y == label)
        plt.scatter(X[idx, 0], X[idx, 1], label=name, marker=marker, alpha=0.8)
    plt.xlabel(feature_names[0]); plt.ylabel(feature_names[1])
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()





def plot_predictions(X_moon,y_moon,clf, axes, title=""):
    x0s = np.linspace(axes[0], axes[1], 300)
    x1s = np.linspace(axes[2], axes[3], 300)
    x0, x1 = np.meshgrid(x0s, x1s)
    grid = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(grid).reshape(x0.shape)
    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(grid).reshape(x0.shape)
        plt.contour(x0, x1, y_score, levels=[-1, 0, 1], linestyles=["--","-","--"], linewidths=1)
    plt.contourf(x0, x1, y_pred, alpha=0.15)
    plt.scatter(X_moon[y_moon==0,0], X_moon[y_moon==0,1], marker="o", alpha=0.9, label="class 0")
    plt.scatter(X_moon[y_moon==1,0], X_moon[y_moon==1,1], marker="^", alpha=0.9, label="class 1")
    plt.legend(); plt.title(title); plt.xlim(axes[0], axes[1]); plt.ylim(axes[2], axes[3])
    plt.tight_layout(); plt.show()




from matplotlib.colors import ListedColormap


def plot_decision_boundary_binary(clf, X, y, feature_names=None,
                                  axes=None, plot_training=True, legend=True, title=None):
    X = pd.DataFrame(X, columns=feature_names or ["x1","x2"])
    x1min, x1max = X.iloc[:,0].min(), X.iloc[:,0].max()
    x2min, x2max = X.iloc[:,1].min(), X.iloc[:,1].max()
    if axes is None:
        pad1 = 0.05 * (x1max - x1min) if x1max > x1min else 1.0
        pad2 = 0.05 * (x2max - x2min) if x2max > x2min else 1.0
        axes = [x1min - pad1, x1max + pad1, x2min - pad2, x2max + pad2]

    x1s = np.linspace(axes[0], axes[1], 400)
    x2s = np.linspace(axes[2], axes[3], 400)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(pd.DataFrame(X_new,columns=feature_names
)).reshape(x1.shape)

    # Background regions (0=no, 1=yes)
    cmap_bg = ListedColormap(['#fafab0', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap_bg)
    plt.contour(x1, x2, y_pred, colors='k', linewidths=1, alpha=0.6)
    if plot_training:
        plt.plot(X.iloc[:,0][y==0], X.iloc[:,1][y==0], "o", label="no", markersize=4)
        plt.plot(X.iloc[:,0][y==1], X.iloc[:,1][y==1], "^", label="yes", markersize=4)

    plt.axis(axes)
    plt.xlabel(feature_names[0] if feature_names else r"$x_1$", fontsize=12)
    plt.ylabel(feature_names[1] if feature_names else r"$x_2$", fontsize=12)
    if legend: plt.legend(loc="lower right")
    if title: plt.title(title)


