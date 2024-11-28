import pandas as pd
import numpy as np
import seaborn as sns
import time

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.model_selection import  StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

from skopt import BayesSearchCV

def loadData(filepath, delimiter=';'):
    data = pd.read_csv(filepath, delimiter=delimiter)
    #print("Columns:", data.columns)
    #print("Dataset shape:", data.shape)
    return data

def preprocess(data):
    X = data.drop(columns=['y'])
    y = data['y']

    num = X.select_dtypes(include=['int64', 'float64']).columns
    cat = X.select_dtypes(include=['object']).columns

    #normalizing and handling missing values
    catImp = SimpleImputer(strategy='most_frequent')
    numImp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    ohe = OneHotEncoder(handle_unknown='ignore')

    #setting up pipeline
    preprocessor = ColumnTransformer(transformers=[('num', Pipeline(steps=[('imputer', numImp), ('scaler', scaler)]), num), ('cat', Pipeline(steps=[('imputer', catImp), ('encoder', ohe)]), cat)])
    X = preprocessor.fit_transform(X)

    coder = LabelEncoder()
    y = coder.fit_transform(y)
    fNames = list(num) + list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(cat))

    return X, y, fNames

#   Plotting functions

def plotY(y):
    labels = ['No', 'Yes']
    values = [sum(y == 0), sum(y == 1)]     #binary
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
    plt.title('Proportion of Target Variable (y)')
    plt.show()

def plotNum(data):
    num = data.select_dtypes(include=['int64', 'float64']).columns
    for col in num:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 1, 1)
        sns.histplot(data[col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {col}')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col], color='orange')
        plt.title(f'Box Plot of {col}')
        plt.show()

def plotCat(data):
    cat = data.select_dtypes(include=['object']).columns
    for col in cat:
        plt.figure(figsize=(6, 4))
        sns.countplot(y=data[col], order=data[col].value_counts().index, palette='viridis')
        plt.title(f'Count Plot of {col}')
        plt.show()

def heatmap(data):
    num = data.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(10, 7))
    m = data[num].corr()
    sns.heatmap(m, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()

#   Analysis functions

def runKmeans(X, k=2):
    kmeans = KMeans(n_clusters=k, random_state=69) #nice
    clusterL = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, clusterL)
    print(f"K-Means Silhouette Score (k={k}): {silhouette_avg:.2f}")
    return clusterL

def outliers(X, c=0.01):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=c)
    labels = lof.fit_predict(X)
    count = (labels == -1).sum()
    print(f"Number of outliers detected: {count}")
    return labels

def plotPCA(X, labels, title, col='Label'):
    pca = PCA(n_components=2)
    T = pca.fit_transform(X)
    df = pd.DataFrame(T, columns=['PC1', 'PC2'])
    df[col] = labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='PC1', y='PC2', hue=col, palette='viridis' if col == 'Cluster' else 'Set1', s=40)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title=col)
    plt.grid(True)
    plt.show()

def selectFeatures(X, y, fNames, n=10, test_size=0.2, random_state=69):
    X_train, X_test, y_train, yt = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    scores = mutual_info_classif(X_train, y_train, random_state=random_state)
    df = pd.DataFrame({'Feature': fNames, 'Mutual Information': scores})
    df = df.sort_values(by='Mutual Information', ascending=False)

    #print("Features and Their MI Scores:")
    #print(df)
    selected = df.head(n)['Feature'].tolist()
    print(f"\nSelected Top {n} Features:")
    print(selected)
    selectedInd = [fNames.index(f) for f in selected]

    return selectedInd, selected

def evaluate(X, y, kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'):

    model = SVC(kernel=kernel, C=C, gamma=gamma, class_weight=class_weight, probability=True, random_state=69) #using smv

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69) #5-fold cross verification

    # to get averages over k folds
    acc, pres, rec, f1, rocaucs = [], [], [], [], []
    trainTimes, predTimes = [], []
    tAll, pAll, ppy = [], [], [] 

    # Loop over folds
    for i, j in skf.split(X, y):
        X_train, X_test = X[i], X[j] 
        y_train, yt = y[i], y[j]

        start = time.time() #   measuring training efficiency
        model.fit(X_train, y_train)
        trainingT = time.time() - start
        trainTimes.append(trainingT)

        start = time.time() #   prediction efficiency
        pred = model.predict(X_test)
        predP = model.predict_proba(X_test)[:, 1]
        predT = time.time() - start
        predTimes.append(predT)
        
        #fidning means
        tAll.extend(yt)
        pAll.extend(pred)
        ppy.extend(predP)

        acc.append(accuracy_score(yt, pred))
        pres.append(precision_score(yt, pred))
        rec.append(recall_score(yt, pred))
        f1.append(f1_score(yt, pred))
        rocaucs.append(roc_auc_score(yt, predP))

    m = {
        'accuracy': np.mean(acc),
        'precision': np.mean(pres),
        'recall': np.mean(rec),
        'f1_score': np.mean(f1),
        'roc_auc': np.mean(rocaucs),
        'trainingT': np.mean(trainTimes),
        'predT': np.mean(predTimes)
    }
    cnf = confusion_matrix(tAll, pAll)
    print("\nConfusion Matrix:\n", cnf)

    # roc curve
    fpr, tpr, _ = roc_curve(tAll, ppy)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve (Aggregate)')
    plt.xlabel('False Positives')
    plt.ylabel('True Positives')
    plt.legend()
    plt.grid()
    plt.show()
    # metrics
    print(f"Performance:")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall: {m['recall']:.4f}")
    print(f"F1-Score: {m['f1_score']:.4f}")
    print(f"ROC-AUC: {m['roc_auc']:.4f}")
    print(f"Average Training Time per Fold: {m['trainingT']:.4f} seconds")
    print(f"Average Prediction Time per Fold: {m['predT']:.4f} seconds")

def hyperparams(X_train, y_train, search_spaces, n_iter=50):

    model = SVC(probability=True, random_state=42)
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=n_iter, # use 2 for test
        scoring='roc_auc',
        cv=5,  # or 2 if it takes too long
        random_state=69,
        n_jobs=1, # to avoid deadlocks in long processing times
        verbose=3 #info output amt
    )

    print("Starting hyperparameter tuning.")
    bayes_search.fit(X_train, y_train)
    best = bayes_search.best_
    print(f"Best Parameters Found: {best}")

    return best

def main():

    data = loadData('bank.csv', delimiter=';')
    X, y, fNames = preprocess(data)

    plotY(y)
    heatmap(data)
    plotNum(data)
    plotCat(data)

    # outlier detection, display and filter dataset
    labels = outliers(X)
    plotPCA(X, labels, title="Outlier Detection Results", col='Outlier')
    X = X[labels == 1]
    y = np.array(y)[labels == 1]
    print(f"Dataset filtered.")

    # clustering
    clusterL = runKmeans(X, k=2)
    plotPCA(X, clusterL, title="K-Means Clustering Results After Filtering Outliers", col='Cluster')

    # evaluate with all features
    print("Evaluating with all features:")
    evaluate(X, y)

    # select and filter features
    selectedInd, selected = selectFeatures(X, y, fNames, n=10)
    X_filtered = X[:, selectedInd]

    # evaluate with selected features
    print("Evaluating with selected features:")
    evaluate(X_filtered, y)

    # hyperparamater tuning search space
    search_spaces = {
        'C': (1e-2, 10, 'log-uniform'),
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    # evaluate with best params
    best = hyperparams(X_filtered, y, search_spaces, n_iter=30)
    evaluate(X, y, kernel=best['kernel'], C=best['C'], gamma=best['gamma'])

if __name__ == "__main__":
    main()
