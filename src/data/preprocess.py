from sklearn.preprocessing import StandardScaler

def preprocess(X_train, X_valid, X_test):
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train),
        scaler.transform(X_valid),
        scaler.transform(X_test),
    )
