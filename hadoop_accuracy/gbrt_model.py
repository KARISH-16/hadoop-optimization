from sklearn.ensemble import GradientBoostingRegressor

def train_gbrt(X_train, y_train):
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model