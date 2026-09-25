from xgboost import XGBRegressor

def train_xgboost(X_train, y_train):
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)
    return model