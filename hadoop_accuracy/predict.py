import numpy as np

def predict(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    return model.predict(input_data)[0]