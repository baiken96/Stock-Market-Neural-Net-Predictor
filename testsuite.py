import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import model_from_json
import sys

def read_data(f):
    df = pd.read_csv(f)
    df['Date'] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    df.index = df['Date']
    df = df.sort_index(ascending=True, axis=0)

    data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

    for i in range(0, len(data)):
        data['Date'][i] = df['Date'][i]
        data['Close'][i] = df['Close'][i]

    data.drop("Date", axis=1, inplace=True)

    return data

def train_model(data, name, split=0.8):
    dataset = data.values
    train = dataset[:int(len(dataset) * split), :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data, train_labels = [], []
    for i in range(60, len(train)):
        train_data.append(scaled_data[i - 60:i, 0])
        train_labels.append(scaled_data[i, 0])
    train_data, train_labels = np.array(train_data), np.array(train_labels)

    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_data, train_labels, epochs=1, batch_size=1, verbose=1)

    json = model.to_json()
    with open(name + "_model.json", 'w+') as jf:
        jf.write(json)
    model.save_weights(name + "_model.h5")
    print("Model saved successfully")

    return model, scaler

def load_model(name):
    jsondata = open(name + "_model.json", 'r')
    model = model_from_json(jsondata.read())
    model.load_weights(name + "_model.h5")

    return model

def validate_model(model, scaler, data, split=0.8):
    train = data[:int(len(data) * split)]
    valid = data[int(len(data) * split):]
    valid_set = data.values[int(len(data) * split):, :]

    inputs = data[len(data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    valid_data = []
    for i in range(60, inputs.shape[0]):
        valid_data.append(inputs[i - 60:i, 0])
    valid_data = np.array(valid_data)

    valid_data = np.reshape(valid_data, (valid_data.shape[0], valid_data.shape[1], 1))

    closing_price = model.predict(valid_data)
    closing_price = scaler.inverse_transform(closing_price)

    rms = np.sqrt(np.mean(np.power((valid_set - closing_price), 2)))

    valid["Predictions"] = closing_price
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])

    return rms

def predict_future(model, scaler, context, n=60):
    if len(context) != 60:
        raise Exception("Context length must be exactly 60")

    predictions = np.array([])

    for i in range(n):
        input = scaler.transform(context.reshape(-1, 1))
        input = np.reshape(input, (input.shape[1], input.shape[0], 1))
        prediction = scaler.inverse_transform(model.predict(input))
        context = np.append(context, [prediction])
        context = context[len(context) - 60:]
        predictions = np.append(predictions, [prediction])

    plt.plot(context)
    plt.plot([n for n in range(60, 60 + n)], predictions)

    return predictions

def evaluate(test, gold):
    if (len(test) != len(gold)):
        raise Exception("Lengths of datasets do not match")

    return np.sqrt(np.mean(np.power((gold - test), 2)))

def main():

    # $ python testsuite.py full_data.csv gold_data.csv ("load")

    name = sys.argv[1].split('.')[0]
    data = read_data(sys.argv[1])
    try:
        if sys.argv[3] == "load":
            model = load_model(name)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit_transform(data.values)
        else:
            raise Exception("Bad arg 3")
    except IndexError:
        model, scaler = train_model(data, name)
    validation_score = validate_model(model, scaler, data)

    test_scores = []

    for nv in [1, 5, 10]:
        predictions = predict_future(model, scaler, data[len(data) - 60:].values, n=nv)
        dis_gold = read_data(sys.argv[2])[:nv]
        test_scores.append(evaluate(predictions, np.array(dis_gold)))

    with open(name+"_results.txt", 'w+') as f:
        f.write("Validation Score 1: {}\n".format(validation_score))
        f.write("Test Score 1: {}\n".format(test_scores[0]))
        f.write("Test Score 5: {}\n".format(test_scores[1]))
        f.write("Test Score 10: {}\n".format(test_scores[2]))

if __name__ == "__main__":
    main()