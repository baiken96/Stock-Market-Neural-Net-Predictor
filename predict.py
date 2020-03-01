from testsuite import *

def main():

    # $ python predict.py model_prefix (#_days)

    name = sys.argv[1]
    model = load_model(name)
    data = read_data(name+".csv")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(data.values)
    t = None
    try:
        t = sys.argv[2]
    except:
        pass

    if t is not None:
        predictions = predict_future(model, scaler, data[len(data) - 60:].values, n=t)
        with open(name + "_" + t + "_predictions.txt", 'w+') as f:
            f.write("Prediction {}: {}\n".format(t, predictions))
        print("Prediction {}: {}\n".format(t, predictions))
    else:
        predictions = []
        for nv in [1, 5, 10]:
            predictions.append(predict_future(model, scaler, data[len(data) - 60:].values, n=nv))
        with open(name + "_predictions.txt", 'w+') as f:
            f.write("Prediction 1: {}\n".format(predictions[0]))
            f.write("Prediction 5: {}\n".format(predictions[1]))
            f.write("Prediction 10: {}\n".format(predictions[2]))
            print("Prediction 1: {}\n".format(predictions[0]))
            print("Prediction 5: {}\n".format(predictions[1]))
            print("Prediction 10: {}\n".format(predictions[2]))

if __name__ == "__main__":
    main()