from inference import predict_species


def test_prediction_is_valid():
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    prediction = predict_species(sample)

    assert prediction in ["setosa", "versicolor", "virginica"]


def test_known_setosa_prediction():
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    assert predict_species(sample) == "setosa"


def test_prediction_from_list():
    sample = [[5.1, 3.5, 1.4, 0.2]]

    assert predict_species(sample) == "setosa"
