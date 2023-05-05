"""Test models and model library for capstone project."""
from src.models import *
from src.pipeline import load_testing_data, predict, calculate_confusion_matrix
from tensorflow.keras.models import Model
from numpy import ndarray
from tensorflow.keras.losses import CategoricalCrossentropy
import pytest

def test_model_compilation():
    """Test whether the model compiles successfully or not."""
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())

@pytest.fixture
def create_test_data():
    """Loads the data for testing purposes."""
    filename = "data/CMMD-set/test.csv"
    tdata = load_testing_data(filename, sample_size=10)
    return tdata


def test_model_output(create_test_data):
    """Test whether the output is recorded within an array."""
    filename = "data/CMMD-set/test.csv"
    tdata = load_testing_data(filename, sample_size=10)
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())
    pdf = predict(tdata, model)
    assert type(pdf['score'].values[0]) == ndarray


def test_model_score_probability(create_test_data):
    """Test whether the output of the models are a set of probabilities."""
    filename = "data/CMMD-set/test.csv"
    tdata = load_testing_data(filename, sample_size=10)
    inputs, output = tumor_classifier(1147,957)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='Adam', loss=CategoricalCrossentropy())
    pdf = predict(tdata, model)
    assert sum(pdf['score'].values[0]) == pytest.approx(1.0,.0001)


def test_model_accuracy(create_test_data): #This test is not failing
    """Test whether the accuracy of the most recent model reaches the standard."""
    filename = "data/CMMD-set/test.csv"
    tdata = load_testing_data(filename, sample_size=100)
    pdf = predict(tdata, 'models/tclass_VGG8')
    ct, metrics = calculate_confusion_matrix(pdf)
    assert metrics['Accuracy'] >= 0.90

if __name__ == "__main__":
    pytest.main()
