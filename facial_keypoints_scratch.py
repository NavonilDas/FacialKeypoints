# For Reading CSV
import csv
import numpy as np
import pickle
# For Checking File Exist or not
import os.path

# Sequential Model
from tensorflow.keras.models import Sequential
# Separable Convolution 2D for Speed
# Apply Activation Function
from tensorflow.keras.layers import BatchNormalization, Activation, SeparableConv2D
# Optimizer
from tensorflow.keras.optimizers import Adam
# Loss Function
from tensorflow.keras.losses import mean_squared_error

TRAINING_FILE = 'dataset/training.csv'
TESTING_FILE = 'dataset/test.csv'

EXTRACT_TRAIN_OUT_FILE = 'saved/extracted_train_output.dmp'
EXTRACT_TRAIN_IN_FILE = 'saved/extracted_train_input.dmp'

SAVED_MODEL = 'saved/facial_keypoints_scratch_model.h5'

def fetch_training_data():
    """
    Function Reads the Training File and extract information and saves it to a dump.
    """
    headers = []
    train_output = []
    train_input = []
    # Check if training data is present
    if not os.path.isfile(TRAINING_FILE):
        print('No Training Data is Present')
        exit()

    with open(TRAINING_FILE, 'r') as fin:
        reader = csv.reader(fin)
        headers = reader.__next__()
        for row in reader:
            # Seprate features and image data
            features = row[:30]

            # Check if all The features are available
            try:
                features = [float(x) for x in features]
            except:
                continue
            # All Features are available
            train_output.append(features)

            # For Image
            img = row[30:][0]
            img = img.split(' ')
            img = [int(x) for x in img]
            img = np.array(img).reshape(96, 96)
            train_input.append(img)

    # Conver From List to numpy array as for the shape (our output layer is not a dense layer)
    train_output = np.array(train_output).reshape(-1, 1, 1, 30)
    # Convert input list to numpy array
    train_input = np.array(train_input).reshape(-1, 96, 96, 1)

    # Now We have to normalize the data

    # For Image We know that the number will be in range 0 - 255
    train_input = train_input / 255

    # For Features we know that our image width and height are in the range of 1 - 96
    train_output = train_output / 96

    # Save The Extracted Data To Preserve some Time
    with open(EXTRACT_TRAIN_IN_FILE, 'wb') as fin:
        pickle.dump(train_input, fin)

    with open(EXTRACT_TRAIN_OUT_FILE, 'wb') as fin:
        pickle.dump(train_output, fin)

    return (headers, train_input, train_output)


def get_training_data():
    """
    Load Training Data From Saved Dump otherwise get training data from csv.
    """
    train_output = []
    train_input = []
    if not os.path.isfile(EXTRACT_TRAIN_OUT_FILE) and not os.path.isfile(EXTRACT_TRAIN_IN_FILE):
        fetch_training_data()

    with open(EXTRACT_TRAIN_IN_FILE, 'rb') as fin:
        train_input = pickle.load(fin)

    with open(EXTRACT_TRAIN_OUT_FILE, 'rb') as fin:
        train_output = pickle.load(fin)

    if isinstance(train_output, list) or isinstance(train_output, list):
        print('Something Went Wrong')
        exit()
    return (train_input, train_output)


def start_traing(x_train, y_train):
    model = Sequential([
        # 2D Convolution Layer with 96x96 input 128 filters & 5x5 window size
        SeparableConv2D(128, 5, input_shape=(96, 96, 1)),
        # Apply Normalization
        BatchNormalization(),
        # Apply Activation Function
        Activation('relu'),

        # Another Convolution Layer of 128 Filters & 3x3 window size with 2 strides
        SeparableConv2D(128, 3, strides=2, activation='relu'),
        BatchNormalization(),

        SeparableConv2D(128, 3, activation='relu'),
        SeparableConv2D(128, 3, activation='relu'),
        BatchNormalization(),

        SeparableConv2D(128, 3, activation='relu'),
        SeparableConv2D(128, 3, activation='relu'),
        BatchNormalization(),

        SeparableConv2D(64, 3, activation='relu'),
        SeparableConv2D(64, 3, activation='relu'),
        BatchNormalization(),

        SeparableConv2D(32, 3, activation='relu'),
        SeparableConv2D(32, 3, activation='relu'),
        BatchNormalization(),

        SeparableConv2D(30, 3, activation='relu'),
        # Output Layer
        SeparableConv2D(30, 3, activation='sigmoid'),
    ])

    # Print Summary of the model
    model.summary()
    # Compile Model with mean squared loss and adam optimizer
    model.compile(loss=mean_squared_error,
                  optimizer=Adam(learning_rate=0.0001), metrics=['mse'])

    # Train the model
    model.fit(x=x_train, y=y_train, epochs=25,
          batch_size=50)

    # Save The Model
    model.save(SAVED_MODEL)


x_train, y_train = get_training_data()

print('Training Input', x_train.shape)
print('Training Output', y_train.shape)

if not os.path.isfile(SAVED_MODEL):
    start_traing(x_train, y_train)
else:
    pass