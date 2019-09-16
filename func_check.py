# from tensorflow.python.client import device_lib
from keras.models import load_model


# print(device_lib.list_local_devices())

imdb_raw = load_model("models/imdb_raw.h5")
imdb_raw.summary()