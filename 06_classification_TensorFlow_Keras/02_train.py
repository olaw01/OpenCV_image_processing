import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from datetime import datetime #zapisanie  jako znacznik daty i czasu
import pandas as pd
import argparse #biblioteka do przekazywania dokumentow podczas uruchamiania skryptow
import pickle #zachowanie obiektow, ktore stworzymy np slownik kodow i etykiet
import os

#plotly - wygenerowanie raportu trenowania (strata i dokladnosc trenowania)

# suppress logs
import warnings # ograniczenie logow printowanych do konsoli
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # kla pozwala na augumentacje danych
from tensorflow.keras.callbacks import ModelCheckpoint # pozwala zapisac najlepszy model podczas trenowania
from tensorflow.keras.optimizers import Adam # optymalizator
from architecture import models # z naszej architektury zaimportujemy modul models (LenNet5)

print(f'Tensorflow version: {tf.__version__}')

# Przykład uruchomienia:
# $ python 02_train.py -e 20
# dodawanie arugmentow
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=1, help='Określ liczbę epok', type=int)
args = vars(ap.parse_args())

#stale
MODEL_NAME = 'LeNet5'
LEARNING_RATE = 0.001
# liczbe epok - przekazujemy
EPOCHS = args['epochs']
BATCH_SIZE = 32
#format danych 150 na 150 px z kanalem 3
INPUT_SHAPE = (150, 150, 3)
TRAIN_DIR = './images/train'
VALID_DIR = './images/valid'

# funkcja oparta o plotly - przygotuje wykres z procesu uczenia
def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))

    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines', marker_color='#f29407'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines', marker_color='#0771f2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines', marker_color='#f29407'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines', marker_color='#0771f2'), row=2, col=1)

    fig.update_xaxes(title_text='Liczba epok', row=1, col=1)
    fig.update_xaxes(title_text='Liczba epok', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title=f"Metrics: {MODEL_NAME}")

    po.plot(fig, filename=filename, auto_open=False)



train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1. / 255.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
#augumentacja danych
valid_datagen = ImageDataGenerator(rescale=1. / 255.)

#generator zbioru treningowego
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
#generator zbioru walidacyjnego
valid_generator = valid_datagen.flow_from_directory(
    directory=VALID_DIR,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

architectures = {MODEL_NAME: models.LeNet5}
architecture = architectures[MODEL_NAME](input_shape=INPUT_SHAPE)
# budowanie modelu build() zaimplementowana w katalogu architecture w module modules
model = architecture.build()

#kompilujemy model uzywajac wskaznika uczenia i starty binary_crossentropy, po0niewaz mamy klasyfikacje binarna
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# podsumowanie modelu
model.summary()

# pobranie znacznika daty i czasu oraz zapis modelu
dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
os.makedirs("output", exist_ok=True)

filepath = os.path.join("output", f"model_{dt}.keras")
checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_accuracy", save_best_only=True)


# trenowanie modelu
print('[INFO] Trenowanie modelu...')
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint],
)


print('[INFO] Eksport wykresu do pliku html...')
filename = os.path.join('output', 'report_' + dt + '.html')
plot_hist(history, filename=filename)

print('[INFO] Eksport etykiet do pliku...')
with open(r'output\labels.pickle', 'wb') as file:
    file.write(pickle.dumps(train_generator.class_indices))

print('[INFO] Koniec')

# Uruchamianie z poziomu terminala
# cd ..
# cd 06_classification_TensorFlow_Keras
# ls -l
# python 02_train.py -e 15 / (15 epok)
