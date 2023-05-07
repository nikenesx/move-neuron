import os

from keras import Sequential
from keras.callbacks import History
from matplotlib import pyplot as plt

LOSS_CHARTS_DIR_NAME = 'loss_charts/'
ACCURACY_CHARTS_DIR_NAME = 'accuracy_charts/'


def save_accuracy_chart(fitted_model: Sequential, model_history: History, batch_size: int) -> None:
    history = model_history.history
    chart_title = 'Точность'
    train_data_label = 'Обучающая выборка'

    epochs_count = len(history['accuracy'])
    train_data_values = history['accuracy'][1:]

    epochs = range(1, epochs_count)
    figure, axes = plt.subplots(1, 1, figsize=(16, 10))
    axes.grid(color='lightgray', which='both', zorder=0)

    axes.plot(epochs, train_data_values, label=train_data_label, color='#03bcff')

    axes.set_title(chart_title)
    axes.set_xlabel('Эпохи')
    axes.set_ylabel(chart_title)
    axes.legend()

    if not os.path.exists(ACCURACY_CHARTS_DIR_NAME):
        os.mkdir(ACCURACY_CHARTS_DIR_NAME)

    neurons_count = '-'.join([str(layer.output.shape[1]) for layer in fitted_model.layers])
    figure.savefig(f'{ACCURACY_CHARTS_DIR_NAME}/{neurons_count}_{batch_size}_{epochs_count}_accuracy.jpg')

    plt.close()


def save_loss_chart(fitted_model: Sequential, model_history: History, batch_size: int) -> None:
    """
    Сохраняет изображения с графиком потерь.
    :param fitted_model: обученная модель
    :param model_history: история обучения модели
    """
    history = model_history.history
    chart_title = 'Потери'
    train_data_label = 'Обучающая выборка'
    validation_data_label = 'Выборка валидации'

    epochs_count = len(history['loss'])
    train_data_values = history['loss'][1:]
    validation__data_values = history['val_loss'][1:]

    epochs = range(1, epochs_count)
    figure, axes = plt.subplots(1, 1, figsize=(16, 10))
    axes.grid(color='lightgray', which='both', zorder=0)

    axes.plot(epochs, train_data_values, label=train_data_label, color='#03bcff')
    axes.plot(epochs, validation__data_values, label=validation_data_label, color='#e06704')

    axes.set_title(chart_title)
    axes.set_xlabel('Эпохи')
    axes.set_ylabel(chart_title)
    axes.legend()

    if not os.path.exists(LOSS_CHARTS_DIR_NAME):
        os.mkdir(LOSS_CHARTS_DIR_NAME)

    neurons_count = '-'.join([str(layer.output.shape[1]) for layer in fitted_model.layers])
    figure.savefig(f'{LOSS_CHARTS_DIR_NAME}/{neurons_count}_{batch_size}_{epochs_count}_loss.jpg')

    plt.close()
