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


def draw_plot(
    x1,
    y1,
    x2=None,
    y2=None,
    x3=None,
    y3=None,
    chart_title='График',
    label1=None,
    label2=None,
    label3=None,
    path_to_save='plot'
) -> None:
    chart_title = chart_title
    figure, axes = plt.subplots(1, 1, figsize=(16, 10))
    axes.grid(color='lightgray', which='both', zorder=0)

    axes.plot(x1, y1, label=label1, color='#03bcff')

    if x2 and y2:
        axes.plot(x2, y2, label=label2, color='#07b836')

    if x3 and y3:
        axes.plot(x3, y3, label=label3, color='#d6046a')

    axes.set_title(chart_title)
    axes.set_xlabel('Время, сек.')
    axes.set_ylabel('Значение')
    axes.legend()

    if not os.path.exists(ACCURACY_CHARTS_DIR_NAME):
        os.mkdir(ACCURACY_CHARTS_DIR_NAME)

    figure.savefig(path_to_save)

    plt.close()
