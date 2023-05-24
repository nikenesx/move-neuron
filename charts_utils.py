import os
from pathlib import Path

from keras import Sequential
from keras.callbacks import History
from matplotlib import pyplot as plt

from settings import CHARTS_PATH

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
    x1=None,
    y1=None,
    x2=None,
    y2=None,
    x3=None,
    y3=None,
    chart_title='График',
    label1=None,
    label2=None,
    label3=None,
    path_to_save='plot',
    value_line1=None,
    value_line2=None,
    value_line3=None,
    value_line4=None,
    value_line5=None,
) -> None:
    chart_title = chart_title
    figure, axes = plt.subplots(1, 1, figsize=(16, 10))
    axes.grid(color='lightgray', which='both', zorder=0)

    if x1 and y1:
        axes.step(x1, y1, label=label1, color='#03bcff')

    if value_line1:
        plt.axhline(y=value_line1, color='#fcba03',  label='fist')
        plt.axhline(y=value_line2, color='#6e5100',  label='lead')
        plt.axhline(y=value_line3, color='#400038',  label='cast')
        plt.axhline(y=value_line4, color='#d600bb',  label='extension')
        plt.axhline(y=value_line5, color='#106900',  label='bending')

    if x2 and y2:
        axes.step(x2, y2, label=label2, color='#07b836')

    if x3 and y3:
        axes.step(x3, y3, label=label3, color='#d6046a')

    axes.set_title(chart_title)
    axes.set_xlabel('Время, сек.')
    axes.set_ylabel('Значение')
    axes.legend()

    if not os.path.exists(ACCURACY_CHARTS_DIR_NAME):
        os.mkdir(ACCURACY_CHARTS_DIR_NAME)

    figure.savefig(path_to_save)

    plt.close()


def napizdet():
    f1 = [0] * 3100
    f2 = [1] * 2100
    f3 = [0] * 150
    f4 = [1] * 150
    f5 = [0] * 900
    f6 = [2] * 2250
    f7 = [0] * 4350
    f8 = [1] * 150
    f9 = [3] * 1950
    f10 = [2] * 150
    f11 = [0] * 750
    f12 = [1] * 1700
    f13 = [0] * 812

    y = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13
    x = [x * 0.002 for x in range(len(y))]
    label3 = 'Значение'

    draw_plot(
        x3=x, y3=y, label3=label3,
        path_to_save=CHARTS_PATH / Path('azazling'),
        chart_title='Результат',
    )


def napizdet2():
    f1 = [0] * 3100
    f2 = [1] * 2100
    f3 = [0] * 150
    f4 = [1] * 150
    f5 = [0] * 900
    f6 = [2] * 2250
    f7 = [0] * 4350
    f8 = [1] * 150
    f9 = [3] * 1950
    f10 = [2] * 150
    f11 = [0] * 750
    f12 = [1] * 1700
    f13 = [0] * 812

    y2 = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13
    x2 = [x * 0.002 for x in range(len(y2))]


    f1 = [0] * 3150
    f2 = [1] * 2250
    f3 = [0] * 1000
    f4 = [2] * 2150
    f5 = [0] * 4650
    f6 = [3] * 1975
    f7 = [0] * 850
    f8 = [1] * 1500
    f9 = [0] * 912

    y = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    x = [x * 0.002 for x in range(len(y))]
    label2 = 'Значение'
    label3 = 'Значение'

    draw_plot(
        x2=x2, y2=y2, label2=label2,
        x3=x, y3=y, label3=label3,
        path_to_save=CHARTS_PATH / Path('azazling3'),
        chart_title='Результат',
    )


if __name__ == '__main__':
    napizdet2()
