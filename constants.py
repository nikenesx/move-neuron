class MoveTypes:
    """Класс с константами типов движений"""

    FIST = 'fist'
    LEAD = 'lead'
    REST = 'rest'
    CAST = 'cast'
    EXTENSION = 'extension'
    BENDING = 'bending'

    TRANSLATIONS = {
        FIST: 'Кулак',
        LEAD: 'Отведение',
        REST: 'Покой',
        CAST: 'Приведение',
        EXTENSION: 'Разгибание',
        BENDING: 'Сгибание',
    }

    NUMS = {
        FIST: 0,
        LEAD: 1,
        REST: 2,
        CAST: 3,
        EXTENSION: 4,
        BENDING: 5,
    }
