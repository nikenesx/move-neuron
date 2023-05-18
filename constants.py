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
        REST: 0,
        FIST: 1,
        BENDING: 2,
        EXTENSION: 3,
        CAST: 4,
        LEAD: 5,
    }

    ALL_TYPES = (FIST, LEAD, REST, CAST, EXTENSION, BENDING)
