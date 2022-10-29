from enum import Enum, unique


@unique
class Languages(Enum):
    def __init__(self, code, full_name):
        self.code = code
        self.full_name = full_name

    RU = ('ru', 'Russian')
    CH = ('ch', 'Chinese')
    EN = ('en', 'English')
    UN = ('unknown', 'Unknown')
