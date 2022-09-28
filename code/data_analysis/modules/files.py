from pathlib import Path


def get_filenames(path_data, session=None, group=None, game_number=None, rule=None, map_type=None, map_number=None):
    for file in Path(path_data).glob("session_*/out/*.csv"):
        if (
            _condition_true(int(file.stem[1:3]), session)
            and _condition_true(file.stem[4], group)
            and _condition_true(int(file.stem[5]), game_number)
            and _condition_true(int(file.stem[8]), rule)
            and _condition_true(file.stem[11], map_type)
            and _condition_true(int(file.stem[13:15]), map_number)
        ):
            yield file.stem


def _condition_true(a, b):
    if b is None:
        return True
    try:
        return a in b
    except TypeError:
        return a == b
