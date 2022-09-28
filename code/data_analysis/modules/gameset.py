from modules.files import get_filenames
from modules.game import Game


class GameSet(set):
    def __init__(
        self,
        path_data,
        session=None,
        group=None,
        game_number=None,
        rule=None,
        map_type=None,
        map_number=None,
    ):
        super().__init__()
        for filename in get_filenames(
            path_data,
            session=session,
            group=group,
            game_number=game_number,
            rule=rule,
            map_type=map_type,
            map_number=map_number,
        ):
            self.add(Game(path_data=path_data, filename=filename))


if __name__ == "__main__":

    path_data = "./data/"

    print(GameSet(path_data))
    print(GameSet(path_data, session=1, rule=1, map_type="R", map_number=1, group="A"))
    print(GameSet(path_data, map_type="R", rule=2))
    print(GameSet(path_data, session=1))
    print(GameSet(path_data, session=[1, 5]))
    print(GameSet(path_data, session={1, 5}))
    print(GameSet(path_data, session=range(11)))
