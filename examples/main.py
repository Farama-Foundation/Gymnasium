import file
import func
import game

#key = game.Environments.ATARI
#key = game.Environments.BOX_2D
key = game.Environments.CLASSIC_CONTROL
#key = game.Environments.MUJO_CO
#key = game.Environments.TOY_TEXT


if __name__ == '__main__':
    print("Starting")
    folder: str = game.Folders[key]
    path: str = f"{folder}/files.txt"
    lines: list = file.init(path)
    name: str = file.load(lines)
    if len(name) > 0:
        print(f"Running '{name}'")
        func.run_game(name)
    print("The end!")