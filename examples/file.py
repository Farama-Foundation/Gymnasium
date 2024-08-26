def init(path: str) -> list:
    with open(path, 'rt') as file_handle:
        return file_handle.readlines()


def load(lines: list) -> str:
    name: str = ""

    for line in lines:
        line = line.strip()
        if not str.startswith(line, '#'):
            name = line
            break

    return name
