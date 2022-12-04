import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide a path")
    filePath = sys.argv[1]

    with open(filePath, "r+") as fp:
        content = fp.read()
        content = content.replace('href="../', 'href="/').replace('src="../', 'src="/')
        fp.seek(0)
        fp.truncate()

        fp.write(content)
