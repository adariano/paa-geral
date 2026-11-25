import sys

from paa_rag.cli import main


if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
