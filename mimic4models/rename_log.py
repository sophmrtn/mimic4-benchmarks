import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, nargs='+')
    args = parser.parse_args()

    if not isinstance(args.log, list):
        args.log = [args.log]

    for log in args.log:
        if log.find("renamed") != -1:
            print(f"{log} is already renamed by hand, skipping...")
            continue
        if os.path.isdir(log):
            print(f"{log} is a directory, skipping...")
            continue
        with open(log) as logfile:
            text = logfile.read()
            ret = re.search("==> model.final_name: (.*)\n", text)
            if ret is None:
                print(f"No model.final_name in log file: {log}. Skipping...")
                continue
            name = ret.group(1)

        dirname = os.path.dirname(log)
        new_path = os.path.join(dirname, f"{name}.log")
        os.rename(log, new_path)

if __name__ == '__main__':
    main()
