
import argparse
import subprocess
from datetime import date, timedelta
import glob
import os

current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("--past", type=int, default=0, required=False)
    parser.add_argument("--python", action="store_true", required=False)
    parser.add_argument("--new", action="store_true", required=False)

    args = parser.parse_args()
    past = args.past

    if args.new:
        current_date = date.today().strftime("%Y%m%d")
        file = f"{current_dir_path}/notes/note_" + current_date + (".py" if args.python else ".txt")
        subprocess.run("code "+file, shell=True)
        exit(0)

    files_pattern = f"{current_dir_path}/notes/*" + (".py" if args.python else ".txt")


    notes = sorted(glob.glob(files_pattern), reverse=True)
    if len(notes)<=past:
        exit(0)
    subprocess.run("code "+notes[past], shell=True)






