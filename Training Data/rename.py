import os
import argparse

parser = argparse.ArgumentParser(description='rename all files in a directory into an ordered list')
parser.add_argument('directory', action="store", type=str, help="directory of files to order")
parser.add_argument('template', action="store", type=str, help="rename files to this template")
parser.add_argument('--start', action="store", type=int, dest="start", default=0, help="number of first file")

arg = parser.parse_args()

file_list = os.listdir(arg.directory)

for i in range(arg.start, len(file_list) + arg.start):
	os.rename(arg.directory + file_list[i-arg.start], arg.template % i)
print("done")
