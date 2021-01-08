from os import listdir
from os.path import isfile, join
import tarfile
import os

path = "."
files = [f for f in listdir(path) if isfile(join(path, f))]

gz_files = [f for f in files if f.endswith('.text.tar.gz')]

current_path = os.path.dirname(os.path.realpath(__file__))

for gz_f in gz_files:
    gz_path = "{}/{}".format(current_path, gz_f)
    print "opening file", gz_path
    with tarfile.open(gz_path) as archive:
        print archive.is_reg