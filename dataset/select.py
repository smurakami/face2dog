import glob
import os

for file in glob.glob('edge_selected/*'):
    print(file)
    basename = os.path.basename(file)
    print(basename)

    os.system('cp processed/%s selected/%s' % (basename, basename))