import os

def check_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
