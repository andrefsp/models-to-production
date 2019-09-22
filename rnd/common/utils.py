import os


def normpath(path):
    if path.startswith('gs://'):
        return "gs://%s" % os.path.normpath(path[len("gs://"):])
    else:
        return os.path.normpath(path)
