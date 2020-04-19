
class ReadWriteFiles(object):
    import numpy as np
    import os

    def __init__(self, *args, **kwargs):
        print(" Class-> ReadWriteFiles Dan")

        self._path = args[0]+"/" if len(args[0]) >0 else ""

        self._name_dir = self.os.walk(self._path).__next__()[1]
        self.name_dir = [self._path + item for item in self._name_dir]

    def ReadTextBasa(self, name_dir):
        name_file=name_dir+".txt"
        path_to_file= self._path + name_dir+ "/" + name_file

        with open(path_to_file, encoding='utf-8-sig') as f:     #utf-8-sig   #utf-8
            myList = [ line.split("\n")[0].lower() for line in f ]
        return myList
            
    def SaveFilePckle(self, name, data):
        import pickle
        with open(self._path+name+'.pickle', 'wb') as f:
            pickle.dump(data, f)

    def LoadFilePckle(self, name):
        import pickle
        with open(self._path+name+'.pickle', 'rb') as f:
            return pickle.load(f)

