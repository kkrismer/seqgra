"""
MIT - CSAIL - Gifford Lab - seqgra

Class with miscellaneous helper functions as static methods

@author: Konstantin Krismer
"""
import os


class MiscHelper:
    @staticmethod
    def prepare_path(path: str, allow_exists: bool = True) -> str:
        path = path.replace("\\", "/").replace("//", "/").strip()
        if not path.endswith("/"):
            path += "/"

        if os.path.exists(path):
            if not os.path.isdir(path):
                raise Exception("directory cannot be created "
                                "(file with same name exists)")
            elif not allow_exists:
                raise Exception("directory cannot be created "
                                "(folder with same name exists)")
        else:
            os.makedirs(path)

        return path
