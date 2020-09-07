"""
MIT - CSAIL - Gifford Lab - seqgra

Class with miscellaneous helper functions as static methods

@author: Konstantin Krismer
"""
import os
import shutil


class MiscHelper:
    @staticmethod
    def prepare_path(path: str, allow_exists: bool = True,
                     allow_non_empty: bool = False) -> str:
        path = path.replace("\\", "/").replace("//", "/").strip()
        if not path.endswith("/"):
            path += "/"

        if os.path.exists(path):
            if not allow_non_empty:
                if not os.path.isdir(path):
                    raise Exception("directory cannot be created "
                                    "(file with same name exists)")
                elif len(os.listdir(path)) > 0:
                    num_files: int = len([name
                                          for name in os.listdir(path)
                                          if os.path.isfile(path + name)])
                    if num_files > 0:
                        if not allow_exists:
                            raise Exception("directory cannot be created "
                                            "(non-empty folder with same "
                                            "name exists)")
                    else:
                        shutil.rmtree(path)
                        os.makedirs(path)
        else:
            os.makedirs(path)

        return path

    @staticmethod
    def print_progress_bar(iteration: int, total: int, prefix: str = "",
                           suffix: str = "", decimals: int = 1,
                           length: int = 100, fill: str = "█",
                           print_end: str = "\r"):
        """Call in a loop to create terminal progress bar

        Arguments:
            iteration (int): current iteration
            total (int): total iterations
            prefix (str, optional): prefix string, defaults to empty string
            suffix (str, optional): suffix string, defaults to empty string
            decimals (int, optional): positive number of decimals in percent
                complete, defaults to 1
            length (int, optional): character length of bar, defaults to 100
            fill (str, optional): bar fill character, defaults to "█"
            print_end (str, optional): end character (e.g. "\r", "\r\n"),
                defaults to "\r"
        """
        if not hasattr(MiscHelper.print_progress_bar, "previous_bar"):
            MiscHelper.print_progress_bar.previous_bar: str = ""
        if not hasattr(MiscHelper.print_progress_bar, "previous_percent"):
            MiscHelper.print_progress_bar.previous_percent: str = ""

        if total < 1:
            total = 1
        percent: str = ("{0:." + str(decimals) +
                        "f}").format(100 * (iteration / float(total)))
        filled_length: int = int(length * iteration // total)
        progress_bar: str = fill * filled_length + \
            "-" * (length - filled_length)
        if MiscHelper.print_progress_bar.previous_bar != progress_bar or \
                MiscHelper.print_progress_bar.previous_percent != percent:
            print("\r%s |%s| %s%% %s" %
                  (prefix, progress_bar, percent, suffix), end=print_end)
            MiscHelper.print_progress_bar.previous_bar = progress_bar
            MiscHelper.print_progress_bar.previous_percent = percent

        if iteration == total:
            print()

    @staticmethod
    def read_config_file(file_name: str) -> str:
        with open(file_name.strip()) as f:
            config: str = f.read()
        return config

    @staticmethod
    def format_output_dir(output_dir: str) -> str:
        output_dir = output_dir.strip().replace("\\", "/")
        if not output_dir.endswith("/"):
            output_dir += "/"
        return output_dir

    @staticmethod
    def get_valid_file(data_file: str) -> str:
        data_file = data_file.replace("\\", "/").replace("//", "/").strip()
        if os.path.isfile(data_file):
            return data_file
        else:
            raise Exception("file does not exist: " + data_file)
