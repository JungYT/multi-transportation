import logging


def set_logger(path, file):
    """[Create a log file to record the experiment's logs]

    Arguments:
        path {string} -- path to the directory
        file {string} -- file name

    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = path.joinpath(file)

    path.mkdir(parents=True, exist_ok=True)
    log_file.touch()

    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logger = logging.getLogger("logs")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(file_logging_format))

    logger.addHandler(fh)
