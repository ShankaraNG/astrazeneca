import logging,yaml,os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_logger(name):

    logfilepath = "logs"
    logfile= "application.log"
    log_file = os.path.join(BASE_DIR,logfilepath,logfile)
    os.makedirs(os.path.dirname(log_file),
                exist_ok=True)
    logger=logging.getLogger(name)
    logger.setLevel('INFO')
    formatter=logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh=logging.FileHandler(log_file)
    sh=logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
