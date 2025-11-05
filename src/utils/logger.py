import logging

def get_logger(level):
    logger = logging.getLogger()
    if level == 'info':
        level = logging.INFO
    else:
        level = logging.DEBUG

    logfmt_str = "%(asctime)s %(levelname)s %(message)s"
    formatter = logging.Formatter(logfmt_str)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(level)

    logger.addHandler(streamHandler)
    logger.setLevel(level)
    return logger