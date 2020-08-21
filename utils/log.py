# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging, os
logger = logging.getLogger()


def init_logger(log_dir=None, log_name=None, log_file_level=logging.NOTSET):
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    log_file = log_dir+log_name
    if not os.path.exists('/'.join(log_file.split('/')[:-1])):
        os.mkdir('/'.join(log_file.split('/')[:-1]))
        
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger