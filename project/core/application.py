from project.core.environment_variables import environment_variables
from project.shared.meta.observable_singleton import ObservableSingletonMeta
from project.shared.system.torch_util import gpu_is_available
from project.observers.observable import Observable
import logging
import os
pid = os.getpid()
using_gpu = gpu_is_available()

class Application(Observable, metaclass=ObservableSingletonMeta):
    def __init__(self):
        super().__init__()
        self.initialize_logger()

    def initialize_logger(self):
        log_level = logging.DEBUG
        logger = logging.getLogger("logger")
        logger.setLevel(log_level)
        logger.addFilter(lambda record: setattr(record, 'pid', pid) or True)

        self.logger = logger

        self.logger.info("Loading application settings and environment variables")

    envs = environment_variables
