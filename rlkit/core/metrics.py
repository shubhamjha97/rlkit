import logging
from collections import defaultdict


class Metrics:
    def __init__(self, logger_name, log_level, comet_experiment=None):
        self.metrics_dict = defaultdict(defaultdict(dict))
        self.comet_experiment = comet_experiment
        self.logger = logging.get_logger(logger_name)
        self.logger.setLevel() # TODO
        # TODO: initialize logger, take log_level as input

    def log_metric(self, metric_name, metric_val, log_step, namespace="default"):
        self.metrics_dict[namespace][log_step][metric_name] = metric_val
        self.logger.info("STEP {}:{}:{}".format(log_step, metric_name, metric_val))
        # TODO: add timestamp

        if self.comet_experiment is not None:
            self.comet_experiment.log_metric(metric_name, metric_val, step=log_step)

        # TODO: Log metric using logger
