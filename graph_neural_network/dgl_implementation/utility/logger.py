from dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity, default_metric_format
import datetime
import os


def step_format(step):
    if type(step) != tuple:
        return str(step)
    if len(step) == 0:
        return str(step)
    else:
        return "{} epoch {:03d}".format(step[1], step[0])


def no_time_prefix_format(timestamp):
    return "DLL - "


def external_metrics_format_wrapper(fn=default_metric_format):
    def __wrapped(metric, metadata, value):
        if "time" in metric or "memory" in metric:
            return ""
        else:
            return fn(metric, metadata, value)
    return __wrapped


class IntegratedLogger:
    def __init__(self, proc_id, add_timer, no_debug=False, print_only=False):
        if proc_id == 0:
            stdbackend = StdOutBackend(
                Verbosity.DEFAULT,
                step_format=step_format,
                metric_format=
                (lambda metric, metadata, value: f"| {default_metric_format(metric, metadata, value)}")
            )

            if not add_timer:
                stdbackend.prefix_format = no_time_prefix_format
                stdbackend.metric_format = external_metrics_format_wrapper(stdbackend.metric_format)

            jsonbackend = JSONStreamBackend(Verbosity.VERBOSE, (os.getenv("RESULTS_DIR", "/results") + "/log.json"))

            if not print_only:
                self.logger = Logger(backends=[stdbackend, jsonbackend])
            else:
                self.logger = Logger(backends=[stdbackend, ])
        else:
            self.logger = None

        self.add_timer = add_timer
        self.do_debug = not no_debug

    def log(self, *args, **kwargs):
        if self.logger:
            self.logger.log(*args, **kwargs)

    def metadata(self, *args, **kwargs):
        if self.logger:
            self.logger.metadata(*args, **kwargs)

    def debug(self, message, do_log=True):
        if self.do_debug and self.logger and self.add_timer and do_log:
            print(f"[{str(datetime.datetime.now())}] [INFO] {message}")


