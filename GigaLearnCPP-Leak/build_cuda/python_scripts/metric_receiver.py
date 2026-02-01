def init():
    print("MetricSender: Running in dummy mode (wandb disabled)")
    return True


def send_metric(*args, **kwargs):
    pass


def close():
    pass
