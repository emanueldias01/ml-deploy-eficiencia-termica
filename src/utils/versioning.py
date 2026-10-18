from datetime import datetime


def get_date():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_version():
    return get_date()
