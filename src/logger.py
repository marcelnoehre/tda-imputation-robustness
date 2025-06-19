import datetime

def _current_time():
    """
    Get the current time in the format of [YYYY-MM-DD HH:MM:SS].

    :return str: Current timestamp
    """
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def log(msg):
    """
    Log a information with timestamp

    :param str msg: The message to be logged
    """
    print(f"\x1b[32;20m{_current_time()}\x1b[37;20m | \x1b[0m{msg}")