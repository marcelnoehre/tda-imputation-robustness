import datetime
import threading
import time

def _current_time():
    """
    Get the current time in the format of [YYYY-MM-DD HH:MM:SS].

    :return str: Current timestamp
    """
    return datetime.datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')

def log(msg):
    """
    Log a information with timestamp

    :param str msg: The message to be logged
    """
    print(f'\x1b[32;20m{_current_time()}\x1b[37;20m | \x1b[0m{msg}', flush=True)

def _duration(seconds):
    """
    Format a duration as e.g. '45s', '12m 03s' or '3h 07m'.

    :param float seconds: The duration in seconds

    :return str: Formatted duration
    """
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m {seconds % 60:02d}s'
    return f'{seconds // 3600}h {seconds % 3600 // 60:02d}m'

def timed(function, *args):
    """
    Run a task and measure its duration. Module-level, so that worker processes can run it.

    :return tuple: (result, seconds)
    """
    start = time.time()
    return function(*args), time.time() - start

class Progress:
    """
    Progress of one stage of an experiment: logs the number of tasks, every task slower than
    `slow_task` seconds (with its label), and every `interval` seconds, also while no task
    finishes, the number of finished tasks with the elapsed and estimated remaining time.
    Cached tasks finish almost instantly, so the remaining time is only a rough guess while
    the cache is being read.
    """
    def __init__(self, stage, total):
        from src.config import PROGRESS_INTERVAL, SLOW_TASK
        self.stage, self.total, self.done = stage, total, 0
        self.slow_task = SLOW_TASK
        self.start = time.time()
        self._stop = threading.Event()
        if total:
            log(f'{stage}: {total} task(s)')
            threading.Thread(target=self._heartbeat, args=(PROGRESS_INTERVAL,), daemon=True).start()

    def _heartbeat(self, interval):
        while not self._stop.wait(interval):
            self._report()

    def _report(self):
        elapsed = time.time() - self.start
        remaining = f', remaining ~{_duration(elapsed / self.done * (self.total - self.done))}' if 0 < self.done < self.total else ''
        log(f'{self.stage}: {self.done}/{self.total} done ({100 * self.done / self.total:.0f}%), elapsed {_duration(elapsed)}{remaining}')

    def __call__(self, timed_result, *label):
        """
        Register a finished task.

        :param tuple timed_result: (result, seconds) as returned by `timed`
        :param label: Parts of the task label (dataset, mechanism, rate, method, ...)

        :return: The result of the task
        """
        result, seconds = timed_result
        self.done += 1
        if seconds >= self.slow_task:
            log(f'{self.stage}: {" | ".join(map(str, label))} took {_duration(seconds)}')
        if self.done == self.total:
            self._stop.set()
            self._report()
        return result
