import time


class TimeMeasure:
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __enter__(self):
        self.start_time = time.time()
        print(f"Start {self.msg}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print(f"{self.msg} finished in {self.interval:.2f}s")
