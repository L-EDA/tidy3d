#!/usr/bin/env python3
import time
from threading import Thread


def factorize(nu):
    for i in range(1, nu + 1):
        if nu % i == 0:
            yield i


class FactorizeThread(Thread):
    def __init__(self, nu):
        super().__init__()
        self.nu = nu
        self.factors = list(factorize(self.nu))


if __name__ == '__main__':
    # series
    numbers = [1536503, 1395999, 1137837, 1162931, 1182120, 1421341, 1543087, 1741235, 1516637, 1852285]
    start = time.time()
    for number in numbers:
        list(factorize(number))
    delta = time.time() - start
    print(f'Took {delta:.3f} seconds')

    # thread parallel
    start = time.time()
    threads = []
    for number in numbers:
        thread = FactorizeThread(number)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    delta = time.time() - start
    print(f'threads Took {delta:.3f} seconds')