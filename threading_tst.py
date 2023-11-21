import threading
import time

a = 1
lock = threading.Lock()
def fun1():
    global a
    while True:
        time.sleep(3)
        with lock:
            a += 1 

def run():
    global a
    while True:
            with lock:
                print("first:%d"%a)
            # time.sleep(0.001)


def main():
    thread1 = threading.Thread(target=fun1)
    thread1.start()
    run()

if __name__ == '__main__':
    main()