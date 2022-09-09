import threading
from queue import Queue

import recognition as r
import webservice.run as server


recognition = r.Recognition(0)
recognition.run(rects=False, window=False)
recognition.deinit()

if __name__ == "__main__":
    to_server = Queue()
    from_server = Queue()

    # Start server in a thread
    threading.Thread(target=server.run, args=())

