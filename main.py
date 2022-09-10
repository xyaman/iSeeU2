import threading, os, logging, time
from queue import Queue

import recognition
import constants
import server
import raspio

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

if __name__ == "__main__":
    to_server = Queue()
    from_server = Queue()

    # Start server in a thread
    server_thread = threading.Thread(target=server.run, args=(to_server, from_server,))
    server_thread.start()

    # Start gpio (Loads some variables)
    raspio.setup()

    # Create recognition entity
    r = recognition.Recognition(recognition.CAMERA)

    # server_thread.join() # TODO: Remove
    while True:
        # Interpret signals
        if not from_server.empty():
            recv = from_server.get()
            if recv == constants.DATABASE_UPDATED:
                # Probably going to hang out a little
                r.update(os.path.dirname(__file__) + "/webserver/static/images/")

        # IO stuff
        if raspio.get_state(raspio.INPUT1):
            pass
        if raspio.get_state(raspio.INPUT2):
            pass

        # Recognition stuff
        r.run_once(rects=False)
        print(r.recognized)
        time.sleep(0.5)
    
    # deinit stuff (when program ends)
    r.deinit()
