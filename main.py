import threading, os, logging, time
from queue import Queue

import recognition
import constants
import server
import raspio
import tts

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

if __name__ == "__main__":
    to_server = Queue()
    from_server = Queue()

    # Start server in a thread
    server_thread = threading.Thread(target=server.run, args=(to_server, from_server,))
    server_thread.start()

    # Create recognition entity
    r = recognition.Recognition(0, load_objects=False)
    
    # Start gpio (Loads some variables)
    raspio.setup()

    while True:
        # IO stuff
        if raspio.get_state(raspio.INPUT1):
            raspio.take(r.capture)
            logging.info("Picture taken")
        if raspio.get_state(raspio.INPUT2):
            logging.info("Key 2 pressed")
            for word in r.recognized:
                logging.info(f"Speaking: {word}")
                tts.speak(word)

        # Interpret signals
        if not from_server.empty():
            recv = from_server.get()
            if recv == constants.DATABASE_UPDATED:
                # Probably going to hang out a little
                r.update(os.path.dirname(__file__) + "/webserver/static/images/")


        # Recognition stuff
        ok, _ = r.run_once(rects=False)
        print(ok, r.recognized)


        time.sleep(0.5)
    
    # deinit stuff (when program ends)
    r.deinit()
