import tkinter as tk
from tkinter import scrolledtext, font
import queue
import threading
import logging

logger = logging.getLogger(__name__)

class TranscriptionUI:
    def __init__(self, ui_config, stop_callback=None):
        self.ui_config = ui_config
        self.title = ui_config.get('title', "Live Transcription")
        self.width = ui_config.get('width', 600)
        self.height = ui_config.get('height', 400)
        self.font_size = ui_config.get('font_size', 12)
        self.stop_callback = stop_callback # Callback to signal main app to stop

        self.root = None
        self.text_area = None
        self.update_queue = queue.Queue()
        self._is_running = False
        self.thread = None

    def _create_ui(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(f"{self.width}x{self.height}")

        # Make the window always on top
        self.root.attributes("-topmost", True)

        # Custom font
        custom_font = font.Font(family="Arial", size=self.font_size)

        self.text_area = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            font=custom_font,
            state='disabled' # Start as read-only
        )
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start a poller for the update queue
        self.root.after(100, self._process_queue)

    def _ui_thread_target(self):
        try:
            self._create_ui()
            logger.info("Transcription UI started.")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in UI thread: {e}", exc_info=True)
        finally:
            logger.info("Transcription UI stopped.")
            self._is_running = False
            if self.stop_callback:
                # Ensure stop_callback is triggered if UI closes unexpectedly or normally
                # But avoid calling if it was already called by _on_closing
                if hasattr(self.root, 'winfo_exists') and self.root.winfo_exists(): # Check if closing was user-initiated
                    pass # _on_closing already called it
                else:
                    self.stop_callback() 

    def start(self):
        if self._is_running:
            logger.warning("UI is already running.")
            return
        self._is_running = True
        self.thread = threading.Thread(target=self._ui_thread_target, daemon=True)
        self.thread.start()

    def stop(self):
        logger.info("Attempting to stop Transcription UI.")
        self._is_running = False # Signal queue processor to stop
        if self.root:
            try:
                # This needs to be called from a thread other than the UI thread if mainloop is active
                # self.root.quit() # Stops mainloop but doesn't destroy window immediately
                self.root.destroy() # Destroys the window
            except tk.TclError as e:
                logger.error(f"Error destroying Tk root: {e}. Might be called from wrong thread or already destroyed.")
            except Exception as e:
                logger.error(f"Generic error stopping UI: {e}")
        self.update_queue.put(None) # Unblock queue poller if it's waiting

        if self.thread and self.thread.is_alive():
            logger.info("Waiting for UI thread to join...")
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                logger.warning("UI thread did not join in time.")
        self.thread = None
        logger.info("UI stop process completed.")

    def _on_closing(self):
        logger.info("UI window closed by user.")
        if self.stop_callback:
            self.stop_callback() # Signal main application to stop
        self.stop() # Clean up the UI itself

    def update_transcription(self, text_segment, append=True):
        if not self._is_running and not self.root : # Check if UI is active or starting up
             # If UI not fully up, try to queue. If queue fails, log and drop.
            try:
                self.update_queue.put(("update", text_segment, append))
            except Exception as e:
                logger.warning(f"UI not ready and queue put failed: {e}. Dropping text: {text_segment}")
            return

        self.update_queue.put(("update", text_segment, append))

    def set_status_message(self, message):
        if not self._is_running and not self.root:
            try:
                self.update_queue.put(("status", message, False))
            except Exception as e:
                logger.warning(f"UI not ready and queue put failed: {e}. Dropping status: {message}")
            return
        self.update_queue.put(("status", message, False)) # Append is false for status

    def _process_queue(self):
        if not self._is_running and not self.update_queue.empty():
            # If stopping, clear the queue gracefully after one last check or if root is gone.
            if not self.root or not self.root.winfo_exists():
                while not self.update_queue.empty():
                    self.update_queue.get_nowait()
                return

        try:
            while not self.update_queue.empty():
                task = self.update_queue.get_nowait()
                if task is None: # Sentinel for stopping the poller
                    continue

                action, content, append_flag = task
                
                if not self.text_area or not self.root or not self.root.winfo_exists():
                    logger.warning("UI text_area not available, requeueing update.")
                    # Re-queue and try again shortly if UI components aren't ready
                    # self.update_queue.put(task) # Careful with re-queuing to avoid infinite loop
                    continue

                self.text_area.config(state='normal')
                if action == "update":
                    if append_flag:
                        self.text_area.insert(tk.END, content + " ")
                    else:
                        self.text_area.delete('1.0', tk.END)
                        self.text_area.insert(tk.END, content)
                    self.text_area.see(tk.END) # Scroll to the end
                elif action == "status":
                    # Could have a dedicated status bar, for now, prepend to text area
                    self.text_area.insert('1.0', f"[{content}]\n")
                self.text_area.config(state='disabled')
                self.update_queue.task_done()

        except queue.Empty:
            pass # No updates to process
        except Exception as e:
            logger.error(f"Error processing UI update queue: {e}", exc_info=True)
        
        if self._is_running and self.root and self.root.winfo_exists():
            self.root.after(100, self._process_queue) # Poll again

    def is_active(self):
        return self._is_running and self.root is not None and self.root.winfo_exists()

if __name__ == '''__main__''':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    dummy_ui_config = {
        "title": "Test UI",
        "width": 500,
        "height": 300,
        "font_size": 14
    }

    main_app_stopped = False
    def app_stop_signal():
        # nonlocal main_app_stopped # Removed: main_app_stopped is in the module-level scope of this block
        global main_app_stopped # Use global if you intend to modify the script-level variable
        print("Main application stop signal received from UI!")
        main_app_stopped = True

    ui = TranscriptionUI(dummy_ui_config, stop_callback=app_stop_signal)
    ui.start()
    print("UI started. Sending test messages...")

    try:
        # Simulate receiving transcriptions
        import time
        ui.update_transcription("Hello, this is a test transcription.")
        time.sleep(1)
        ui.update_transcription("Another segment of text appears here.")
        time.sleep(1)
        ui.set_status_message("STATUS: System Normal")
        time.sleep(1)
        ui.update_transcription("More text...", append=True)
        time.sleep(2)
        ui.update_transcription("This will replace everything.", append=False)
        time.sleep(2)

        # Keep the main thread alive until UI signals to stop or a timeout
        timeout = 10 # seconds
        start_time = time.time()
        while ui.is_active() and not main_app_stopped and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not main_app_stopped and ui.is_active():
            print("Timeout reached, stopping UI from main thread.")
            ui.stop() # If user didn't close, stop it now
        elif main_app_stopped:
            print("UI signalled main app to stop.")
        elif not ui.is_active():
            print("UI stopped on its own or failed to start properly.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping UI.")
        ui.stop()
    except Exception as e:
        print(f"Error in test: {e}")
        logger.error("Error in test script", exc_info=True)
        ui.stop()
    finally:
        if ui.is_active():
            print("Ensuring UI is stopped in finally block.")
            ui.stop() # Ensure cleanup
        print("Test finished.") 