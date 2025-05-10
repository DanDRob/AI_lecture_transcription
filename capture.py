import soundcard as sc
import numpy as np
import queue
import threading
import time
import logging

logger = logging.getLogger(__name__)

class AudioCapture:
    def __init__(self, device_name="default", sample_rate=16000, channels=1, chunk_size=1024):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self._is_running = False
        self._thread = None
        self.recorder = None
        self.microphone = None

        try:
            if self.device_name and self.device_name.lower() != "default":
                logger.info(f"Attempting to use specified audio device: {self.device_name}")
                # Try to find by exact name first
                all_mics = sc.all_microphones(include_loopback=True)
                self.microphone = next((mic for mic in all_mics if mic.name == self.device_name), None)
                
                if not self.microphone:
                    logger.warning(f"Device named '{self.device_name}' not found by exact name. Trying default loopback.")
                    self.microphone = sc.default_microphone() # Fallback or try specific loopback
                    if not self.microphone.isloopback:
                        logger.warning("Default microphone is not a loopback device. Loopback capture might not work as expected.")
                        loopback_mics = [m for m in all_mics if m.isloopback]
                        if loopback_mics:
                            self.microphone = loopback_mics[0] # Pick first available loopback
                            logger.info(f"Selected first available loopback device: {self.microphone.name}")
                        else:
                            logger.error("No loopback devices found. Using system default input. This will likely capture microphone.")
                            self.microphone = sc.default_microphone()

            else: # "default" or None
                logger.info("Attempting to use default loopback audio device.")
                # Prefer default loopback if available
                default_mic = sc.default_microphone()
                if default_mic and default_mic.isloopback:
                    self.microphone = default_mic
                    logger.info(f"Using default loopback device: {self.microphone.name}")
                else:
                    all_loopback_mics = sc.all_microphones(include_loopback=True)
                    loopback_mics = [m for m in all_loopback_mics if m.isloopback]
                    if loopback_mics:
                        self.microphone = loopback_mics[0] # Pick first available loopback
                        logger.info(f"No default loopback. Using first available loopback device: {self.microphone.name}")
                    else:
                        logger.warning("No loopback devices found. Using system default input. This will likely capture your microphone.")
                        self.microphone = default_mic if default_mic else sc.all_microphones()[0] # Fallback to any mic
            
            if not self.microphone:
                raise RuntimeError("No suitable audio input device found.")

            logger.info(f"Selected audio device: {self.microphone.name} (Loopback: {self.microphone.isloopback})")
            logger.info(f"Supported channels: {self.microphone.channels}, Preferred: {self.channels}")
            logger.info(f"Supported samplerates: {self.microphone.samplerates}, Preferred: {self.sample_rate}")

            # Ensure preferred sample rate is supported, otherwise pick a default
            if self.sample_rate not in self.microphone.samplerates:
                logger.warning(f"Preferred sample rate {self.sample_rate}Hz not supported by {self.microphone.name}. Using default.")
                self.sample_rate = self.microphone.samplerates[0] if self.microphone.samplerates else 48000 # A common default

        except Exception as e:
            logger.error(f"Error initializing audio device: {e}")
            logger.info("Available microphones:")
            for mic in sc.all_microphones(include_loopback=True):
                logger.info(f"  Name: {mic.name}, ID: {mic.id}, Channels: {mic.channels}, Loopback: {mic.isloopback}")
            raise

    def _capture_loop(self):
        try:
            # The `soundcard` library's record method takes numframes, not chunk_size in bytes
            # We want to process roughly chunk_size samples at a time.
            # The callback receives data with shape (frames, channels)
            logger.info(f"Starting capture with mic: {self.microphone.name}, samplerate: {self.sample_rate}, channels: {self.channels}, chunk_size: {self.chunk_size}")
            
            # The recorder context manager handles start and stop
            with self.microphone.recorder(samplerate=self.sample_rate, channels=self.channels, blocksize=self.chunk_size) as rec:
                self.recorder = rec # Store for external stop
                logger.info("Audio recording started.")
                while self._is_running:
                    data = self.recorder.record(numframes=self.chunk_size) # Record a block
                    if not self._is_running: # Check again after blocking call
                        break
                    if data.shape[0] > 0:
                        # Ensure data is float32, as Whisper expects
                        # soundcard typically returns float32 or float64, ensure it's float32
                        processed_data = data.astype(np.float32)
                        if self.channels > 1: # If stereo, take the first channel or mix down
                             processed_data = processed_data[:, 0]
                        self.audio_queue.put(processed_data.tobytes())
                    else:
                        time.sleep(0.01) # Avoid busy-waiting if no data
        except Exception as e:
            logger.error(f"Error in audio capture loop: {e}", exc_info=True)
            self.audio_queue.put(None) # Signal error to consumer
        finally:
            logger.info("Audio capture loop stopped.")
            self.recorder = None


    def start(self):
        if self._is_running:
            logger.warning("Audio capture already running.")
            return
        logger.info("Starting audio capture.")
        self._is_running = True
        self.audio_queue = queue.Queue() # Clear queue on start
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        logger.info("Stopping audio capture.")
        self._is_running = False
        if self.recorder:
            # The recorder should stop when the `with` block in _capture_loop exits.
            # Forcing a stop if it's stuck or needs immediate termination, though
            # relying on the `with` block is cleaner.
            # `soundcard` doesn't have an explicit stop on the recorder object itself
            # once started with `record()`. The loop control `_is_running` handles this.
            pass
        if self._thread and self._thread.is_alive():
            logger.info("Waiting for capture thread to join...")
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Capture thread did not join in time.")
        self._thread = None
        logger.info("Audio capture stopped.")

    def get_audio_chunk(self, timeout=1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

if __name__ == '''__main__''':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Example: List all microphones including loopback devices
    print("Available microphones:")
    all_mics = sc.all_microphones(include_loopback=True)
    if not all_mics:
        print("No microphones found.")
    for i, mic in enumerate(all_mics):
        print(f"  {i}: Name: '{mic.name}', ID: {mic.id}, Channels: {mic.channels}, Loopback: {mic.isloopback}, Sample Rates: {mic.samplerates}")

    # Example: Use default loopback or first available loopback
    # You might need to configure your system to have a loopback device active (e.g., Stereo Mix on Windows)
    # or install a virtual audio cable like BlackHole on macOS.
    
    capture_device_name = None # Set to specific name from list above if needed, e.g., "Stereo Mix" or "BlackHole"
    # For testing, find a loopback device if possible
    default_loopback = None
    for mic in all_mics:
        if mic.isloopback:
            default_loopback = mic
            break
    
    if default_loopback:
        capture_device_name = default_loopback.name
        print(f"\nAttempting to use loopback device: {capture_device_name}")
    elif all_mics:
        capture_device_name = all_mics[0].name # Fallback to first available mic if no loopback
        print(f"\nNo loopback device found. Attempting to use first available device (likely microphone): {capture_device_name}")
    else:
        print("\nNo audio devices found. Exiting.")
        exit()

    if not capture_device_name:
        print("Could not determine a device to use. Exiting example.")
        exit()

    try:
        audio_capture = AudioCapture(device_name=capture_device_name, sample_rate=16000, chunk_size=4096) # Larger chunk for testing
        audio_capture.start()
        print("Audio capture started. Recording for 5 seconds...")
        
        # Capture for a few seconds
        all_data = []
        for _ in range(5 * audio_capture.sample_rate // audio_capture.chunk_size +1): # 5 seconds of data
            chunk = audio_capture.get_audio_chunk(timeout=0.5)
            if chunk:
                print(f"Captured chunk of size: {len(chunk)} bytes")
                all_data.append(np.frombuffer(chunk, dtype=np.float32))
            elif not audio_capture._is_running:
                print("Capture stopped prematurely.")
                break
            else:
                print("No audio chunk received in timeout.")
        
        audio_capture.stop()
        print("Audio capture stopped.")

        if all_data:
            full_audio = np.concatenate(all_data)
            print(f"Total audio data captured: {full_audio.shape[0]} samples ({full_audio.shape[0]/audio_capture.sample_rate:.2f} seconds)")
            # You could save this to a WAV file for inspection if needed
            # import soundfile as sf
            # sf.write("captured_audio_test.wav", full_audio, audio_capture.sample_rate)
            # print("Saved test audio to captured_audio_test.wav")
        else:
            print("No data captured.")

    except RuntimeError as e:
        print(f"Error during audio capture test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'audio_capture' in locals() and audio_capture._is_running:
            audio_capture.stop() 