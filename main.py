import logging
import yaml
import time
import threading
import queue
import os
from datetime import datetime
from dotenv import load_dotenv
import re

from capture import AudioCapture
from stt import SpeechToText
from ui import TranscriptionUI
from analysis import extract_keywords, generate_extractive_summary, ensure_nltk_data

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # To console
        # logging.FileHandler("transcription_app.log") # Optionally log to a file
    ]
)
logger = logging.getLogger(__name__)

# --- Global Application State ---
APP_RUNNING = True
TRANSCRIPTION_QUEUE = queue.Queue() # For STT results to UI/logger
AUDIO_BUFFER_SECONDS = 5 # How much audio to buffer before sending to STT (Whisper works better with longer chunks)
FULL_TRANSCRIPT_LIST = [] # To store all transcript segments for final analysis

def load_config(config_path="config.yml"):
    """Loads the YAML configuration file and replaces ${VAR} with environment variables."""
    load_dotenv(override=True)  # Load .env file if present
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        # Recursively replace ${VAR} with env values
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(i) for i in obj]
            elif isinstance(obj, str):
                # Replace ${VAR} with os.environ[VAR] if present
                def repl(match):
                    var = match.group(1)
                    return os.environ.get(var, match.group(0))
                return re.sub(r'\$\{([^}]+)\}', repl, obj)
            else:
                return obj
        config = replace_env_vars(config)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found. Please create it.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        return None

def transcription_file_writer(log_config, ui_queue_for_display):
    """Handles writing transcriptions to a log file and can also send to UI."""
    global APP_RUNNING, TRANSCRIPTION_QUEUE, FULL_TRANSCRIPT_LIST
    
    log_dir = log_config.get("log_directory", "transcripts")
    log_prefix = log_config.get("log_file_prefix", "transcription_log_")
    
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            logger.info(f"Created log directory: {log_dir}")
        except OSError as e:
            logger.error(f"Could not create log directory {log_dir}: {e}")
            # Fallback to current directory if creation fails
            log_dir = "."

    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"{log_prefix}{current_date}.txt")
    logger.info(f"Transcription log file: {log_file_path}")

    # Clear full transcript list at the start of a new session log
    FULL_TRANSCRIPT_LIST.clear()

    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"--- Transcription Log Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
            while APP_RUNNING or not TRANSCRIPTION_QUEUE.empty():
                try:
                    transcript_segment = TRANSCRIPTION_QUEUE.get(timeout=0.5)
                    if transcript_segment is None: # Sentinel to stop
                        if APP_RUNNING: # Only break if APP_RUNNING is also false
                            continue
                        else:
                            break
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    log_entry = f"[{timestamp}] {transcript_segment}\n"
                    
                    log_file.write(log_entry)
                    log_file.flush()
                    
                    # Also send to UI for display (if UI is used directly by this thread)
                    if ui_queue_for_display: # Check if a UI queue is provided
                         ui_queue_for_display.update_transcription(transcript_segment, append=True)
                    FULL_TRANSCRIPT_LIST.append(transcript_segment) # Accumulate for final analysis

                except queue.Empty:
                    if not APP_RUNNING and TRANSCRIPTION_QUEUE.empty():
                        break
                    continue # No new transcript, continue loop
                except Exception as e:
                    logger.error(f"Error in transcription writer: {e}", exc_info=True)
                    time.sleep(0.1) # Avoid busy loop on error
            
            # Return the path for potential use in appending summary/keywords
            return log_file_path 
            
    except IOError as e:
        logger.error(f"Error opening or writing to log file {log_file_path}: {e}")
        return None # Indicate failure
    finally:
        logger.info("Transcription file writer stopped.")


def stop_application_signal():
    """Callback for UI to signal application stop."""
    global APP_RUNNING
    logger.info("Stop signal received. Initiating application shutdown...")
    APP_RUNNING = False
    TRANSCRIPTION_QUEUE.put(None) # Sentinel for writer thread

def main():
    global APP_RUNNING, TRANSCRIPTION_QUEUE, AUDIO_BUFFER_SECONDS, FULL_TRANSCRIPT_LIST

    config = load_config()
    if not config:
        return

    # Ensure NLTK data is available/downloaded before STT starts (or at least before analysis)
    # This can take a moment on first run.
    logger.info("Checking/Ensuring NLTK data (punkt, stopwords)...")
    ensure_nltk_data()
    logger.info("NLTK data check complete.")

    audio_config = config.get('audio', {})
    stt_config = config.get('stt', {})
    ui_config = config.get('ui', {})
    log_config = config.get('logging', {})
    vad_config = config.get('vad', {}) # VAD config might be used by STT or main loop

    # Initialize components
    try:
        audio_capture = AudioCapture(
            device_name=audio_config.get('device_name', 'default'),
            sample_rate=audio_config.get('sample_rate', 16000),
            channels=audio_config.get('channels', 1),
            chunk_size=audio_config.get('chunk_size', 1024)
        )
        stt_engine = SpeechToText(config) # Pass full config as STT might need various parts
        transcription_ui = TranscriptionUI(ui_config, stop_callback=stop_application_signal)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        return

    # Start UI and transcription writer thread
    transcription_ui.start()
    
    # The UI object has its own update_queue, so we pass that to the writer
    # Or the writer uses the global TRANSCRIPTION_QUEUE and UI polls that (less direct)
    # For now, let main loop put into TRANSCRIPTION_QUEUE, and writer reads it and updates UI.
    writer_thread_context = {"log_file_path": None}
    def writer_thread_target_wrapper(log_cfg, ui_obj, context):
        context["log_file_path"] = transcription_file_writer(log_cfg, ui_obj)
    
    writer_thread = threading.Thread(
        target=writer_thread_target_wrapper, 
        args=(log_config, transcription_ui, writer_thread_context),
        daemon=True,
        name="FileWriterThread"
    )
    writer_thread.start()
    
    transcription_ui.set_status_message("Initializing...")

    # Start audio capture
    try:
        # Use actual sample rate from initialized audio_capture for STT processing
        actual_sample_rate = audio_capture.sample_rate
        actual_channels = audio_capture.channels
        
        audio_capture.start()
        logger.info(f"Audio capture started. Device: {audio_capture.microphone.name if audio_capture.microphone else 'N/A'}, Rate: {actual_sample_rate}, Channels: {actual_channels}")
        transcription_ui.set_status_message(f"Listening on: {audio_capture.microphone.name if audio_capture.microphone else 'Unknown'}...")
    except Exception as e:
        logger.error(f"Failed to start audio capture: {e}", exc_info=True)
        transcription_ui.set_status_message(f"ERROR: Failed to start audio: {e}")
        APP_RUNNING = False # Ensure app stops if audio fails critically

    # --- Main Application Loop ---
    audio_buffer = bytearray()
    bytes_per_second = actual_sample_rate * actual_channels * 2 # Assuming 16-bit, though capture.py gives float32 (4 bytes)
                                                                # capture.py puts float32 bytes, so 4 bytes per sample.
    bytes_per_second_float32 = actual_sample_rate * actual_channels * 4


    # For Azure continuous STT
    azure_push_stream = None
    if stt_engine.stt_engine_type == 'azure':
        try:
            # Azure SDK expects specific format, often S16LE for push stream.
            # capture.py provides float32. stt.py (azure part) converts float32 to int16.
            # The format details here are for Azure to know what to expect *after* our conversion in stt.py
            azure_audio_format = SpeechToText.speechsdk.audio.AudioStreamFormat( # Access via class if instance not fully ready or for static-like use
                samples_per_second=actual_sample_rate,
                bits_per_sample=16, # This is what stt.py will convert to for Azure
                channels=actual_channels
            )
            azure_push_stream = stt_engine.start_azure_continuous_recognition(azure_audio_format)
            if not azure_push_stream:
                raise RuntimeError("Failed to start Azure continuous recognition stream.")
            logger.info("Azure continuous recognition started and push stream is ready.")
            transcription_ui.set_status_message("Azure STT ready (continuous mode).")
        except Exception as e:
            logger.error(f"Failed to initialize Azure continuous STT: {e}", exc_info=True)
            transcription_ui.set_status_message(f"ERROR: Azure STT failed: {e}")
            APP_RUNNING = False # Critical failure

    try:
        while APP_RUNNING:
            chunk = audio_capture.get_audio_chunk(timeout=0.1) # Short timeout for responsiveness
            if chunk:
                if stt_engine.stt_engine_type == 'azure' and azure_push_stream:
                    stt_engine.push_audio_to_azure(chunk) # Already float32 bytes from capture
                    # For Azure continuous, transcriptions come via callbacks to _azure_transcriptions
                    # We need to periodically check stt_engine._azure_transcriptions and send to our TRANSCRIPTION_QUEUE
                    new_azure_transcripts = []
                    if stt_engine._azure_transcriptions: # Check if list has content
                        # Drain the list; careful with thread safety if STT callbacks modify it directly
                        # A lock in stt.py around _azure_transcriptions access would be safer
                        # For now, assume callbacks append and we pop.
                        while stt_engine._azure_transcriptions:
                            try:
                                new_azure_transcripts.append(stt_engine._azure_transcriptions.pop(0))
                            except IndexError: # list became empty
                                break 
                    for transcript in new_azure_transcripts:
                        if transcript: # Ensure not empty
                            TRANSCRIPTION_QUEUE.put(transcript)

                elif stt_engine.stt_engine_type == 'whisper':
                    audio_buffer.extend(chunk)
                    # Check if buffer has enough audio
                    # Whisper expects float32 audio. capture.py provides this as bytes.
                    # Size of float32 is 4 bytes.
                    # Target buffer size in bytes:
                    buffer_target_bytes = int(AUDIO_BUFFER_SECONDS * bytes_per_second_float32)

                    if len(audio_buffer) >= buffer_target_bytes:
                        logger.debug(f"Processing audio buffer of size: {len(audio_buffer)} bytes for Whisper.")
                        
                        # Prepare audio data for STT
                        # audio_data_for_stt = np.frombuffer(audio_buffer, dtype=np.float32)
                        # For Whisper, pass bytes directly. stt.py will handle np.frombuffer
                        transcript = stt_engine.transcribe_chunk(bytes(audio_buffer), actual_sample_rate, actual_channels)
                        audio_buffer.clear() # Clear buffer after processing

                        if transcript:
                            logger.info(f"Whisper Transcription: {transcript}")
                            TRANSCRIPTION_QUEUE.put(transcript)
            
            elif not audio_capture._is_running and APP_RUNNING:
                logger.warning("Audio capture seems to have stopped unexpectedly.")
                transcription_ui.set_status_message("WARNING: Audio capture stopped!")
                # Decide if this is a critical failure
                # APP_RUNNING = False # Option: stop app if audio capture dies
                time.sleep(0.5) # Avoid busy loop if it keeps happening

            if not transcription_ui.is_active() and APP_RUNNING:
                logger.info("UI window has been closed. Initiating shutdown.")
                APP_RUNNING = False # If UI closes, app should stop
                TRANSCRIPTION_QUEUE.put(None) # Ensure writer thread also knows

            time.sleep(0.01) # Small delay to prevent busy-looping if no audio

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        APP_RUNNING = False
    except Exception as e:
        logger.error(f"Unhandled error in main loop: {e}", exc_info=True)
        APP_RUNNING = False
    finally:
        logger.info("Cleaning up resources...")
        APP_RUNNING = False # Ensure it's set for all threads

        if 'audio_capture' in locals() and audio_capture._is_running:
            logger.info("Stopping audio capture...")
            audio_capture.stop()

        if stt_engine and stt_engine.stt_engine_type == 'azure' and stt_engine._azure_recognizing:
            logger.info("Stopping Azure continuous recognition...")
            final_azure_texts = stt_engine.stop_azure_continuous_recognition()
            for text in final_azure_texts: # Process any final transcripts
                if text: TRANSCRIPTION_QUEUE.put(text)
        
        TRANSCRIPTION_QUEUE.put(None) # Send sentinel to writer thread
        
        if 'writer_thread' in locals() and writer_thread.is_alive():
            logger.info("Waiting for transcription writer thread to finish...")
            writer_thread.join(timeout=2)

        if 'transcription_ui' in locals() and transcription_ui.is_active():
            logger.info("Stopping UI...")
            transcription_ui.stop()
        
        logger.info("Application shutdown complete.")

        # --- Perform final analysis and log --- 
        final_transcript_text = " ".join(FULL_TRANSCRIPT_LIST)
        log_file_to_append = writer_thread_context.get("log_file_path")

        if final_transcript_text.strip():
            logger.info("\n--- Full Transcript for Analysis ---")
            # logger.info(final_transcript_text) # Can be very long
            logger.info(f"Total characters in transcript: {len(final_transcript_text)}")
            
            transcription_ui.set_status_message("Analyzing transcript...")
            summary = "Error generating summary."
            keywords = []
            try:
                summary = generate_extractive_summary(final_transcript_text)
                keywords = extract_keywords(final_transcript_text)
            except Exception as analysis_exc:
                logger.error(f"Error during text analysis: {analysis_exc}", exc_info=True)

            analysis_output = f"\n\n--- Transcript Analysis ---\n"
            analysis_output += f"Summary:\n{summary}\n\n"
            analysis_output += f"Keywords:\n{(', '.join(keywords) if keywords else 'None extracted')}\n"
            analysis_output += "--- End of Analysis ---\n"

            print(analysis_output) # Print to console
            transcription_ui.set_status_message("Analysis complete. See log for details.")

            if log_file_to_append and os.path.exists(log_file_to_append):
                try:
                    with open(log_file_to_append, 'a', encoding='utf-8') as f_log:
                        f_log.write(analysis_output)
                    logger.info(f"Appended analysis to log file: {log_file_to_append}")
                except IOError as e:
                    logger.error(f"Error appending analysis to log file {log_file_to_append}: {e}")
            elif log_file_to_append:
                logger.warning(f"Log file {log_file_to_append} not found for appending analysis.")
            else:
                logger.warning("No log file path was available to append analysis.")
        else:
            logger.info("No transcript content to analyze.")
            transcription_ui.set_status_message("No transcript to analyze.")

if __name__ == "__main__":
    main() 