import logging
import numpy as np
import threading
import io

# Conditional imports for STT engines
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None # So references don't break

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    speechsdk = None # So references don't break

logger = logging.getLogger(__name__)

class SpeechToText:
    def __init__(self, config):
        self.config = config
        self.stt_engine_type = config.get('stt', {}).get('engine', 'whisper').lower()
        self.language_hint = config.get('stt', {}).get('language_hint', 'en-US')
        self.model = None
        self.speech_config = None
        self.audio_config = None # For Azure continuous recognition
        self.speech_recognizer = None # For Azure
        self._azure_recognition_thread = None
        self._azure_recognizing = False
        self._azure_transcriptions = [] # Store results from Azure callbacks

        if self.stt_engine_type == 'whisper':
            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper library is not installed. Please install it via pip install openai-whisper.")
            self._init_whisper()
        elif self.stt_engine_type == 'azure':
            if not AZURE_AVAILABLE:
                raise ImportError("Azure Speech SDK is not installed. Please install it via pip install azure-cognitiveservices-speech.")
            self._init_azure()
        else:
            raise ValueError(f"Unsupported STT engine: {self.stt_engine_type}")

    def _init_whisper(self):
        whisper_config = self.config.get('stt', {}).get('whisper', {})
        model_size = whisper_config.get('model_size', 'base')
        # device = whisper_config.get('device') # Whisper auto-detects device (cpu/cuda)
        logger.info(f"Initializing Whisper STT engine with model size: {model_size}")
        try:
            # self.model = whisper.load_model(model_size, device=device) # Specify device if needed
            self.model = whisper.load_model(model_size)
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_size}': {e}")
            logger.error("Please ensure the model name is correct and any necessary dependencies (like ffmpeg) are installed.")
            if "ffmpeg" in str(e).lower():
                logger.error("Whisper requires ffmpeg. Please install it and ensure it's in your system's PATH.")
            raise
        logger.info("Whisper STT engine initialized.")

    def _init_azure(self):
        azure_config = self.config.get('stt', {}).get('azure', {})
        subscription_key = azure_config.get('subscription_key')
        region = azure_config.get('region')

        if not subscription_key or not region or subscription_key == "YOUR_AZURE_SPEECH_SUBSCRIPTION_KEY":
            raise ValueError("Azure subscription key and region must be configured in config.yml")

        logger.info(f"Initializing Azure STT engine in region: {region}")
        self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
        self.speech_config.speech_recognition_language = self.language_hint
        # self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_PostProcessingOption, "TrueText") # Optional: for better formatting

        # Using PushAudioInputStream for feeding chunks manually
        # However, for real-time, continuous recognition with SDK's internal VAD is better
        # Let's set up for continuous recognition. The main loop will push audio.
        # self.push_stream = speechsdk.audio.PushAudioInputStream()
        # self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

        logger.info("Azure STT engine initialized (config done, recognizer created on first transcribe call or start_continuous).")


    def transcribe_chunk_whisper(self, audio_chunk_bytes, sample_rate, channels):
        """
        Transcribes a single audio chunk using Whisper.
        audio_chunk_bytes: raw audio data as bytes
        sample_rate: sample rate of the audio
        channels: number of audio channels
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded.")

        # Convert byte data to NumPy array
        # Whisper expects float32 mono audio
        try:
            audio_np = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
            # If stereo, it should have been converted to mono in capture.py
            # but as a fallback, or if capture.py changes:
            if channels > 1:
                # This assumes interleaved stereo. This should ideally be handled upstream.
                # For now, let's assume capture.py already made it mono float32.
                # If it was stereo, it would be (samples, channels), then flattened.
                # audio_np = audio_np.reshape(-1, channels)[:, 0] # Take first channel
                pass # Assuming mono float32 from capture.py

        except Exception as e:
            logger.error(f"Error processing audio chunk for Whisper: {e}")
            return ""

        if audio_np.size == 0:
            return ""

        # Whisper options
        options = whisper.DecodingOptions(
            language=self.language_hint.split('-')[0] if self.language_hint else None, # Whisper expects language code like 'en', 'es'
            fp16=False # Set to True if using CUDA and GPU supports it, for faster processing
        )
        
        try:
            result = self.model.transcribe(audio_np, **options.__dict__)
            # For chunk-by-chunk, we might want to manage segments and context better
            # This is a basic implementation.
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            return ""

    def transcribe_chunk_azure(self, audio_chunk_bytes, sample_rate, channels):
        """
        Transcribes a single audio chunk using Azure. This is for one-off recognition.
        For continuous, use start_continuous_recognition and push audio.
        """
        if not self.speech_config:
            raise RuntimeError("Azure SpeechConfig not initialized.")

        # Create a PushAudioInputStream for this single chunk
        push_stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=sample_rate,
                bits_per_sample=16, # Azure SDK typically works well with 16-bit PCM
                channels=channels
            )
        )
        audio_config_temp = speechsdk.audio.AudioConfig(stream=push_stream)
        
        # Convert float32 to int16 if necessary, as Azure SDK often expects PCM S16LE
        audio_np_float32 = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
        if audio_np_float32.size == 0:
            return ""
        
        audio_np_int16 = (audio_np_float32 * 32767).astype(np.int16)
        
        push_stream.write(audio_np_int16.tobytes())
        push_stream.close() # Signal end of stream for this chunk

        recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config_temp)
        
        result = recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip()
        elif result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug("Azure: No speech could be recognized from the audio.")
            return ""
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            logger.error(f"Azure: Speech Recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Azure: Error details: {cancellation_details.error_details}")
            return ""
        return ""

    def transcribe_chunk(self, audio_chunk_bytes, sample_rate, channels):
        """
        Main method to transcribe an audio chunk.
        Routes to the appropriate engine.
        For Azure, this implies a single recognition. For continuous, use specific Azure methods.
        """
        if not audio_chunk_bytes:
            return ""
            
        if self.stt_engine_type == 'whisper':
            return self.transcribe_chunk_whisper(audio_chunk_bytes, sample_rate, channels)
        elif self.stt_engine_type == 'azure':
            # This one-shot method for Azure might be slow for real-time.
            # Consider implementing continuous recognition if Azure is primary.
            return self.transcribe_chunk_azure(audio_chunk_bytes, sample_rate, channels)
        else:
            logger.error(f"STT engine {self.stt_engine_type} not supported for chunk transcription here.")
            return ""

    # --- Azure Continuous Recognition Methods ---
    def _azure_continuous_recognizing_cb(self, evt):
        logger.debug(f'AZURE RECOGNIZING: {evt.result.text}')
        # You can use this for intermediate results if needed by UI
        # For this app, we'll mostly rely on Recognized event for final segments.

    def _azure_continuous_recognized_cb(self, evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            logger.info(f'AZURE RECOGNIZED: "{evt.result.text}"')
            self._azure_transcriptions.append(evt.result.text)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug('AZURE NOMATCH: Speech could not be recognized.')
        elif evt.result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = evt.result.cancellation_details
            logger.warning(f"AZURE CANCELED: Reason={cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                logger.error(f"AZURE CANCELED: ErrorDetails={cancellation_details.error_details}")
            self._azure_recognizing = False # Stop if error


    def _azure_continuous_session_stopped_cb(self, evt):
        logger.info(f'AZURE SESSION STOPPED: {evt}')
        self._azure_recognizing = False

    def _azure_continuous_canceled_cb(self, evt):
        logger.warning(f'AZURE RECOGNITION CANCELED: {evt.cancellation_details.reason}')
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            logger.error(f'AZURE Error details: {evt.cancellation_details.error_details}')
        self._azure_recognizing = False


    def start_azure_continuous_recognition(self, audio_stream_format_details):
        if self.stt_engine_type != 'azure':
            logger.error("Azure continuous recognition only available for Azure engine.")
            return None

        if not self.speech_config:
            self._init_azure() # Ensure it's initialized

        # Setup PushAudioInputStream
        self.push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_stream_format_details)
        self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)

        # Connect callbacks
        self.speech_recognizer.recognizing.connect(self._azure_continuous_recognizing_cb)
        self.speech_recognizer.recognized.connect(self._azure_continuous_recognized_cb)
        self.speech_recognizer.session_stopped.connect(self._azure_continuous_session_stopped_cb)
        self.speech_recognizer.canceled.connect(self._azure_continuous_canceled_cb)

        self._azure_transcriptions = [] # Clear previous results
        self._azure_recognizing = True
        
        # Start continuous recognition. This will run in a background thread managed by the SDK.
        # The call to start_continuous_recognition_async() is non-blocking.
        self.speech_recognizer.start_continuous_recognition_async()
        logger.info("Azure continuous recognition started.")
        return self.push_stream # Return the stream so audio data can be pushed

    def push_audio_to_azure(self, audio_chunk_bytes):
        if self.stt_engine_type != 'azure' or not self._azure_recognizing or not self.push_stream:
            # logger.warning("Cannot push audio: Azure not in continuous recognition mode or stream not ready.")
            return

        # Azure SDK expects PCM S16LE for PushAudioInputStream typically
        # Convert float32 from capture to int16
        try:
            audio_np_float32 = np.frombuffer(audio_chunk_bytes, dtype=np.float32)
            if audio_np_float32.size == 0:
                return
            audio_np_int16 = (audio_np_float32 * 32767).astype(np.int16)
            self.push_stream.write(audio_np_int16.tobytes())
        except Exception as e:
            logger.error(f"Error pushing audio to Azure: {e}")


    def stop_azure_continuous_recognition(self):
        if self.stt_engine_type != 'azure' or not self._azure_recognizing:
            return []

        logger.info("Stopping Azure continuous recognition...")
        self._azure_recognizing = False # Signal to stop
        
        if self.speech_recognizer:
            self.speech_recognizer.stop_continuous_recognition_async().get() # Wait for it to actually stop
            logger.info("Azure continuous recognition stopped.")

        if self.push_stream:
            self.push_stream.close() # Close the audio stream
            self.push_stream = None

        # Deregister callbacks (important to avoid issues if re-init)
        if self.speech_recognizer:
            self.speech_recognizer.recognizing.disconnect_all()
            self.speech_recognizer.recognized.disconnect_all()
            self.speech_recognizer.session_stopped.disconnect_all()
            self.speech_recognizer.canceled.disconnect_all()
            self.speech_recognizer = None

        return self._azure_transcriptions


if __name__ == '''__main__''':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Create a dummy config for testing
    dummy_config_whisper = {
        "stt": {
            "engine": "whisper",
            "language_hint": "en", # Whisper uses short codes
            "whisper": {
                "model_size": "tiny" # Use tiny for quick testing
            }
        },
        "audio": { # Needed for sample rate, channels during transcription call
            "sample_rate": 16000,
            "channels": 1
        }
    }

    # Azure test requires valid credentials
    # dummy_config_azure = {
    #     "stt": {
    #         "engine": "azure",
    #         "language_hint": "en-US",
    #         "azure": {
    #             "subscription_key": "YOUR_KEY_HERE", 
    #             "region": "YOUR_REGION_HERE"
    #         }
    #     },
    #      "audio": {
    #         "sample_rate": 16000,
    #         "channels": 1
    #     }
    # }

    print("--- Testing Whisper STT ---")
    if WHISPER_AVAILABLE:
        try:
            stt_whisper = SpeechToText(dummy_config_whisper)
            # Create a dummy audio chunk (e.g., 1 second of silence or simple tone)
            sample_rate = dummy_config_whisper['audio']['sample_rate']
            channels = dummy_config_whisper['audio']['channels']
            duration_sec = 1
            # Create a simple sine wave for testing; replace with actual audio for real test
            frequency = 440  # A4 note
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
            # audio_data_np = 0.5 * np.sin(2 * np.pi * frequency * t) # Keep amplitude low
            # For transcription, silence is fine to test plumbing
            audio_data_np = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)

            # Simulate spoken words (very crudely) by adding some noise bursts
            # This is just for a basic test, Whisper is robust enough for real audio
            burst_len = int(sample_rate * 0.1)
            audio_data_np[int(sample_rate*0.1) : int(sample_rate*0.1) + burst_len] = (np.random.rand(burst_len) - 0.5) * 0.1
            audio_data_np[int(sample_rate*0.5) : int(sample_rate*0.5) + burst_len] = (np.random.rand(burst_len) - 0.5) * 0.1
            
            audio_bytes = audio_data_np.astype(np.float32).tobytes()

            print(f"Transcribing {duration_sec}s dummy audio chunk with Whisper...")
            transcription = stt_whisper.transcribe_chunk(audio_bytes, sample_rate, channels)
            print(f"Whisper transcription: '{transcription}'") # Expected: empty or gibberish for silence/noise
        except Exception as e:
            print(f"Error testing Whisper: {e}")
    else:
        print("Whisper not available, skipping Whisper STT test.")

    # print("\n--- Testing Azure STT (Chunk) ---")
    # if AZURE_AVAILABLE:
    #     if dummy_config_azure["stt"]["azure"]["subscription_key"] == "YOUR_KEY_HERE":
    #         print("Azure credentials not set in test code. Skipping Azure chunk test.")
    #     else:
    #         try:
    #             stt_azure_chunk = SpeechToText(dummy_config_azure)
    #             # Use same dummy audio as Whisper
    #             sample_rate = dummy_config_azure['audio']['sample_rate']
    #             channels = dummy_config_azure['audio']['channels']
    #             audio_data_np = np.zeros(int(sample_rate * 1), dtype=np.float32) # 1 sec silence
    #             audio_bytes = audio_data_np.astype(np.float32).tobytes()

    #             print("Transcribing 1s dummy audio chunk with Azure (one-shot)...")
    #             transcription_azure_chunk = stt_azure_chunk.transcribe_chunk(audio_bytes, sample_rate, channels)
    #             print(f"Azure (chunk) transcription: '{transcription_azure_chunk}'")
    #         except Exception as e:
    #             print(f"Error testing Azure (chunk): {e}")
    # else:
    #     print("Azure SDK not available, skipping Azure STT chunk test.")


    # print("\n--- Testing Azure STT (Continuous) ---")
    # if AZURE_AVAILABLE:
    #     if dummy_config_azure["stt"]["azure"]["subscription_key"] == "YOUR_KEY_HERE":
    #         print("Azure credentials not set in test code. Skipping Azure continuous test.")
    #     else:
    #         try:
    #             stt_azure_cont = SpeechToText(dummy_config_azure)
    #             sample_rate = dummy_config_azure['audio']['sample_rate']
    #             channels = dummy_config_azure['audio']['channels']
                
    #             # Azure audio stream format
    #             # For push stream, Azure SDK typically expects 16-bit PCM samples.
    #             azure_audio_format = speechsdk.audio.AudioStreamFormat(
    #                 samples_per_second=sample_rate,
    #                 bits_per_sample=16, 
    #                 channels=channels
    #             )

    #             push_stream = stt_azure_cont.start_azure_continuous_recognition(azure_audio_format)
    #             if push_stream:
    #                 print("Azure continuous recognition started. Pushing 3 seconds of dummy audio in chunks...")
                    
    #                 # Push a few chunks of audio
    #                 for i in range(3): # 3 chunks, ~1 second each if chunk_size = sample_rate
    #                     # Create some audio data (float32 initially)
    #                     audio_data_np_float32 = np.random.randn(sample_rate).astype(np.float32) * 0.1 # 1 sec of noise
                        
    #                     # Convert to bytes (as float32, stt.py will handle conversion to int16 for Azure)
    #                     audio_chunk_bytes = audio_data_np_float32.tobytes()
                        
    #                     stt_azure_cont.push_audio_to_azure(audio_chunk_bytes)
    #                     print(f"Pushed chunk {i+1} to Azure.")
    #                     time.sleep(1) # Simulate real-time chunk arrival

    #                 final_transcriptions = stt_azure_cont.stop_azure_continuous_recognition()
    #                 print("Azure continuous recognition stopped.")
    #                 print("Final transcriptions from Azure:")
    #                 for tscript in final_transcriptions:
    #                     print(f" - "{tscript}"")
    #             else:
    #                 print("Failed to start Azure continuous recognition.")
                    
    #         except Exception as e:
    #             print(f"Error testing Azure continuous: {e}")
    # else:
    #     print("Azure SDK not available, skipping Azure STT continuous test.") 