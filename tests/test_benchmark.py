import os
import sys
import time
import jiwer # For WER/CER calculation (pip install jiwer)
import numpy as np
import soundfile as sf # For reading audio files (pip install soundfile)
import yaml
import logging

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from stt import SpeechToText
# We might not need AudioCapture here if we process files directly
# from capture import AudioCapture 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# --- Configuration ---
# Path to the main config file (to get STT engine settings)
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, "config.yml")

# Path to a dedicated test/benchmark configuration (optional, could override main config)
BENCHMARK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "tests", "benchmark_config.yml")

# Directory containing test audio files (e.g., WAV format)
TEST_AUDIO_DIR = os.path.join(PROJECT_ROOT, "tests", "sample_audio")
# Directory containing reference transcripts (e.g., .txt files, names matching audio files)
REFERENCE_TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "tests", "reference_transcripts")

# --- Helper Functions ---

def load_main_config():
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.error(f"Main config file not found: {CONFIG_FILE_PATH}")
        return None
    with open(CONFIG_FILE_PATH, 'r') as f:
        return yaml.safe_load(f)

def load_benchmark_config():
    if not os.path.exists(BENCHMARK_CONFIG_PATH):
        logger.warning(f"Benchmark config file not found: {BENCHMARK_CONFIG_PATH}. Using default settings.")
        return {
            "audio_files_to_test": [], # e.g. ["audio1.wav", "audio2.wav"]
            "stt_engine_override": None # e.g. "whisper" or "azure"
        }
    with open(BENCHMARK_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def ensure_test_dirs():
    if not os.path.exists(TEST_AUDIO_DIR):
        os.makedirs(TEST_AUDIO_DIR)
        logger.info(f"Created sample audio directory: {TEST_AUDIO_DIR}")
        with open(os.path.join(TEST_AUDIO_DIR, "_placeholder.txt"), 'w') as f:
            f.write("Place your .wav test audio files here. Ensure they are 16kHz mono for best results with most STT engines.")
    
    if not os.path.exists(REFERENCE_TRANSCRIPTS_DIR):
        os.makedirs(REFERENCE_TRANSCRIPTS_DIR)
        logger.info(f"Created reference transcripts directory: {REFERENCE_TRANSCRIPTS_DIR}")
        with open(os.path.join(REFERENCE_TRANSCRIPTS_DIR, "_placeholder.txt"), 'w') as f:
            f.write("Place your .txt reference transcript files here. Name them to match audio files (e.g., audio1.txt for audio1.wav).")

def calculate_metrics(reference, hypothesis):
    """Calculates WER and CER."""
    if not reference and not hypothesis:
        return 0.0, 0.0 # Perfect match for empty strings
    if not reference and hypothesis: # All insertions
        return 1.0, 1.0 
    if reference and not hypothesis: # All deletions
        return 1.0, 1.0
        
    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    return wer, cer

def get_audio_duration(file_path):
    """Gets the duration of an audio file in seconds."""
    try:
        data, samplerate = sf.read(file_path)
        return len(data) / samplerate
    except Exception as e:
        logger.error(f"Error reading audio file duration {file_path}: {e}")
        return 0

# --- Main Test Function ---

def run_benchmark():
    main_config = load_main_config()
    if not main_config:
        return

    benchmark_config = load_benchmark_config()
    ensure_test_dirs()

    # --- Initialize STT Engine ---
    # Allow benchmark config to override STT engine from main config
    stt_override = benchmark_config.get("stt_engine_override")
    if stt_override:
        logger.info(f"Overriding STT engine to: {stt_override}")
        main_config['stt']['engine'] = stt_override
        if stt_override == "whisper" and 'whisper' not in main_config['stt']:
            main_config['stt']['whisper'] = {"model_size": "base"} # Default if not specified
        elif stt_override == "azure" and 'azure' not in main_config['stt']:
            logger.error("Azure STT override selected but no Azure config found in main config.yml")
            print("Please add your Azure key/region to config.yml under stt.azure")
            return
    
    try:
        stt = SpeechToText(main_config)
        logger.info(f"Using STT Engine: {stt.stt_engine_type}")
        if stt.stt_engine_type == 'whisper':
            logger.info(f"Whisper model: {stt.model.name if hasattr(stt.model, 'name') else main_config.get('stt',{}).get('whisper',{}).get('model_size')}")
    except Exception as e:
        logger.error(f"Failed to initialize STT engine: {e}", exc_info=True)
        if "ffmpeg" in str(e).lower():
            print("Whisper requires ffmpeg. Please install it and ensure it's in your system's PATH.")
        elif "YOUR_AZURE_SPEECH_SUBSCRIPTION_KEY" in str(e):
            print("Azure STT selected. Please update YOUR_AZURE_SPEECH_SUBSCRIPTION_KEY and region in config.yml")
        return

    # --- Process Audio Files ---
    audio_files_to_test = benchmark_config.get("audio_files_to_test", [])
    if not audio_files_to_test: # If not specified, try to find all .wav in TEST_AUDIO_DIR
        logger.info(f"No specific audio files listed in benchmark_config.yml. Looking for .wav files in {TEST_AUDIO_DIR}")
        try:
            audio_files_to_test = [f for f in os.listdir(TEST_AUDIO_DIR) if f.lower().endswith('.wav')]
        except FileNotFoundError:
            logger.error(f"Test audio directory not found: {TEST_AUDIO_DIR}")
            print(f"Please create {TEST_AUDIO_DIR} and add some .wav files.")
            return
    
    if not audio_files_to_test:
        logger.warning(f"No audio files found or specified for benchmarking in {TEST_AUDIO_DIR}.")
        print(f"Please add .wav files to {TEST_AUDIO_DIR} or list them in benchmark_config.yml.")
        print("Sample audio files (e.g., from LibriSpeech test-clean) are good for benchmarking.")
        # Create a placeholder benchmark_config.yml if it doesn't exist with an example
        if not os.path.exists(BENCHMARK_CONFIG_PATH):
            with open(BENCHMARK_CONFIG_PATH, 'w') as bc_file:
                yaml.dump({
                    "audio_files_to_test": ["sample1.wav", "sample2.wav"],
                    "stt_engine_override": None # "whisper" or "azure"
                }, bc_file, indent=2)
            logger.info(f"Created a sample benchmark_config.yml. Please edit it.")
        return

    total_wer = 0
    total_cer = 0
    total_audio_duration = 0
    total_processing_time = 0
    num_files_processed = 0

    logger.info(f"Starting benchmark for {len(audio_files_to_test)} audio file(s)...")

    for audio_file_name in audio_files_to_test:
        audio_file_path = os.path.join(TEST_AUDIO_DIR, audio_file_name)
        base_name, _ = os.path.splitext(audio_file_name)
        reference_file_path = os.path.join(REFERENCE_TRANSCRIPTS_DIR, base_name + ".txt")

        if not os.path.exists(audio_file_path):
            logger.warning(f"Audio file not found: {audio_file_path}. Skipping.")
            continue
        if not os.path.exists(reference_file_path):
            logger.warning(f"Reference transcript not found: {reference_file_path}. Skipping metrics for {audio_file_name}.")
            reference_text = None # Cannot calculate WER/CER
        else:
            with open(reference_file_path, 'r', encoding='utf-8') as f_ref:
                reference_text = f_ref.read().strip()

        logger.info(f"\nProcessing: {audio_file_name}")
        try:
            # Read audio file (Whisper expects float32 numpy array, Azure SDK handles various formats via stream)
            # For fair comparison, let's provide audio data as STT modules expect
            audio_data, sample_rate = sf.read(audio_file_path, dtype='float32')
            audio_duration = len(audio_data) / sample_rate
            total_audio_duration += audio_duration

            # Ensure mono if STT expects it (most do for simplicity, e.g. Whisper)
            # Our STT module's transcribe_chunk expects raw bytes of float32 mono audio
            # The stt.py module itself handles numpy conversion from bytes for Whisper.
            # For Azure, it converts to int16 pcm bytes for the push stream.
            
            if audio_data.ndim > 1:
                 logger.info(f"Audio {audio_file_name} is stereo, converting to mono by taking the mean of channels.")
                 audio_data = np.mean(audio_data, axis=1)
            
            # Convert to bytes as expected by our stt.py's transcribe_chunk or push_audio_to_azure
            audio_bytes = audio_data.tobytes()

            # --- Transcribe ---
            hypothesis_text = ""
            start_time = time.time()

            if stt.stt_engine_type == 'whisper':
                # Whisper processes the whole file/chunk at once
                # The transcribe_chunk in stt.py is designed for this.
                hypothesis_text = stt.transcribe_chunk(audio_bytes, sample_rate, channels=1) # Assuming mono
            
            elif stt.stt_engine_type == 'azure':
                # For Azure, simulate streaming for a file-based test (or use recognize_once for simplicity if preferred)
                # Using continuous recognition for benchmarking consistency with real-time use.
                azure_audio_format = speechsdk.audio.AudioStreamFormat(
                    samples_per_second=sample_rate,
                    bits_per_sample=16, # What stt.py converts to
                    channels=1 # Assuming mono
                )
                push_stream = stt.start_azure_continuous_recognition(azure_audio_format)
                if not push_stream:
                    raise RuntimeError("Failed to start Azure continuous recognition for benchmark.")

                # Push audio in chunks (e.g., 100ms chunks)
                # This simulates how main.py would stream live audio
                chunk_size_samples = int(sample_rate * 0.1) # 100ms chunks
                offset = 0
                while offset < len(audio_data):
                    chunk_np_float32 = audio_data[offset : offset + chunk_size_samples]
                    stt.push_audio_to_azure(chunk_np_float32.tobytes()) # stt.py handles conversion to int16
                    offset += len(chunk_np_float32)
                    time.sleep(0.01) # Simulate small delay, not strictly necessary for file test
                
                # Azure SDK needs time for final recognition after stream ends
                # The stop call will fetch all recognized segments.
                recognized_segments = stt.stop_azure_continuous_recognition()
                hypothesis_text = " ".join(recognized_segments).strip()
            else:
                logger.error(f"Benchmarking not implemented for STT engine: {stt.stt_engine_type}")
                continue

            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            num_files_processed += 1

            logger.info(f"  Hypothesis: \"{hypothesis_text}\"")
            if reference_text is not None:
                logger.info(f"  Reference : \"{reference_text}\"")
                wer, cer = calculate_metrics(reference_text, hypothesis_text)
                total_wer += wer
                total_cer += cer
                logger.info(f"  Metrics for {audio_file_name}: WER = {wer:.4f}, CER = {cer:.4f}")
            else:
                logger.info(f"  Metrics for {audio_file_name}: Not calculated (no reference text).")
            
            logger.info(f"  Audio duration: {audio_duration:.2f}s, Processing time: {processing_time:.2f}s")
            if audio_duration > 0:
                rtf = processing_time / audio_duration
                logger.info(f"  Real-Time Factor (RTF): {rtf:.3f}")
            else:
                logger.info("  Real-Time Factor (RTF): N/A (audio duration is zero)")

        except Exception as e:
            logger.error(f"Error processing file {audio_file_name}: {e}", exc_info=True)
    
    # --- Aggregate Results ---
    if num_files_processed > 0:
        avg_wer = total_wer / num_files_processed if reference_text is not None else -1 # Only if all files had refs
        avg_cer = total_cer / num_files_processed if reference_text is not None else -1
        avg_processing_time = total_processing_time / num_files_processed
        
        print("\n--- Benchmark Summary ---")
        print(f"STT Engine: {stt.stt_engine_type}")
        if stt.stt_engine_type == 'whisper':
             print(f"Whisper Model: {stt.model.name if hasattr(stt.model, 'name') else main_config.get('stt',{}).get('whisper',{}).get('model_size')}")
        print(f"Files processed: {num_files_processed}")
        print(f"Total audio duration: {total_audio_duration:.2f}s")
        print(f"Total processing time: {total_processing_time:.2f}s")
        if total_audio_duration > 0:
            overall_rtf = total_processing_time / total_audio_duration
            print(f"Overall Real-Time Factor (RTF): {overall_rtf:.3f}")
        else:
            print("Overall Real-Time Factor (RTF): N/A")

        if avg_wer != -1:
            print(f"Average Word Error Rate (WER): {avg_wer:.4f}")
            print(f"Average Character Error Rate (CER): {avg_cer:.4f}")
        else:
            print("Average WER/CER not calculated as some reference transcripts were missing.")
        print("-------------------------")
        
        # Sample results section for the README/output
        print("\nSample Results (to be included in documentation):")
        print(f"- Engine: {stt.stt_engine_type}")
        if stt.stt_engine_type == 'whisper': print(f"- Whisper Model: {stt.model.name if hasattr(stt.model, 'name') else main_config.get('stt',{}).get('whisper',{}).get('model_size')}")
        print(f"- Average WER: {avg_wer:.4f if avg_wer != -1 else 'N/A'}")
        print(f"- Average CER: {avg_cer:.4f if avg_cer != -1 else 'N/A'}")
        print(f"- Overall RTF: {overall_rtf:.3f if total_audio_duration > 0 else 'N/A'}")
        print("- Note: These results depend heavily on the test dataset and system hardware.")

    else:
        print("No files were processed. Benchmark did not run.")

if __name__ == "__main__":
    # Create dummy files and configs for a first-time run if they don't exist.
    ensure_test_dirs()
    if not os.path.exists(BENCHMARK_CONFIG_PATH):
        with open(BENCHMARK_CONFIG_PATH, 'w') as bc_file:
            yaml.dump({
                "audio_files_to_test": ["example_audio.wav"], # User should replace this
                "stt_engine_override": None 
            }, bc_file, indent=2)
        logger.info(f"Created a sample {BENCHMARK_CONFIG_PATH}. Please edit it with actual audio file names.")

    # Create a dummy audio file and transcript for demonstration if none exist
    if not any(f.lower().endswith('.wav') for f in os.listdir(TEST_AUDIO_DIR)):
        sample_rate = 16000
        duration = 5 # seconds
        frequency = 440
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # A simple tone, not actual speech, just for placeholder
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t) 
        # Add a bit of noise to make it slightly more interesting than pure silence for STT
        noise = np.random.normal(0, 0.02, audio_data.shape)
        audio_data += noise
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        dummy_audio_path = os.path.join(TEST_AUDIO_DIR, "example_audio.wav")
        sf.write(dummy_audio_path, audio_data, sample_rate, subtype='PCM_16') # PCM_16 for wider compatibility
        logger.info(f"Created a dummy audio file: {dummy_audio_path}")
        
        dummy_transcript_path = os.path.join(REFERENCE_TRANSCRIPTS_DIR, "example_audio.txt")
        with open(dummy_transcript_path, 'w') as f_ref:
            # Whisper on a sine wave + noise might produce nothing or random words.
            f_ref.write("This is a dummy reference transcript for the example audio.") 
        logger.info(f"Created a dummy reference transcript: {dummy_transcript_path}")
        print("\nPlease replace 'example_audio.wav' and 'example_audio.txt' with your actual test data.")

    run_benchmark() 