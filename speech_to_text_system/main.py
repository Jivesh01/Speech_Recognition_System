import speech_recognition as sr
import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            print("Listening...")
            audio = recognizer.record(source)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Request error from Google API: {e}")
        return None

# Test the function with the file path
audio_path = "sample.wav"
transcribe_audio(audio_path)

def transcribe_with_wav2vec(audio_path):
    try:
        # Use Wav2Vec2Processor instead of Wav2Vec2Tokenizer
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    except Exception as e:
        print(f"Error loading Wav2Vec2 model or processor: {e}")
        return None

    try:
        # Attempt to load the audio file using torchaudio
        try:
            waveform, rate = torchaudio.load(audio_path)
        except Exception:
            # Fallback to soundfile if torchaudio fails
            print("Falling back to soundfile for loading audio...")
            waveform, rate = sf.read(audio_path)
            waveform = torch.tensor(waveform).unsqueeze(0)  # Add batch dimension

        if rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
            waveform = resampler(waveform)
    except FileNotFoundError:
        print(f"Error: File not found at {audio_path}")
        return None
    except Exception as e:
        print(f"Error loading or resampling audio file: {e}")
        return None

    try:
        # Process the waveform and perform transcription
        input_values = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        print("Transcription:", transcription)
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# Test the Wav2Vec2 function with the file path
transcribe_with_wav2vec(audio_path)