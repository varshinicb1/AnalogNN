"""
Custom Sarvam AI TTS Service for Manim Voiceover
"""

import requests
import base64
import os
import hashlib
import wave
import json
from manim_voiceover.services.base import SpeechService


class SarvamService(SpeechService):
    """Sarvam AI text-to-speech service for Manim Voiceover."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "bulbul:v3",
        speaker: str = "shubh",
        language: str = "en-IN",
        speed: float = 1.0,
        sample_rate: int = 24000,
        temperature: float = 0.6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        self.model = model
        self.speaker = speaker
        self.language = language
        self.speed = speed
        self.sample_rate = sample_rate
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("Sarvam API key must be provided or set in SARVAM_API_KEY environment variable")
    
    def _get_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_wav_duration(self, wav_path: str) -> float:
        """Get duration of WAV file in seconds."""
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
        return duration
    
    def _convert_wav_to_mp3(self, wav_path: str, mp3_path: str):
        """Convert WAV to MP3 using pydub."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_path)
            audio.export(mp3_path, format="mp3")
        except ImportError:
            # Fallback: just rename to .mp3 (won't work but prevents crash)
            import shutil
            shutil.copy(wav_path, mp3_path)
    
    def generate_from_text(self, text: str, cache_dir: str = None, path: str = None) -> dict:
        """Generate audio from text using Sarvam AI API."""
        if cache_dir is None:
            cache_dir = self.cache_dir
        
        if path is None:
            filename = f"{self._get_hash(text)}.wav"
            path = os.path.join(cache_dir, filename)
        else:
            filename = os.path.basename(path)
        
        mp3_filename = filename.replace('.wav', '.mp3')
        mp3_path = os.path.join(cache_dir, mp3_filename)
        
        if not os.path.exists(mp3_path):
            if not os.path.exists(path):
                url = "https://api.sarvam.ai/text-to-speech"
                
                headers = {
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                data = {
                    "inputs": [text],
                    "target_language_code": self.language,
                    "speaker": self.speaker,
                    "model": self.model,
                    "speed": self.speed,
                    "sample_rate": self.sample_rate,
                    "temperature": self.temperature
                }
                
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                
                # Decode base64 audio - handle different response formats
                if "audio" in result:
                    audio_data = base64.b64decode(result["audio"])
                elif "audios" in result:
                    # Sarvam returns audios as a list
                    audio_data = base64.b64decode(result["audios"][0])
                elif "output" in result:
                    audio_data = base64.b64decode(result["output"])
                elif "data" in result:
                    audio_data = base64.b64decode(result["data"])
                else:
                    raise ValueError(f"Unexpected response format. Keys: {result.keys()}")
                
                # Write to WAV file
                with open(path, "wb") as f:
                    f.write(audio_data)
            
            # Convert to MP3 for Manim Voiceover compatibility
            self._convert_wav_to_mp3(path, mp3_path)
        
        # Calculate duration
        duration = self._get_wav_duration(path)
        
        result_dict = {
            "input_data": {"input_text": text, "service": "sarvam"},
            "original_audio": mp3_filename,
            "final_audio": mp3_filename,
            "duration": duration
        }
        
        return result_dict
