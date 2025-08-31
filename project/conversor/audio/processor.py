from pydub import AudioSegment# type: ignore
from pydub import AudioSegment# type: ignore
from pydub.silence import detect_nonsilent# type: ignore
import io
import librosa
import numpy as np
import torch# type: ignore
import torchaudio# type: ignore
import logging

logger = logging.getLogger(__name__)

class AudioProcessor():
    def _create_silence(self, duration_ms, sample_rate=24000, padding=0.95) -> np.ndarray:
        """Create a silence segment with specified duration."""
        silence_samples = int(duration_ms * sample_rate / 1000 * padding)
        return np.zeros(silence_samples)

    def _process_initial_audio(self, output, silence_duration) -> torch.Tensor:
        """Process the initial audio with silence removal and trimming."""
        audio_processed = self.remove_excessive_silence(output)
        audio_trim = librosa.effects.trim(audio_processed, top_db=60)[0]
        
        silence = self._create_silence(silence_duration)
        audio = np.concatenate((silence, audio_trim, silence), axis=None)
        
        return torch.tensor(audio).unsqueeze(0)

    def _refine_audio_segments(self, audio_segment, silence_duration) -> AudioSegment:
        """Refine audio segments by detecting non-silent ranges and adding padding."""
        non_silent_ranges = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-50)
        
        if non_silent_ranges:
            start_trim = non_silent_ranges[0][0]
            end_trim = non_silent_ranges[-1][1]
            audio_segment = audio_segment[start_trim:end_trim]

        silence_segment = AudioSegment.silent(duration=silence_duration)
        return silence_segment + audio_segment + silence_segment

    def apply_audio_silence(self, output) -> bytes:
        """Main method to process audio with silence padding."""
        silence_duration = 250
        logger.info(f"Applying audio silence: {silence_duration} ms")

        # Initial processing with numpy and torch
        buffer = io.BytesIO()
        audio_tensor = self._process_initial_audio(output, silence_duration)
        torchaudio.save(buffer, audio_tensor, 24000, format="wav")
        
        # Further processing with pydub
        audio_segment = AudioSegment.from_wav(buffer)
        padded_audio = self._refine_audio_segments(audio_segment, silence_duration)

        # Export final audio
        buffer = io.BytesIO()
        padded_audio.export(buffer, format="wav")
        buffer.seek(0)
        
        return buffer.read()

    def remove_excessive_silence(self, audio, max_silence_duration=30, sample_rate=24000, silence_db_threshold=55):
        """
        Remove silences longer than max_silence_duration and replace them with a fixed silence.
        
        :param audio: The input audio as a numpy array.
        :param max_silence_duration: The maximum allowed duration for silence in milliseconds.
        :param sample_rate: The sample rate of the audio.
        :param silence_db_threshold: The threshold in decibels to consider a segment as silence.
        :return: Audio with silences longer than max_silence_duration replaced by fixed-length silences.
        """
        max_silence_samples = int((max_silence_duration / 1000) * sample_rate)

        non_silent_intervals = librosa.effects.split(audio, top_db=silence_db_threshold)

        output_audio = []
        for i, (start, end) in enumerate(non_silent_intervals):
            output_audio.append(audio[start:end])
            
            if i < len(non_silent_intervals) - 1:
                next_start = non_silent_intervals[i + 1][0]
                silence_duration = next_start - end
                if silence_duration > max_silence_samples:
                    silence_duration = max_silence_samples
                
                silence = np.zeros(silence_duration, dtype=audio.dtype)
                output_audio.append(silence)

        return np.concatenate(output_audio)

def apply_audio_silence_clean(output) -> bytes:
    buffer = io.BytesIO()

    start_end_silence = 250
    padding = 0.95

    logger.info("Applying audio silence: {} ms".format(start_end_silence)) 

    audio_trim = librosa.effects.trim(output, top_db=50)[0]

    silence_duration_samples = int(start_end_silence * 24000 / 1000 * padding)
    silence = np.zeros(silence_duration_samples)

    audio = np.concatenate((silence, audio_trim, silence), axis=None)

    audio_tensor = torch.tensor(audio).unsqueeze(0)
    torchaudio.save(buffer, audio_tensor, 24000, format="wav")

    buffer.seek(0)
    audio_bytes = buffer.read()
    return audio_bytes