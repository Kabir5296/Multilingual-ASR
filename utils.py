import torchaudio
from banglanlptoolkit import BanglaPunctuation
import torch, gc, os
import numpy as np
import io, numpy
import base64
from scipy.io.wavfile import write
from whisperx import DiarizationPipeline
bnpunct = BanglaPunctuation()

def numpytobytes(audio_data: numpy.ndarray, sample_rate: int = 16000) -> str:
    '''
    Converts audio numpy array to base64 encoding.
    
    Arguements:
    -----------
        audio_data (numpy.ndarray): The single channel audio numpy array that needs to be converted.
        sample_rate (int, Optional): Sampling rate of the audio.
        
    Returns:
    --------
        String: A string that represents base64 encoding of audio array.
    '''
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio_data)
    buffer.seek(0)
    b64_encoded_audio = base64.b64encode(buffer.getvalue())
    return b64_encoded_audio
    
def get_audio(audio_path: str) -> tuple:
    '''
    Load audio, resample and then return single channel of that audio and sampling rate.
    
    Arguements:
    -----------
    
        audio_path (str): Path to the audio file as a string.
    
    Returns:
    --------
    
        Returns a tuple of audio as ndarray and sampling rate as an integer.
    '''
    arr, org_sr = torchaudio.load(audio_path)
    arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=16000)
    return arr.numpy()[0], 16000

def post_process_bn(text: str) -> str:
    '''
    Post process Bengali transcripted string.
    
    Arguements:
    -----------
        text (str): String need to be post processed.
    
    Returns:
    --------
        Post processed Bengali string.
    '''
    if len(text) <= 1:
        text = ''
        
    text = text.replace('ট্রেনিং প্রেসিডেন্ট','')
    text = text.replace('ট্রেনিং প্রেসিডেন্ট','')
    text = text.replace('প্রেসিডেন্ট প্রেসিডেন্ট','')
    text = text.replace('প্রেসিডেন্ট প্রেসিডেন্ট প্রেসিডেন্ট','')
    text = text.replace('আসসালামু আলাইকুম','')
    text = bnpunct.add_punctuation(text)
    return text

def get_segments(audio_path: str, diarization_pipeline: DiarizationPipeline) -> tuple[list, list, list]:
    '''
    Returns diarized audio segments, timestamps and speakers information from a audio file.
    
    Arguements:
    -----------
        audio_path (str): Path to the audio file.
        diarization_pipeline (DiarizationPipeline): A diarization pipeline from pyannote. We recommend using Whisperx Diarization Pipeline.
        
    Returns:
    ---------
        Tuple of segments, timestamps and speakers.
        segments (list): Speakerwise separated audio segments/
        timestamp (list): A list that contains speaker information and start and end timestamp of each lines.
        speakers (list): Speakers listed as the flow of the conversation.
    '''
    audio_array, org_sr = get_audio(audio_path)
    diarization = diarization_pipeline(audio_path)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print('Diarization Done.')
    prev_sp_tag = -1
    speakers = []
    timestamp = []
    segments = []
            
    for _, data in enumerate(zip(diarization.speaker, diarization.start, diarization.end)):
        speaker , start, end = data
        start = int(start * org_sr)
        end = int(end * org_sr)
        sp_tag = speaker
        
        if sp_tag != prev_sp_tag:
            speakers.append(speaker)
            timestamp.append([speaker, start, end])
            segments.append(audio_array[start : end])
            
            prev_sp_tag = sp_tag
            prev_start = start
            prev_end = end
        elif sp_tag == prev_sp_tag:
            timestamp[-1] = [speaker, prev_start, end]
            segments[-1] = audio_array[prev_start: end]
            
    return segments, timestamp, speakers