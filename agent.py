import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from pyannote.audio import Pipeline
import torchaudio, gc, os
# from dotenv import load_dotenv
from datetime import datetime
from Levenshtein import distance as levenshtein_distance
import numpy, whisperx
from docx import Document
from typing import Type, Union
from utils import get_audio, get_segments, post_process_bn, numpytobytes

# load_dotenv()
class CONFIG:
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    chunk_length_s=30
    batch_size=12
    torch_dtype=torch.float16
    # token = os.getenv('HUGGINGFACE_TOKEN')
    transcription_model='kabir5297/whisper_bn_medium'
    diarization_model="pyannote/speaker-diarization-3.1"
    en_transcription_model = 'distil-whisper/distil-large-v2'
    punctuation_model = 'kabir5297/bn_punctuation_model'
    default_language = 'bn'
    keywords = ['ডিউলাক্স','নিরোলাক']
    threshold = 0.75
    id2label = {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2', 3: 'LABEL_3'}
    label2id = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2, 'LABEL_3': 3}
    id2punc = {0: '', 1: '।', 2: ',', 3: '?'}
    
    
class TranscriberAgent():
    def __init__(self,CONFIG: CONFIG = CONFIG) -> None:
        '''
        This is the main agent. You can use the mehtods for raw transcription, diarization and conversation generation, adding punctuation and getting highlighted keywords.
        
        Arguements:
        -----------
            CONFIG: Class of configs. It contains the following values in a class.
            
                device: Device to be used for loading models and inference. By default, it looks for a GPU and if not available, CPU is used.
                
                chunk_length: Audio chunk length for Whisper transcription in seconds. Max value is 30s. Audio longer than 30s will be created into chunks and then transcripted. Defualt value is 30s.
                
                batch_size: Batch size used for model inference. Default value is 24.
                
                torch_dtype: The insanely fast Whisper inference requires the inference data type to be float16.
                
                token: Huggingface token.
                
                transcription_model: Repository for Bengali transcription model.
                
                diarization_model: Repository for Diarization model. By default, we use pyannote version 3.1
                
                en_transcription_model: Repository for English transcription model.
                
                punctuation_model: Repository for punctuation model.
                
                default_language: The methods can take two languages for input, 'bn' and 'en'. If not defined, the models will use 'bn' by default.
                
                keywords: List of words to be searched in the transcriptions. The default list of words for keywords are: ['ডিউলাক্স','এসিআই','নিরোলাক']. You can use your own words to be searched simply with comma (,) separated list.
                
                threshold: The keyword search method uses Levenshtein distance for calculating similarity. The similarity threshold is then used for identifying similar words. The default value is 0.75.
                
                id2label: The id2label dictionary for punctuation model.
                
                label2id: The label2id dictionary for punctuation model.
                
                id2punc: Dictionary used for converting ids to punctuations.
        '''
        self.CONFIG = CONFIG
        
        self.transcription_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.CONFIG.transcription_model,
            torch_dtype=self.CONFIG.torch_dtype,
            device=self.CONFIG.device,
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
            # token=self.CONFIG.token,
            )

        self.en_transcription_pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self.CONFIG.en_transcription_model,
            torch_dtype=self.CONFIG.torch_dtype,
            device=self.CONFIG.device,
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
            )
        
        self.diarization_pipeline = whisperx.DiarizationPipeline(model_name=self.CONFIG.diarization_model,
                                                                 use_auth_token=self.CONFIG.token,
                                                                 device=self.CONFIG.device)
        
        self.punctuation_pipeline = pipeline(
            task = 'ner',
            model=self.CONFIG.punctuation_model,
            device=self.CONFIG.device,
            # token=self.CONFIG.token,
        )
    
    def get_raw_transcription(self, audio_path: Union[str, numpy.ndarray], language: str ='bn') -> str:
        '''
        Get raw audio transcription of an audio path or audio file.
        
        Arguements:
        -----------
        
            audio_path (str or numpy.ndarray): Path to an audio file or a numpy array of audio.
            
            language (str, Optional): Language to use for transcription. 'bn' for Bengali and 'en' for English.
        
        Returns:
        --------
        
            Transcripted string with punctuation.
        '''
        if language == 'bn':
            transcriptions = self.transcription_pipeline(audio_path,
                                                    batch_size=self.CONFIG.batch_size,
                                                    chunk_length_s=self.CONFIG.chunk_length_s,
                                                    return_timestamps=False,
                                                    )
            if type(transcriptions) == dict:
                transcriptions = post_process_bn(transcriptions['text'])
            elif type(transcriptions) == list:
                for transcription in transcriptions:
                    transcription['text'] = post_process_bn(transcription['text'])
            
        elif language != 'bn':
            transcriptions = self.en_transcription_pipeline(audio_path,
                                                    batch_size=self.CONFIG.batch_size,
                                                    chunk_length_s=self.CONFIG.chunk_length_s,
                                                    return_timestamps=False,
                                                    )
            if type(transcriptions) == dict:
                transcriptions = transcriptions['text']
            elif type(transcriptions) == list:
                for transcription in transcriptions:
                    transcription['text'] = transcription['text']
            
        torch.cuda.empty_cache()
        gc.collect()
        return transcriptions
    
    def create_conversation(self,audio_path: str,language: str ='bn') -> list:
        '''
        Diarize the audio file, transcribe with punctuations and generate conversation.
        
        Arguements:
        -----------
        
            audio_path (str): Path to audio file. Only path is allowed, numpy array of audio file won't work.
            
            language (str, Optional): Language to use for transcription. 'bn' for Bengali and 'en' for English.
        
        Returns:
        --------
        
            List of list of strings. Each list in the entire list contains 2 strings, first one is the speaker tag and the second one is the transcribed string.        
        '''        
        segments, _, speakers = get_segments(audio_path=audio_path, diarization_pipeline=self.diarization_pipeline)
        
        diarize = []
        diarized = self.get_raw_transcription(segments, language=language)
        del segments
        for speaker, transcription in zip(speakers, diarized):
            if transcription['text'] != '':
                diarize.append([speaker, transcription['text']])
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return diarize
    
    def get_keywords(self, audio_path: Union[str, numpy.ndarray], keywords: list = CONFIG.keywords, language: str='bn') -> dict:
        '''
        Count specified keywords from the transcription and return frequency of each words.
        
        Arguements:
        -----------
        
            audio_path (str or numpy.ndarray): Path to an audio file or a numpy array of audio.
            
            keywords (list, Optional): List of words to search for in the transcription.
            
            language (str, Optional): Language to use for transcription. 'bn' for Bengali and 'en' for English.
        
        Returns:
        --------
        
            Dictionary of keys and frequency of each keys in the transcribed text.
        '''
        sentence = self.get_raw_transcription(audio_path,language=language)
        tokens = list(set(sentence.split()))
        distance = []
        keys = []
        key_dict = {}
        for keyword in keywords:
            distance = []
            count = 0
            distance.append([1 - levenshtein_distance(token,keyword)/(max(len(token),len(keyword))) for token in tokens])
            for key in distance:
                for index, value in enumerate(key):
                    if value >= self.CONFIG.threshold:
                        count += 1
                        keys.append(tokens[index])
                        key_dict[keyword] = count
        torch.cuda.empty_cache()
        gc.collect()
        return {'keys':keys, 'count':key_dict}

if __name__=='__main__':
    agent = TranscriberAgent()
    print(agent.get_raw_transcription('test_audio_file.wav'))
    print(agent.create_conversation('test_audio_file.wav'))
    print(agent.get_keywords('test_audio_file.wav'))