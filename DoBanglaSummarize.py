import torch
from transformers import pipeline
class CONFIG:
    batch_size=12
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    summary_model = 'facebook/bart-large-cnn'
    
class DoBanglaSummarize():
    def __init__(self, batch_size: int =CONFIG.batch_size, device: str =CONFIG.device) -> None:
        '''
        Summarizer agent class. This agent translates Bengali to English. Then summarizes the English translated text and translates back to Bengali. For English text, the text is direct summarized and returned, no translation is used.
        
        Arguements:
        -----------
            
            batch_size (str, Optional): Batch size to use for inference.
            
            device (str, Optional): Device to use for inference.
        '''
        self.batch_size = batch_size
        self.device = device
        
        self.bn2en = pipeline(model='csebuetnlp/banglat5_nmt_bn_en',
                task='translation',
                use_fast=False,
                batch_size=self.batch_size,
                device=self.device,
                )
        
        self.en2bn = pipeline(model='csebuetnlp/banglat5_nmt_en_bn',
                task='translation',
                use_fast=False,
                batch_size=self.batch_size,
                device=self.device,
                )

        self.summarize = pipeline(model=CONFIG.summary_model,
                task='summarization',
                use_fast=True,
                # batch_size=self.batch_size,
                device=self.device,
                )
        
    def summary_kore_felo(self, dialogue: str, language: str = 'bn') -> str:
        '''
        Summarize the given input.
        
        Arguements:
        -----------
            dialogue (str): Text file that needs to be summarized.
            
            language (str, Optional): The language being used for summarization. If 'bn', the Bengali is translated to English, summarized and then translated back to Bengali. If 'en', the English text is summarized and returned. Default value is 'bn'.
            
        Returns:
        --------
        
            String of summarized text.
        '''
        if language == 'bn':
            return ([bangla['translation_text'] for bangla in self.en2bn([summary['summary_text'] for summary in self.summarize([lines['translation_text'] for lines in self.bn2en(dialogue)])])])
        elif language == 'en':
            return [summary['summary_text']for summary in self.summarize(dialogue)]