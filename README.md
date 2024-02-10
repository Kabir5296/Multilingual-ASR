## Multilingual ASR

This repository contains multilingual ASR for Bengali and English.

Created by <b>A F M Mahfuzul Kabir</b> \
<a href='mahfuzulkabir.com'>mahfuzulkabir.com</a> \
https://www.linkedin.com/in/mahfuzulkabir \

### How to use:

Initialize:
```
from DoBanglaSummarize import DoBanglaSummarize
from agent import TranscriberAgent

agent = TranscriberAgent()
summarizer_agent = DoBanglaSummarize()
```

Transcribe, Create Conversation and Get Keywords:
```
transcription = agent.get_raw_transcription(audio_path)
conversation = agent.create_conversation(audio_path)
keywords = agent.get_keywords(audio_path)
# you can also define your own set of keywords by passing it throug an arguement
keywords = agent.get_keywords(audio_path, keywords=list_of_keywords)
```

### FastAPI Integration

The repository is now integrated with FastAPI for hosting endpoints.

```
Endpoint 1, 2 & 3: Transcribe, Converse, Summarize
    Input:  'file': String = File path to the audio file.
            'language' : String = 'bn' for Bengali, 'en' for English
    
    Output: Dictionary =    {'content': 
                                {'filename': Filename, 
                                'transcription': Transcription (string) for endpoint 1, List of list for endpoint 2, Summary (string) for endpoint 3
                                }
                            }
```

```
Endpoint 4: Get Keyword
    Input:  'file' : String = File path to the audio file.
            'keyword_str' : String = String with keywords separated with comma ','. Example: 'Apple, Mango, Juice'
            'language' : String = 'bn' for Bengali, 'en' for English

    Output: Dictionary =    {'content': 
                                {'filename': Filename, 
                                'keywords': 
                                    {
                                        'keys': Keywords given in the input,
                                        'count': 
                                            {
                                                'key1' : count,
                                                'key2' : count,
                                                .
                                                .
                                            }
                                    }
                                }
                            }
```