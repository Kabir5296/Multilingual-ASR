## Multilingual ASR

This repository contains multilingual ASR for Bengali and English. The models used are mostly token secured for company purposes. But you can load your own model and use the library as it is. \

Created By: \
A F M Mahfuzul Kabir \
Machine Learning Engineer, \
ACI Limited

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

Summarize:
```
conversation = agent.create_conversation(audio_path)
dialogue = ''
for line in conversation:
    dialogue += ':'.join(line) + '. '
summary = summarizer_agent.summary_kore_felo(dialogue)
```
