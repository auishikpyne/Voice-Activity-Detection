import librosa
import torch
from pydub import AudioSegment
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint
SAMPLING_RATE = 16000

USE_ONNX = False # change this to True if you want to test onnx model
# if USE_ONNX:
    # !pip install -q onnxruntime
print('here')
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)



(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils



## using VADIterator class
start_end_list = []
vad_iterator = VADIterator(model, threshold=0.8, min_silence_duration_ms=50)
wav = read_audio(f'/home/auishik/silero_vad/Long_seq/045a3de9-2503-4f27-bc40-db3b1909f5c3.flac', sampling_rate=SAMPLING_RATE)

window_size_samples = 1536 # number of samples in a single audio chunk
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    # print("speech dict . . . ", speech_dict.values())
    
    if speech_dict:
      
        print(list(speech_dict.values()), end=' ')
        for value in list(speech_dict.values()):
          print(value)
          start_end_list.append(value)
print(start_end_list)
start_end_list_pairs = [[start_end_list[i], start_end_list[i+1]] for i in range(0, len(start_end_list)-1, 2)]
print(start_end_list_pairs)

audio = AudioSegment.from_file("/home/auishik/silero_vad/Long_seq/045a3de9-2503-4f27-bc40-db3b1909f5c3.flac")

for i, (start_time, end_time) in enumerate(start_end_list_pairs):
  
  start_ms = start_time * 1000
  end_ms = end_time * 1000
  
  segment = audio[start_ms:end_ms]
  segment.export(f"/home/auishik/silero_vad/silero-vad/saved_chunks/segment_{i}.flac", format="flac")
# vad_iterator.reset_states() # reset model states after each audio

print('done')