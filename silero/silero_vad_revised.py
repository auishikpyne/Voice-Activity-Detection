import torch 
import librosa
from pydub import AudioSegment
from typing import List, Tuple
from pathlib import Path
import os
import shutil
import torch.cuda


#torch.set_num_threads(1)

model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
            onnx=False
        )

class SileroVAD:
    def __init__(self, threshold:float = 0.5, min_silence_duration_ms: int = 100, use_onnx: bool = False):
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.use_onnx = use_onnx
        self.model = model
        self.utils = utils
        
        
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = self.utils

        self.window_size_samples = 1536
        self.sampling_rate = 16000
        
    def process_audio(self, audio_path:str) -> List[Tuple[float, float]]:
        start_end_list = []
        vad_iterator = self.VADIterator(self.model.to("cuda:1"), threshold=self.threshold, min_silence_duration_ms=self.min_silence_duration_ms) # for not using cuda use self.model
        wav = self.read_audio(audio_path, sampling_rate=self.sampling_rate).to('cuda:1') # for not using cuda remove it
        
        for i in range(0, len(wav), self.window_size_samples):
            chunk = wav[i: i+ self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                break
            speech_dict = vad_iterator(chunk, return_seconds=True)
            
            if speech_dict:
                start_end_list.extend(speech_dict.values())

        vad_iterator.reset_states()

        return start_end_list
    
    def split_audio(self, audio_path: str, output_dir: str) -> List[str]:
        audio = AudioSegment.from_file(audio_path)
        duration = audio.duration_seconds
        print(duration)
        start_end_list = self.process_audio(audio_path)
        print(start_end_list)
        if len(start_end_list) % 2 == 0:
            pass
        else:
            start_end_list.append(duration)
            
        print(start_end_list)
            
        start_end_list_pairs = [[start_end_list[i], start_end_list[i+1]] for i in range(0, len(start_end_list)-1, 2)]
        print(start_end_list_pairs)
        output_paths = []
        for i, (start_time, end_time) in enumerate(start_end_list_pairs):
            start_ms = start_time * 1000
            end_ms = end_time * 1000

            segment = audio[start_ms:end_ms]
            output_path = Path(output_dir) / f"segment_{i}.flac"
            segment.export(output_path, format="flac")
            output_paths.append(str(output_path))

        return output_paths
    
if __name__ == '__main__':
    
   
    input_audio_path = "/home/auishik/silero_vad/Long_seq/db067d5b-5fed-450c-871f-28d3ff9279e8.flac"
    output_dir = "/home/auishik/silero_vad/silero-vad/saved_chunks"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    os.mkdir(output_dir)

    vad = SileroVAD(threshold=0.5, min_silence_duration_ms=100, use_onnx=False)

    start_end_list = vad.process_audio(input_audio_path)
    

    output_paths = vad.split_audio(input_audio_path, output_dir)
    
        
        