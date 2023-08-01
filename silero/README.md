# Silero-VAD (Modified)

**To run Silero-VAD, first execute:**

```bash
pip install -r requirements.txt
```
Please make sure to use Python 3.8.16 

Use [silero_vad_revised.py](silero_vad_revised.py) to get the Speech chunks or start end list of speech segments.

To optimize the chunks of the vad output for various purposes please change the value of these parameters `threshold` and `min_silence_duration_ms` 

Default `vad = SileroVAD(threshold=0.5, min_silence_duration_ms=100, use_onnx=False)`

