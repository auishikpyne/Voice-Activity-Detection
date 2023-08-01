
# WebRTC-VAD (Modified)

**To run WebRTC-VAD, first execute:**

```bash
pip install -r requirements.txt
```
Please make sure to use Python 3.9.12 

Use [example.py](example.py) to get the Speech chunks or start end list of speech segments.

To optimize the chunks or control the aggressiveness of the vad output for various purposes please change the value between '0 and 3'. '3' is the most aggressive.

`vad = webrtcvad.Vad(3) # agressiveness`


