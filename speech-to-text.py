import speech_recognition as sr
import pyaudio
import wave
from pydub.playback import play
from pydub import AudioSegment
import numpy as np
import struct
import time
sr.__version__

FORMAT = pyaudio.paInt16
LEN = 10**100
PASS = 5
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3
MIN_STRING_LIST_LENGTH = 9
WAVE_OUTPUT_FILENAME = "./data/wav/file.wav"



po = pyaudio.PyAudio()

for index in range(po.get_device_count()): 
    desc = po.get_device_info_by_index(index)
    print("INDEX:  %s  RATE:  %s   DEVICE: %s" %(index, int(desc["defaultSampleRate"]), desc["name"]))


while(True):
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=pyaudio.paInt16, 
                    channels=CHANNELS, 
                    rate=RATE, 
                    input=True, 
                    input_device_index=1,
                    frames_per_buffer=CHUNK)

    frames, string_list = [], []
    
    for i in range(LEN):
        data = stream.read(CHUNK)
        frames.append(data)
        string = np.fromstring(data, np.int16)[0]
        string_list.append(string)
        
         # stop Recording
        if string == 0 and i > PASS:
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    if len(string_list) > MIN_STRING_LIST_LENGTH:
        r = sr.Recognizer()
        korean_audio = sr.AudioFile("./data/wav/file.wav")

        with korean_audio as source:
            mandarin = r.record(source)
        
        try :
            sentence = r.recognize_google(audio_data=mandarin, language="ko-KR")
            print(sentence)
            if sentence in '종료':
                break
        except:
            print('*** 다시 말해주세요 ***')

            
