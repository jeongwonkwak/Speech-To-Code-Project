import sys
from datetime import datetime
import tkinter.font
from tkinter import*
from datetime import datetime
from PIL import Image, ImageTk
import threading
from random import *
import openpyxl
import matplotlib.pyplot as plt
import time

import speech_recognition as sr
import pyaudio
import wave
from pydub.playback import play
from pydub import AudioSegment
import numpy as np
import struct
import time
sr.__version__
import tkinter.ttk

FORMAT = pyaudio.paInt16
LEN = 10**100
PASS = 5
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 3
MIN_STRING_LIST_LENGTH = 9
WAVE_OUTPUT_FILENAME = "./data/wav/file.wav"

class Gui() :
    
    def __init__(self):
         
        self.window = tkinter.Tk()
        
        self.font = tkinter.font.Font(family = "나눔스퀘어라운드 Bold", size = 28)
        self.font3 = tkinter.font.Font(family = "나눔스퀘어라운드 Regular", size = 20)
        self.font4 = tkinter.font.Font(family = "나눔스퀘어라운드 Regular", size = 18)
        self.font5 = tkinter.font.Font(family = "나눔스퀘어라운드 Regular", size = 15)
        self.font6 = tkinter.font.Font(family = "나눔스퀘어라운드 Light", size = 12)
        
        self.window.title("STC(Speech to Code)프로그램 - 괭수팀")
        self.window.geometry("1090x800")
        self.window.resizable(False, False)
        

        self.show_outer_line()
        self.show_voice_listbox()
        self.show_label()
        self.show_button()
        self.show_combobox()
        self.show_checkbox()
        #arrow_image = Image.open("image/arrow.png")
        #arrow_image = arrow_image.resize((20, 10), Image.ANTIALIAS)
        
        #arrow = ImageTk.PhotoImage(arrow_image)
        #arrow_label = Label(self.window,image = arrow)
        #arrow_label.place(x = 360, y = 125, width = 90, height = 60)
        
        t = threading.Thread(target=self.voice_to_text)
        t.start()

        self.window.mainloop() 
        
    def show_label(self):
        
        self.label_voice=tkinter.Label(self.window, text="TEXT",font = self.font, fg = 'black')
        self.label_voice.place(x = 510, y = 110, width = 90, height = 30)
        
        self.label_code=tkinter.Label(self.window, text="CODE",font = self.font, fg = 'black')
        self.label_code.place(x = 860, y = 110, width = 90, height = 30)
        
        label_convert=tkinter.Label(self.window, text="convert",font = self.font4, fg = 'grey')
        label_convert.place(x = 420, y = 45, width = 90, height = 30)
        
        label_record=tkinter.Label(self.window, text="record",font = self.font4, fg = 'grey')
        label_record.place(x = 60, y = 45, width = 90, height = 30)
        
        label_langugae=tkinter.Label(self.window, text="language",font = self.font4, fg = 'grey')
        label_langugae.place(x = 60, y = 375, width = 100, height = 30)
        
        label_software=tkinter.Label(self.window, text="software",font = self.font4, fg = 'grey')
        label_software.place(x = 60, y = 515, width = 100, height = 30)
        
        arrow = PhotoImage(file="next.png")
        self.arrow = Label(image=arrow, height=40)
        self.arrow.image = arrow
        self.arrow.place(x = 710, y = 400, width = 45, height = 50)
    
    def show_voice_listbox(self) :
        
        #scrollbar=tkinter.Scrollbar(self.window,relief='solid',bd = 4)
        #scrollbar.place(x = 160, y = 100, width = 30, height = 80)
        
        self.voice_listbox=tkinter.Listbox(self.window, relief='groove', bd=2, font=self.font5)
        self.voice_listbox.place(x = 405, y = 150, width = 300, height = 610)
        
        self.code_listbox=tkinter.Listbox(self.window, relief='groove', bd=2, font=self.font5)
        self.code_listbox.place(x = 755, y = 150, width = 300, height = 610)
    
        
    def show_outer_line(self):
        
        outer_line1=tkinter.Label(self.window, relief="groove",bd = 2)
        outer_line1.place(x = 390, y = 60, width = 680, height = 720)
        
        outer_line2=tkinter.Label(self.window, relief="groove",bd = 2)
        outer_line2.place(x = 30, y = 60, width = 330, height = 300)
        
        outer_line3=tkinter.Label(self.window, relief="groove",bd = 2)
        outer_line3.place(x = 30, y = 385, width = 330, height = 120)
        
        outer_line4=tkinter.Label(self.window, relief="groove",bd = 2)
        outer_line4.place(x = 30, y = 530, width = 330, height = 150)  
        
        
    def show_button(self) :
 
        self.button_start = tkinter.Button(self.window, relief="raised" ,repeatdelay=1000, repeatinterval=1000, \
                                     bg = 'white',bd = 3,text = "start",font = self.font, highlightcolor = 'grey')
        self.button_start.place(x= 65,y = 100,width = 255,height = 50)
        
        self.button_stop = tkinter.Button(self.window, relief="raised" ,repeatdelay=1000, repeatinterval=1000, \
                                     bg = 'white',bd = 3,text = "stop",font = self.font, highlightcolor = 'grey')
        self.button_stop.place(x= 65,y = 180,width = 255,height = 50)
        
        self.button_play = tkinter.Button(self.window, relief="raised" ,repeatdelay=1000, repeatinterval=1000, \
                                     bg = 'white',bd = 3,text = "play",font = self.font)
        self.button_play.place(x= 65,y = 260,width = 255,height = 50)
        
        self.button_send = tkinter.Button(self.window, relief="groove" ,repeatdelay=1000, repeatinterval=1000, \
                                     bd = 3,text = "send",font = self.font4)
        self.button_send.place(x= 170,y = 675,width = 150,height = 30)
        
        self.delete=tkinter.Button(self.window, text="새로고침", font=self.font6, command=self.deleteText)
        self.delete.place(x=30, y=720, width=70, height=40)
        
    def getText(self):

        self.result1 = self.voice_listbox.get(0, "end")
        self.result2 = self.code_listbox.get(0, "end")
        return self.result1+self.result2
        
    def deleteText(self):
        
        result = self.getText()
        for i in result:
            self.voice_listbox.delete(0, "end")
            self.code_listbox.delete(0, "end")
            
            
    def show_combobox(self) :

        values1 = ["  Python","  C", "  Java"]
        combobox1=tkinter.ttk.Combobox(self.window, height=10, values=values1, font = self.font3)
        combobox1.place(x= 65,y = 440,width = 255,height = 50) 
        combobox1.set("           Python")
        
        values2 = ["  Jupyter Notebook","  Pycharm"]
        combobox2=tkinter.ttk.Combobox(self.window, height=10, values=values2, font = self.font4)
        combobox2.place(x= 65,y = 610,width = 255,height = 50)
        combobox2.set("   Jupyter Notebook")
    
    def show_checkbox(self) :
        
        checkVar1=tkinter.IntVar()
        ckeck_box=tkinter.Checkbutton(self.window,text=" auto",variable=checkVar1,font = self.font4)
        ckeck_box.place(x = 60, y = 670)
    
        
    def voice_to_text(self):
        
        
        while(1):     
            
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
                string = np.frombuffer(data, np.int16)[0]
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
                    self.voice_listbox.insert(END,sentence)
                    print(sentence)
                    if sentence in '종료':
                            
                        break
                except:
                    print('*** 다시 말해주세요 ***')
                    
    def voice_play(self):
        audio_file = AudioSegment.from_file(file="./data/wav/file.wav")
        play(audio_file)
                            
                    
    Gui()
