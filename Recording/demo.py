#!/usr/bin/env python3

from tkinter import *
from tkinter import ttk
from datetime import datetime
import os
import subprocess
import signal


class RecordingSetup:

    def __init__(self, root):

        root.title("Consultation Recorder")
        
        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

       
        # Doctor Id entry
        self.id = StringVar()
        id_entry = ttk.Entry(mainframe, width=7, textvariable=self.id)
        id_entry.grid(column=2, row=1, sticky=(W, E))
        ttk.Label(mainframe, text="Doctor Id").grid(column=1, row=1)

        # Error Label for Button
        self.error = StringVar()
        ttk.Label(mainframe, textvariable=self.error).grid(column=3, row=1)

        self.spec = StringVar()
        spec_entry = ttk.Combobox(mainframe, textvariable=self.spec)
        spec_entry.grid(column=2, row=2, sticky=(W, E))
        spec_entry['values'] = ('Neurologia', 'MedInt', 'MedGeral', 'Genecologia')
        spec_entry.state(["readonly"])
        ttk.Label(mainframe, text="Doctor Specialty").grid(column=1, row=2)

        
        
        self.mode =  StringVar()
        mode_entry_presential = ttk.Radiobutton(mainframe, text="Presential", variable=self.mode, value="Presential")
        mode_entry_presential.grid(column=2)
        mode_entry_virtual = ttk.Radiobutton(mainframe, text="Virtual", variable=self.mode, value="Virtual")
        mode_entry_virtual.grid(column=2)
        ttk.Label(mainframe, text="Type Of Consultation").grid(column=1, row=3)

        #Record Button
        self.btn = ttk.Button(mainframe, text="Record", command=self.record)
        self.btn.grid(
            column=2,sticky=(W, E))
        

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.recNotice = ttk.Label(
            mainframe, text="Recording", foreground='red')
        self.recNotice.grid(column=1, row=5)
        self.recNotice.grid_remove()
        
    def record(self, *args):
        self.btn.configure(text="Stop")
        self.btn.configure(command=self.stop)
        self.recNotice.grid(column=1, row=5)

        
        
        #str = '../videos/Testing/' + self.id.get()+".mp4"
        
        id = self.id.get()
        mode =  self.mode.get()
        spec = self.spec.get()
        curr_day = datetime.now().strftime("%d-%m-%Y") 
        curr_time = datetime.now().strftime("%d-%m-%Y__%H_%M_%S") 
        
        spec_dir = f'/media/vislab/My_Passport/Tese/videos/{spec}'
        #spec_dir = f'/home/vislab/Eye-to-Eye/Recording/{spec}'
        doctor_dir = f'{spec_dir}/D{id}'
        mode_dir = f'{doctor_dir}/{mode}'
        day_dir = f'{mode_dir}/{curr_day}'
        
        
        
        if not os.path.exists(spec_dir):
            print(f'No {spec_dir} directory found \n Creating it \n')
            os.mkdir(spec_dir)
        if not os.path.exists(doctor_dir):
            print(f'No {doctor_dir} directory found \n Creating it \n')
            os.mkdir(doctor_dir)
        if not os.path.exists(mode_dir):
            print(f'No {mode_dir} directory found \n Creating it \n')
            os.mkdir(mode_dir)
        if not os.path.exists(day_dir):
            print(f'No {day_dir} directory found \n Creating it \n')
            os.mkdir(day_dir)
            
         
        print("Recording video to " + f'D{id}/{mode}/{curr_day}/D{id}_{mode[0]}-{curr_time}.mp4')
        #str = "../videos/" + self.mode.get() + "/" +  self.id.get() + self.timestamp.strftime("-T-%m-%d-%Y__%H_%M_%S") +".mp4"
        name = f'D{id}_{mode}-{curr_time}.mp4'
        
        video_location = f'{day_dir}/{name}'  
        # Command : ffmpeg -f video4linux2 -s hd720 -r 15 -input_format mjpeg -i /dev/video0 out.mp4
        self.p = subprocess.Popen(["ffmpeg", "-f", "video4linux2", "-s","hd720","-input_format","mjpeg", "-i", "/dev/video0", "-r", "15", video_location])
       

    def stop(self, *args):
        self.btn.configure(text="Record")
        self.btn.configure(command=self.record)
        self.recNotice.grid_remove()

        self.p.send_signal(signal.SIGINT)
        


root = Tk()
RecordingSetup(root)
root.mainloop()
