from tkinter import *
from tkinter import ttk
from datetime import datetime
import os
import subprocess
import signal


class RecordingSetup:

    def __init__(self, root):

        root.title("Consultation Recorder")

        self.mainframe = ttk.Frame(root, padding="3 3 12 12")
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        
        for r in range(6):
            self.mainframe.rowconfigure(r, weight=1)
        for c in range(3):
            self.mainframe.columnconfigure(c, weight=1)

        # Doctor Id entry
        self.id = StringVar()
        id_entry = ttk.Entry(self.mainframe, width=7, textvariable=self.id)
        id_entry.grid(column=2, row=1, sticky=(W, E))
        ttk.Label(self.mainframe, text="Doctor Id").grid(column=1, row=1)

        # Error Label for Button
        self.error = StringVar()
        ttk.Label(self.mainframe, textvariable=self.error).grid(
            column=3, row=1)

        self.spec = StringVar()
        spec_entry = ttk.Combobox(self.mainframe, textvariable=self.spec)
        spec_entry.grid(column=2, row=2, sticky=(W, E))
        spec_entry['values'] = ('Spec1', 'Spec2', 'Spec3')
        spec_entry.state(["readonly"])
        ttk.Label(self.mainframe, text="Doctor Specialty").grid(
            column=1, row=2)

        self.mode = StringVar()
        mode_entry_presential = ttk.Radiobutton(
            self.mainframe, text="Presential", variable=self.mode, value="Presential")
        mode_entry_presential.grid(column=2)
        mode_entry_virtual = ttk.Radiobutton(
            self.mainframe, text="Virtual", variable=self.mode, value="Virtual")
        mode_entry_virtual.grid(column=2)
        ttk.Label(self.mainframe, text="Type Of Consultation").grid(
            column=1, row=3)
        self.flag = True
        # Record Button
        self.btn = ttk.Button(
            self.mainframe, text="Record", command=self.record)
        self.btn.grid(
            column=2, sticky=(W, E))

        for child in self.mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        self.recNotice = ttk.Label(
            self.mainframe, text="Recording", foreground='red')
        self.recNotice.grid(column=1, row=5)
        self.recNotice.grid_remove()

    def record(self, *args):
        self.btn.configure(text="Stop")
        self.btn.configure(command=self.stop)
        self.recNotice.grid(column=1, row=5)
        str = '../videos/Testing/' + self.id.get()+".mp4"

        # Command : ffmpeg -f video4linux2 -s hd720 -r 15 -input_format mjpeg -i /dev/video0 out.mp4
        self.p = subprocess.Popen(["ffmpeg", "-f", "video4linux2", "-s", "hd720",
                                   "-input_format", "mjpeg", "-i", "/dev/video0", "-r", "15", str])

    def stop(self, *args):
        self.btn.configure(text="Record")
        self.btn.configure(command=self.record)
        self.recNotice.grid_remove()
        self.p.send_signal(signal.SIGINT)


root = Tk()
RecordingSetup(root)
root.mainloop()
