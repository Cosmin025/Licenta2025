import tkinter as tk
from tkinter import filedialog
import threading
import os

from recognition import action_recognition

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def button_live_recognition_action(play: bool, save: bool):
    print(f'Live recognition, play = {play}, save = {save}')
    # threading.Thread(target=lambda: live_rec(play, save), daemon=True).start()
    threading.Thread(target = lambda : action_recognition(True,play,save,""),daemon=True).start()

def button_recording_recognition_action(play: bool, save: bool, video_path: str):
    print(f'Recording recognition, play = {play}, save = {save}, video = {video_path}')
    # threading.Thread(target=lambda: recording_rec(play, save, video_path), daemon=True).start()
    threading.Thread(target=lambda: action_recognition(False, play, save, video_path),daemon=True).start()

def browse_video():
    file_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if file_path:
        video_path_entry.delete(0, tk.END)
        video_path_entry.insert(0, file_path)



root = tk.Tk()
root.title("Human Action Recognition")
root.geometry("800x600")

center_frame = tk.Frame(root)
center_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


frame_live = tk.Frame(center_frame)
frame_live.pack(pady=10, anchor=tk.W)

button_live_recognition = tk.Button(
    frame_live, text="Live Recognition",
    command=lambda: button_live_recognition_action(live_play.get(), live_save_output.get())
)
button_live_recognition.pack(side=tk.LEFT, padx=10)


live_play = tk.BooleanVar()
live_play.set(True)
live_save_output = tk.BooleanVar()

frame_recording = tk.Frame(center_frame)
frame_recording.pack(pady=10, anchor=tk.W)

button_recording_recognition = tk.Button(
    frame_recording, text="Recording Recognition",
    command=lambda: button_recording_recognition_action(
        recording_play.get(),
        recording_save_output.get(),
        video_path_entry.get()
    )
)
button_recording_recognition.pack(side=tk.LEFT, padx=10)

video_path_entry = tk.Entry(frame_recording, width=50)
video_path_entry.pack(side=tk.LEFT, padx=5)

browse_button = tk.Button(frame_recording, text="Browse", command=browse_video)
browse_button.pack(side=tk.LEFT, padx=5)

recording_play = tk.BooleanVar()
recording_play.set(True)
recording_save_output = tk.BooleanVar()


root.mainloop()
