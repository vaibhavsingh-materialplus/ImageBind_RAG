from pydub import AudioSegment
import os

def convert_audio(input_file, output_file, bitrate=256000, channels=1, sample_rate=16000):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Set parameters
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(channels)
    # Set bit rate
    audio.export(output_file, format="wav", bitrate=f"{bitrate}bps")


#for file in os.listdir("./sample_audios"):
 #   convert_audio(f"./sample_audios/{file}",f"./sample_audios/{str(file)}.wav")
