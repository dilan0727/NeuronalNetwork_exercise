import os
import time
import pyaudio
import wave

def record_audio(output_folder, label, num_recordings):
    os.makedirs(output_folder, exist_ok=True)
    for i in range(1, num_recordings + 1):
        filename = os.path.join(output_folder, f'{label}_{i}.wav')
        print(f'Recording {label} {i}/{num_recordings}...')
        
        frames = []
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100
        seconds = 2  # Tiempo de grabación en segundos
        
        p = pyaudio.PyAudio()

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        print("Recording...")

        for _ in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f'Audio recorded and saved as {filename}\n')
        time.sleep(1)

def main():
    for num in range(1, 11):
        output_folder = f'train/{num}'
        record_audio(output_folder, str(num), 5)

if __name__ == "__main__":
    main()
