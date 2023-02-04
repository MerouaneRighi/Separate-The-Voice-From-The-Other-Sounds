import librosa 
#load the audio clip
y,sr =librosa.load(r'D:/python_sound/gg.mp3')
S = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128)
#Use a machine learning model o separate the voice from the other sounds
voice =separate_voice(S)
#Save the separated voice audio
librosa.output.write_wave('separated_voice.wav',voice,sr)
librosa.output.write_wave('voice.wav',voice,y)