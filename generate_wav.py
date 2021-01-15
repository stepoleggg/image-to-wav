import math
import wave
import struct
import cv2
import numpy as np
import sys

# Audio will contain a long list of samples (i.e. floating point numbers describing the
# waveform).  If you were working with a very long sound you'd want to stream this to
# disk instead of buffering it all in memory list this.  But most sounds will fit in 
# memory.
audio = []
sample_rate = 44100.0


def append_silence(duration_milliseconds=500):
    """
    Adding silence is easy - we add zeros to the end of our array
    """
    num_samples = duration_milliseconds * (sample_rate / 1000.0)

    for x in range(int(num_samples)): 
        audio.append(0.0)

    return


def append_sinewave(
        freq=440.0, 
        duration_milliseconds=500, 
        volume=1.0):
    """
    The sine wave generated here is the standard beep.  If you want something
    more aggresive you could try a square or saw tooth waveform.   Though there
    are some rather complicated issues with making high quality square and
    sawtooth waves... which we won't address here :) 
    """ 

    global audio # using global variables isn't cool.

    num_samples = duration_milliseconds * (sample_rate / 1000.0)

    for x in range(int(num_samples)):
        audio.append(volume * math.sin(2 * math.pi * freq * ( x / sample_rate )))

    return


def save_wav(file_name):
    # Open up a wav file
    wav_file=wave.open(file_name,"w")

    # wav params
    nchannels = 1

    sampwidth = 2

    # 44100 is the industry standard sample rate - CD quality.  If you need to
    # save on file size you can adjust it downwards. The stanard for low quality
    # is 8000 or 8kHz.
    nframes = len(audio)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))

    # WAV files here are using short, 16 bit, signed integers for the 
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.
    for sample in audio:
        wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))

    wav_file.close()

    return

def read_image_old(file_name, duration_milliseconds = 1000, min_freq = 20.0, max_freq = 20000.0):
  img = cv2.imread(file_name, 0)
  width = len(img[0])
  height = len(img)
  k = math.log10(max_freq / min_freq)

  global audio
  num_samples = int(duration_milliseconds * (sample_rate / 1000.0) / height) * height
  for y in range(int(num_samples)):
    print(y)
    v = 0
    layer_n = int(y / num_samples * height)
    next_layer_n = int(y / num_samples * height) + 1
    layer_weight = 1 - y / num_samples * height + layer_n

    for x in range(width):
      freq = min_freq * 10 ** (x / width * k)
      v += math.sin(2 * math.pi * freq * (y / sample_rate )) * layer_weight * img[layer_n][x]

    if next_layer_n < height:
      for x in range(width):
        freq = min_freq * 10 ** (x / width * k)
        v += math.sin(2 * math.pi * freq * (y / sample_rate )) * (1 - layer_weight) * img[next_layer_n][x]
    #print(y / num_samples * height)
    audio.append(v)

  audio = np.array(audio)
  audio = audio / np.linalg.norm(audio)

def read_image(file_name, duration_milliseconds = 10000, min_freq = 100.0, max_freq = 10000.0, logo=True, logo_root=2, random_phase=True):
  img = cv2.imread(file_name, 0)
  width = len(img[0])
  height = len(img)
  k = math.log(max_freq / min_freq, logo_root)

  global audio
  audio = np.array([])
  num_samples = int(duration_milliseconds * (sample_rate / 1000.0) / height) * height

  layer_freqs = np.arange(0, width, 1)
  if logo:
    layer_freqs = min_freq * logo_root ** (layer_freqs / width * k) 
  else:
    layer_freqs = layer_freqs / width * (max_freq - min_freq) + min_freq
  
  layer_phases = np.random.uniform(0.0, 1.0, width) * 2 * math.pi
  if not random_phase:
    layer_phases = np.zeros(width)
  samples_per_layer = int(num_samples / height)
  for y in range(height):
    full_layer = np.zeros(samples_per_layer)
    layer_sound = np.arange(y * samples_per_layer, (y + 1) * samples_per_layer, 1)
    layer_sound = layer_sound / sample_rate * 2 * math.pi
    for freq_n in range(width):
      i = freq_n
      phase = layer_phases[i]
      freq = layer_freqs[i]
      full_layer = full_layer + np.sin(layer_sound * freq + phase) * img[y][i]
    audio = np.concatenate((audio, full_layer))
  minv = abs(np.amax(audio))
  maxv = abs(np.amin(audio))
  if minv > maxv:
    maxv = minv
  audio = audio / maxv

#read_image('box.jpg', duration_milliseconds=10000, min_freq=100.0)
#save_wav("box.wav")
read_image(sys.argv[1])
save_wav(sys.argv[2])