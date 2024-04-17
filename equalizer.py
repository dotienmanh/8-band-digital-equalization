import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from numpy.fft import fft, ifft
import wave


def fixed_point_binary_to_float(value, total_bits, fraction_bits, is_signed=True):
    scale = 2 ** fraction_bits

    if is_signed and value[0] == '1':
        # For signed numbers, if the MSB is 1, it's negative
        value_int = int(value, 2) - (1 << total_bits)
    else:
        value_int = int(value, 2)

    return value_int / scale

def apply_filter(audio_data, hn):
        # Assumming 16-bit PCM wav format
        audio_data = np.convolve(audio_data, hn, mode='same')
        # Ensure 16-bit PCM format
        audio_data = np.asarray(audio_data, dtype=np.int16)
        return audio_data

def spectrum(h, fs,linewidth=2,color="red"):
    w, H = signal.freqz(h, fs=fs)
    frequencies = w # Hz
    magnitude = 20 * np.log10(np.abs(H)) # dB
    plt.figure()
    plt.title('Frequency Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.plot(frequencies, magnitude, linewidth=linewidth, color=color)
    plt.show()

def low_pass(audio_data,fc,fs,m):
    PI = 3.14
    FC = fc/(fs) 
    M = m 
    H = np.zeros(M+1)
    for i in range(M+1):
        if (i - M//2) == 0:
            H[i] = 2 * PI * FC
        else:
            H[i] = np.sin(2 * PI * FC * (i - M//2)) / (i - M//2)
        H[i] *= (0.54 - 0.46 * np.cos(2 * PI * i / M))

    sum_H = np.sum(H)
    H /= sum_H
    Y = np.zeros_like(audio_data)
    Y=apply_filter(audio_data,H)
    return H,Y

def band_pass(audio_data,fl,fh,fs,m):
    PI = 3.14
    M = m 
    A = np.zeros(M+1)
    B = np.zeros(M+1)
    H = np.zeros(M+1)
    FC_A = fl/(fs)  
    for i in range(M+1):
        if (i - M//2) == 0:
            A[i] = 2 * PI * FC_A
        else:
            A[i] = np.sin(2 * PI * FC_A * (i - M//2)) / (i - M//2)
        A[i] *= (0.54 - 0.46 * np.cos(2 * PI * i / M))
    sum_A = np.sum(A)
    A /= sum_A
    FC_B = fh/(fs)  
    for i in range(M+1):
        if (i - M//2) == 0:
            B[i] = 2 * PI * FC_B
        else:
            B[i] = np.sin(2 * PI * FC_B * (i - M//2)) / (i - M//2)
        B[i] *= (0.54 - 0.46 * np.cos(2 * PI * i / M))
    sum_B = np.sum(B)
    B /= sum_B
    for i in range(M+1):
        B[i] = -B[i]
    B[M//2] += 1

    for i in range(M+1):
        H[i] = A[i] + B[i]

    for i in range(M+1):
        H[i] = -H[i]
    H[M//2] += 1
    
    Y = np.zeros_like(audio_data)
    Y=apply_filter(audio_data,H)
    return H,Y

def high_pass(audio_data,fc,fs,m):
    PI = 3.1415926
    M=m  
    H = np.zeros(M+1)
    FC_H = fc/(fs) 
    for i in range(M+1): #create h[n]
        if (i - M//2) == 0:
            H[i] = 2 * PI * FC_H
        else:
            H[i] = np.sin(2 * PI * FC_H * (i - M//2)) / (i - M//2)
        H[i] *= (0.42 - 0.5 * np.cos(2 * PI * i / M) + 0.08 * np.cos(4 * PI * i / M))

    sum_H = np.sum(H)
    H /= sum_H
    for i in range(M+1):
        H[i] = -H[i]
    H[M//2] += 1
  
    Y = np.zeros_like(audio_data)#create output
    Y=apply_filter(audio_data,H)
    return H,Y

def combine_wav(audio1, audio2):
        # Determine the maximum length among the audio signals
        max_length = max(len(audio1), len(audio2))
        # Ensure both signals have the same length
        audio1 = np.pad(audio1[:max_length], (0, max(
            0, max_length - len(audio1))), 'constant')
        audio2 = np.pad(audio2[:max_length], (0, max(
            0, max_length - len(audio2))), 'constant')
        # Perform Fourier transform
        this_fft = fft(audio1)
        other_fft = fft(audio2)
        # Add frequency representations
        combined_fft = this_fft + other_fft
        # Take inverse Fourier transform
        combined_audio = np.real(ifft(combined_fft))
        return combined_audio
    
def gain(audio_data,gain):
    audio=fft(audio_data)*(gain)
    audio=np.real(ifft(audio))
    return audio

def equalizer(audio_data,gaindata):
    h1,audio1=low_pass(audio_data,100,16000,14)
    print("low pass:")
    #spectrum(h1,16000)
    print(h1)
    h1_fp=convert_float_to_fixed_point(h1,16,15)
    write_fixed_point_to_file(h1_fp,"b1.txt")
    print('\n')
    audio1=gain(audio1,gaindata[0])

    h8,audio8=high_pass(audio_data,7000,16000,14)
    print("high pass:")
    print(h8)
    #spectrum(h8,16000)
    h8_fp=convert_float_to_fixed_point(h8,16,15)
    write_fixed_point_to_file(h8_fp,"b8.txt")
    print('\n')

    audio8=gain(audio8,gaindata[7])
    output=combine_wav(audio1,audio8)


    h2,audio2=band_pass(audio_data,100,250,16000,14)
    print("band pass 2:")
    print(h2)
    #spectrum(h2,16000)
    h2_fp=convert_float_to_fixed_point(h2,16,15)
    write_fixed_point_to_file(h2_fp,"b2.txt")
    audio2=gain(audio2,gaindata[1])
    output=combine_wav(output,audio2)

    h3,audio3=band_pass(audio_data,250,750,16000,14)
    print("band pass 3:")
    #spectrum(h3,16000)
    print(h3)
    h3_fp=convert_float_to_fixed_point(h3,16,15)
    write_fixed_point_to_file(h3_fp,"b3.txt")
    audio3=gain(audio3,gaindata[2])
    output=combine_wav(output,audio3)

    h4,audio4=band_pass(audio_data,750,1500,16000,14)
    print("band pass 4:")
    #spectrum(h4,16000)
    print(h4)
    h4_fp=convert_float_to_fixed_point(h4,16,15)
    write_fixed_point_to_file(h4_fp,"b4.txt")
    audio4=gain(audio4,gaindata[3])
    output=combine_wav(output,audio4)

    h5,audio5=band_pass(audio_data,1500,2500,16000,14)
    print("band pass 5:")
    #spectrum(h5,16000)
    print(h5)
    h5_fp=convert_float_to_fixed_point(h5,16,15)
    write_fixed_point_to_file(h5_fp,"b5.txt")
    audio5=gain(audio5,gaindata[4])
    output=combine_wav(output,audio5)

    h6,audio6=band_pass(audio_data,2500,4000,16000,14)
    print("band pass 6:")
    #spectrum(h6,16000)
    print(h6)
    h6_fp=convert_float_to_fixed_point(h6,16,15)
    write_fixed_point_to_file(h6_fp,"b6.txt")
    audio6=gain(audio6,gaindata[5])
    output=combine_wav(output,audio6)

    h7,audio7=band_pass(audio_data,4000,7000,16000,14)
    print("band pass 7:")
    #spectrum(h7,16000)
    print(h7)
    h7_fp=convert_float_to_fixed_point(h7,16,15)
    write_fixed_point_to_file(h7_fp,"b7.txt")
    audio7=gain(audio7,gaindata[6])
    output=combine_wav(output,audio7)


    return output

def write_fixed_point_to_file(data, filename):
  """Writes a list of 16-bit fixed point numbers to a text file with their binary representation (little endian).

  Args:
      data: A list of 16-bit signed integers representing fixed point values.
      filename: The name of the text file to write to.
  """
  with open(filename, "w") as file:
    for value in data:
      # Convert integer to binary string with leading zeros (16 bits)
      binary_data = format(value & 0xFFFF, '016b')  # Mask with 0xFFFF to get 16 bits
      # Write binary data to file as a string
      file.write(binary_data + "\n")

def load_from_txt(txt_path):
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                audio_samples = np.empty(len(lines), dtype=np.int16)
                for i, line in enumerate(lines):
                    binary_data = line.strip()
                    # Convert binary data back to audio sample
                    sample = int(binary_data, 2)
                    if sample >= 1 << (2 * 8 - 1):
                        sample -= 1 << (2 * 8)
                    audio_samples[i] = sample
                audio_data = audio_samples
                print(f"Loaded audio samples from {txt_path}")
                return audio_data
        except FileNotFoundError:
            print(f"Error: File {txt_path} not found.")
        except Exception as e:
            print(f"Error loading file: {e}")

def convert_float_to_fixed_point(input_array, total_bits=16, fractional_bits=0):
    # Calculate the range of values representable in the fixed-point format
    max_val = 2**(total_bits - 1) - 1
    min_val = -2**(total_bits - 1)

    # Scale factor to convert floating-point to fixed-point
    scale_factor = 2**fractional_bits

    # Clip input values to the range representable in the fixed-point format
    clipped_array = np.clip(input_array * scale_factor, min_val, max_val)

    # Round the clipped values and convert to 16-bit signed integers
    fixed_point_array = clipped_array.astype(np.int16)

    return fixed_point_array

def draw_spectrum(h, fs):
    plt.title("Spectrum")
    plt.plot(np.linspace(0,fs,len(h)), np.abs(np.fft.fft(h)))
    plt.xlim(0,fs/2)
    plt.show()

def write_gain_to_file(data,filename):
     with open(filename, "w") as file:
      for value in data:
      # Convert integer to binary string with leading zeros (16 bits)
       binary_data = format(value & 0x1F, '05b')  # Mask with 0xFFFF to get 16 bits
      # Write binary data to file as a string
       file.write(binary_data + "\n")
    
#main start here
with wave.open("input.wav") as wav_file:
    metadata=wav_file.getparams()
    print(metadata)
    frames=wav_file.readframes(metadata.nframes)

pcm_samples=np.frombuffer(frames,dtype='<h')
gaindata=[1,1,1,1,1,1,1,1]
gaindata_fxp=convert_float_to_fixed_point(gaindata,5,0)
write_gain_to_file(gaindata_fxp,"gain.txt")

output=equalizer(pcm_samples,gaindata)


with wave.open("output.wav",'w') as out:
   out.setnchannels(1)
   out.setsampwidth(2)
   out.setframerate(16000)
   out.setnframes(77065)
   out.writeframes(pcm_samples)
write_fixed_point_to_file(pcm_samples,"input.txt")
testoutput=convert_float_to_fixed_point(output)
draw_spectrum(testoutput,16000)
write_fixed_point_to_file(testoutput,"output.txt")

# this code below write output to output.wav
with wave.open("output.wav",'w') as out:
   out.setnchannels(1)
   out.setsampwidth(2)
   out.setframerate(16000)
   out.setnframes(77065)
   out.writeframes(testoutput)


