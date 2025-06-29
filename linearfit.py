'''SUMMARY OF THIS FILE:
TAKES IN CSV DATA TO FIND THE FIT BETWEEN THE DECAY RATE AND THE FREQUENCY OF THE ONSET'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
file_path = 'decay_data.csv'  # Replace with your file path

# === FUNCTION: MIDI to Frequency ===
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))

# === LOAD CSV (no headers) ===
try:
    df = pd.read_csv(file_path, header=None, names=['MIDI', 'DecayRate'])
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Make sure the file exists.")
    exit()

# === CONVERT MIDI TO FREQUENCY ===
df['Frequency'] = midi_to_freq(df['MIDI'])
df['log10_freq'] = np.log10(df['Frequency'])

# === FIT LOG-LINEAR MODEL: DecayRate = A + B * log10(f) ===
x = df['log10_freq'].values
y = df['DecayRate'].values
B, A = np.polyfit(x, y, 1)

# === GENERATE FIT CURVE ===
x_fit = np.linspace(min(x), max(x), 100)
y_fit = A + B * x_fit

# === PLOT ===
plt.figure(figsize=(10, 6))
plt.scatter(df['Frequency'], df['DecayRate'], color='blue', label='Digitized Data')
plt.plot(10 ** x_fit, y_fit, color='red', label=f'Fit: Decay = {A:.2f} + {B:.2f}·log10(f)')
plt.xscale('log')
plt.xlabel('Frequency (Hz) [log scale]')
plt.ylabel('Decay Rate (dB/s)')
plt.title('Fitted Decay Rate Model from Digitized Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === OUTPUT MODEL ===
print(f"\n✅ Fitted model:")
print(f"DecayRate_dB/s = {A:.2f} + {B:.2f}·log10(frequency in Hz)")