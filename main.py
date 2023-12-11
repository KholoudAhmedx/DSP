import math
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from comparesignals import SignalSamplesAreEqual
from QuanTest1 import QuantizationTest1
from QuanTest2 import QuantizationTest2
from ConvTest import ConvTest
from Shift_Fold_Signal import Shift_Fold_Signal
from tkinter import ttk
from signalcompare import *
from comparesignal2 import *
from DerivativeSignal import DerivativeSignal
from ConvTest import ConvTest
# create Window
root = Tk()
root.title('Task1')
root.geometry("500x435")
# Label(root, background='#A9A9A9').pack(expand=True, fill='both')
Y1 = []
X1 = []
Y2 = []
X2 = []
samples = []
time = []

def open_file():
    FilePath = filedialog.askopenfilename()
    File = open(FilePath, 'r')
    signal_values = File.readlines()
    File.close()
    samples.clear()
    time.clear()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[1]
    signal_values.pop(0)
    N = signal_values[2]
    signal_values.pop(0)
    for line in signal_values:
        values = line.strip().split()
        time.append(float(values[0]))
        samples.append(float(values[1]))

OpenButton = Button(root, text="Open File", bg='#DDA0DD', width=20, height=2, command=open_file)
OpenButton.pack()


def quantize_signal(input_signal, levels):
    quantized_signal = []
    quantization_error = []
    intervals_index = []
    encoded_intervals = []
    intervals = []
    mid_points = []
    num_of_bits = int(math.log2(levels))
    min_amp = np.min(input_signal)
    max_amp = np.max(input_signal)
    delta_a = (max_amp - min_amp) / levels
    Z = round(min_amp + delta_a, 2)
    intervals.append(min_amp)
    mid_points.append(round((min_amp + Z) / 2, 2))
    for i in range(levels - 1):
        new_val = round(Z + delta_a, 2)
        intervals.append(Z)
        mid_points.append(round((Z + new_val) / 2, 2))
        Z = new_val
    intervals.append(max_amp)

    for i, sample in enumerate(input_signal):
        for j in range(len(intervals) - 1):
            if intervals[j] <= sample <= intervals[j + 1]:
                quantized_signal.append(round(mid_points[j], 2))
                intervals_index.append(j + 1)
                encoded_intervals.append(format(j, f'0{num_of_bits}b'))
                break
    quantization_error = np.subtract(quantized_signal, input_signal)
    quantization_error = [round(error, 2) for error in
                          quantization_error]  # Round quantization error values to two decimal places
    return intervals_index, encoded_intervals, quantized_signal, quantization_error


def Quantization_window():
    global bits_level_entry
    global tree
    global columns
    quantization = tk.Toplevel()
    quantization.geometry('400x400')
    quantization.title('Quantization')

    bits_level_label = tk.Label(quantization, text='Number of bits/ Number of levels')
    bits_level_label.pack()
    bits_level_entry = tk.Entry(quantization)
    bits_level_entry.pack()

    bits_button = tk.Button(quantization, text='Bits', command=need_bits)
    bits_button.pack()

    levels_button = tk.Button(quantization, text='Levels', command=need_levels)
    levels_button.pack()

    columns = ('Input Signal', "Interval Index", "Encoded Signals", "Quantized Signal", "Quantization Error")
    tree = ttk.Treeview(quantization, columns=columns, show='headings')

    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)

    tree.pack()

def need_bits():
    global bits_level_entry
    global tree
    global columns
    num_of_bits = int(bits_level_entry.get())

    num_levels = 2 ** num_of_bits
    inputSignal = [0.387, 0.430, 0.478, 0.531, 0.590, 0.6561, 0.729, 0.81, 0.9, 1, 0.2]
    # inputSignal = [-1.22, 1.5, 3.24, 3.94, 2.20, -1.1, -2.26, -1.88, -1.2]
    intervals_index, encoded_intervals, quantized_signal, quantization_error = quantize_signal(inputSignal, num_levels)

    # Clear existing data in the treeview
    tree.delete(*tree.get_children())

    # Insert the data into the treeview
    for i in range(len(encoded_intervals)):
        values = []
        for col in columns:
            if col == "Input Signal":
                values.append(inputSignal[i])
            elif col == "Interval Index":
                values.append("")  # Leave empty for this column
            elif col == "Encoded Signals":
                values.append(encoded_intervals[i])
            elif col == "Quantized Signal":
                values.append(quantized_signal[i])
            else:
                values.append("")  # Leave empty for other columns
        tree.insert("", "end", values=values)

    print("Test Case 1:")
    QuantizationTest1('Files/Quan1_Out.txt', encoded_intervals, quantized_signal)
    # print("Encoded Intervals:", encoded_intervals)
    # print("Quantized Signal:", quantized_signal)
    # print()


def need_levels():
    global bits_level_entry
    global columns
    num_levels = int(bits_level_entry.get())
    inputSignal = [-1.22, 1.5, 3.24, 3.94, 2.20, -1.1, -2.26, -1.88, -1.2]
    intervals_index, encoded_intervals, quantized_signal, quantization_error = quantize_signal(inputSignal, num_levels)

    # Clear existing data in the treeview
    tree.delete(*tree.get_children())

    # Insert the data into the treeview
    for i in range(len(intervals_index)):
        values = []
        for col in columns:
            if col == "Input Signal":
                values.append(inputSignal[i])
            elif col == "Interval Index":
                values.append(intervals_index[i])
            elif col == "Encoded Signals":
                values.append(encoded_intervals[i])
            elif col == "Quantized Signal":
                values.append(quantized_signal[i])
            elif col == "Quantization Error":
                values.append(quantization_error[i])
            else:
                values.append("")  # Leave empty for other columns
        tree.insert("", "end", values=values)

    print("Test Case 2:")
    QuantizationTest2('Files/Quan2_Out.txt', intervals_index, encoded_intervals, quantized_signal, quantization_error)
    # print("intervals_index", intervals_index)
    # print("Encoded Intervals:", encoded_intervals)
    # print("Quantized Signal:", quantized_signal)
    # print("quantization_error", quantization_error)
    print()

def show_continuous():
    plt.figure(figsize=(10, 5))
    plt.plot(time, samples)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Continuous Representation of the Signal')
    plt.grid(True)
    plt.show()

def show_discrete():
    # Plot the discrete representation
    plt.figure(figsize=(10, 5))
    plt.stem(samples)
    plt.xlabel('Index SampleAmp')
    plt.ylabel('Amplitude')
    plt.title('Discrete Representation of the Signal')
    plt.grid(True)
    plt.show()


def read_signal1():
    with open('Files/Input_conv_Sig1.txt', 'r') as file:
        signal_values = file.readlines()
    Y1.clear()
    X1.clear()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[1]
    signal_values.pop(0)
    N = signal_values[2]
    signal_values.pop(0)
    # Parse the signal samples
    for line in signal_values:
        values = line.strip().split()
        X1.append(int(values[0]))
        Y1.append(int(values[1]))
    return X1, Y1

def read_signal2():
    with open('Files/Input_conv_Sig2.txt', 'r') as file:
        signal_values = file.readlines()
    Y2.clear()
    X2.clear()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[1]
    signal_values.pop(0)
    N = signal_values[2]
    signal_values.pop(0)
    # Parse the signal samples
    for line in signal_values:
        values = line.strip().split()
        X2.append(int(values[0]))
        Y2.append(int(values[1]))
    return X2, Y2, N

def add():
    x1, y1 = read_signal1()
    x2, y2 = read_signal2()
    # result_addition = [y1 + y2 for y1, y2 in zip(y1, y2)]

    # If lengths are not eqaul
    if len(y1) < len(y2):
        for i in range(len(y2) - len(y1)):
            y1.append(0)
    elif len(y2) < len(y1):
        for i in range(len(y1) - len(y2)):
            y2.append(0)

    addition_list = []
    for i in range(len(y1)):
        result_additon = y1[i] + y2[i]
        addition_list.append(result_additon)

    # Test
    SignalSamplesAreEqual('Files/Signal1+signal2.txt', time, addition_list)

    # fig = plt.figure(figsize=(8, 6))
    # gs = gridspec.GridSpec(2, 2, figure=fig)
    # s1 = fig.add_subplot(gs[0, 0])
    # s1.set_title("Signal1")
    # s1.plot(x1, y1, 'b')
    # s2 = fig.add_subplot(gs[0, 1])
    # s2.set_title("Signal2")
    # s2.plot(x2, y2, 'b')
    # s1_add_s2 = fig.add_subplot(gs[1, 0])
    # s1_add_s2.set_title("Addition")
    # s1_add_s2.plot(x1, addition_list, 'r')
    # plt.tight_layout()
    # plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label='Original Signal 1')
    plt.plot(x2, y2, label='Original Signal 2')

    plt.plot(x1, addition_list, label='added signal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Signal addition')
    plt.legend()
    plt.grid(True)
    plt.show()

def sub():
    x1, y1 = read_signal1()
    x2, y2 = read_signal2()
    # result_subtraction = [abs(y1 - y2) for y1, y2 in zip(y1, y2)]

    if len(y1) < len(y2):
        for i in range(len(y2) - len(y1)):
            y1.append(0)
    elif len(y2) < len(y1):
        for i in range(len(y1) - len(y2)):
            y2.append(0)

    subtraction_list = []
    for i in range(len(y1)):
        result_subtraction = abs(y1[i] - y2[i])
        subtraction_list.append(result_subtraction)

    # Test
    SignalSamplesAreEqual('signal1-signal2.txt', time, subtraction_list)

    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label='Original Signal 1')
    plt.plot(x2, y2, label='Original Signal 2')

    plt.plot(x1, subtraction_list, label='subtracted Signal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Signal subtraction')
    plt.legend()
    plt.grid(True)
    plt.show()

def mutliply_window():
    global constant_entry

    signal_generation = tk.Toplevel()
    signal_generation.title('Multiply')
    signal_generation.geometry('400x350')

    constant_label = tk.Label(signal_generation, text='constant value:')
    constant_label.pack()
    constant_entry = tk.Entry(signal_generation)
    constant_entry.pack()

    shifting_button = tk.Button(signal_generation, text='Read', command=multiply)
    shifting_button.pack()

def multiply():
    global constant_entry
    x1, y1 = read_signal1()

    constant_val = int(constant_entry.get())

    result_multiplication = [y * constant_val for y in y1]

    SignalSamplesAreEqual('Files/MultiplySignalByConstant-Signal1 - by 5.txt', time, result_multiplication)

    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label='Original Signal')
    plt.plot(x1, result_multiplication, label='multiplied Signal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Signal multiplying')
    plt.legend()
    plt.grid(True)
    plt.show()

def normalize():
    global max_entry, min_entry

    x1, y1 = read_signal1()  # read one signal

    # Read entries
    max = float(max_entry.get())
    min = float(min_entry.get())

    # max and min values
    y_min = np.min(y1, axis=0)
    y_max = np.max(y1, axis=0)

    # we want to noramlize y vlaues
    y1_std = (y1 - y_min) / (y_max - y_min)
    y1_scaled = y1_std * (max - min) + min

    SignalSamplesAreEqual('Files/normalize of signal 1 -- output.txt', time, y1_scaled)

    # Only printing the normalized signal
    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1_scaled)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('normalized function ')
    plt.grid(True)
    plt.show()

def normalize_window():
    # need to read the min and max form the user
    global max_entry, min_entry

    signal_generation = tk.Toplevel()
    signal_generation.title('Normalize signal')
    signal_generation.geometry('400x350')

    max_label = tk.Label(signal_generation, text='max value:')
    max_label.pack()
    max_entry = tk.Entry(signal_generation)
    max_entry.pack()

    min_label = tk.Label(signal_generation, text='min value: ')
    min_label.pack()
    min_entry = tk.Entry(signal_generation)
    min_entry.pack()

    normalize_button = tk.Button(signal_generation, text='Read', command=normalize)
    normalize_button.pack()

def accumulate():
    x1, y1 = read_signal1()

    sum = 0
    samples = []
    for i in range(len(y1)):
        sum += y1[i]
        samples.append(sum)

    SignalSamplesAreEqual('Files/output accumulation for signal1.txt', time, samples)

def show_sin():
    global amplitude_entry, sample_freq_entry, analog_freq_entry, phaseshift_entry

    amplitude = float(amplitude_entry.get())
    sample_freq = int(sample_freq_entry.get())
    analog_freq = int(analog_freq_entry.get())
    phaseshift = float(phaseshift_entry.get())
    No_samples_per_cycle = sample_freq / analog_freq

    if sample_freq > 0:

        t = np.arange(0, 1, 1 / sample_freq)
        samples = amplitude * np.sin(2 * np.pi * analog_freq * t + phaseshift)
    # sample_indices = amplitude * np.sin(2*np.pi*(analog_freq/sample_freq) * No_samples_per_cycle + phaseshift)
    else:
        t = np.arange(0, 1, 1 / (2 * analog_freq))

        # t = np.linspace(0, 2*np.pi, 1000)
        samples = amplitude * np.sin(2 * np.pi * analog_freq * t + phaseshift)
        # sample_indices = amplitude * np.sin(2*np.pi*(analog_freq/sample_freq) * No_samples_per_cycle + phaseshift)

    SignalSamplesAreEqual('Files/SinOutput.txt',samples)

    # Plot the signal and number of samples
    plt.figure(figsize=(10, 5))
    plt.plot(t, samples, label='Sin function')
    # plt.plot(t, sample_indices, '.', label='Number of Samples')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Sin Function')
    plt.grid(True)
    plt.legend()
    plt.show()

def show_cos():
    global amplitude_entry, sample_freq_entry, analog_freq_entry, phaseshift_entry

    amplitude = float(amplitude_entry.get())
    sample_freq = int(sample_freq_entry.get())
    analog_freq = int(analog_freq_entry.get())
    phaseshift = float(phaseshift_entry.get())

    if sample_freq > 0:

        # t = np.linspace(0, 1/sample_freq, N)
        t = np.arange(0, 1, 1 / sample_freq)
        samples = amplitude * np.cos(2 * np.pi * analog_freq * t + phaseshift)
    else:
        # t = np.linspace(0, 1/ (2* analog_freq), N)
        t = np.arange(0, 1, 1 / (2 * analog_freq))

        # t = np.linspace(0, 2*np.pi, 1000)
        samples = amplitude * np.cos(2 * np.pi * analog_freq * t + phaseshift)

    # Calculate number of samples per cycle
    samples_per_cycle = int(sample_freq / analog_freq)

    SignalSamplesAreEqual('Files/CosOutput.txt', samples)
    plt.figure(figsize=(10, 5))
    plt.plot(t, samples)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Cos Function')
    plt.grid(True)

    plt.show()

def squaring():
    x1, y1 = read_signal1()
    result_multiplication = [(x ** 2) for x in y1]

    SignalSamplesAreEqual('Files/Output squaring signal 1.txt', time, result_multiplication)

    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label='Original Signal')
    plt.plot(x1, result_multiplication, label='squarred Signal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Signal squaring')
    plt.legend()
    plt.grid(True)
    plt.show()

def shifting():
    global constant_entry

    # Read signal
    x1, y1 = read_signal1()

    constant_val = int(constant_entry.get())
    shifted_values = []
    new_val = 0
    for i in range(len(x1)):
        new_val = (x1[i] + constant_val)
        shifted_values.append(new_val)
    print(shifted_values)
    SignalSamplesAreEqual('Files/output shifting by minus 500.txt', time, shifted_values)

    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, label='Original Signal')
    plt.plot(shifted_values, y1, label='Shifted Signal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Signal Shifting')
    plt.legend()
    plt.grid(True)
    plt.show()


def Modify_values():
    global Amp_entry, Phaseshift_entry, Index_entry

    # Read values from the user 
    amp_value = Amp_entry.get()
    phaseshift_value = Phaseshift_entry.get()
    index_value = Index_entry.get()
    return amp_value, phaseshift_value, index_value


def DFT():
    # Read the modifed values form the user
    amp_value, phaseshift_value, index_value = Modify_values()
    global fs_entry

    fs = fs_entry.get()
    harmonics, samples, N = read_signal2()
    X_k = 0

    X_k_list = []
    index_list = []
    Amp_list = []
    Phase_shift_list = []

    # Apply DFT
    index = 0  # to represend the x axis in the graph (Frequency)
    df = (2 * np.pi * float(fs)) / len(samples)  # fundamental frequency
    for i in range(len(harmonics)):
        index += df
        index_list.append(index)

        for j in range(len(samples)):
            real_value = (2 * 180 * i * j) / len(samples)
            imaginary = -1j
            e = math.cos(math.radians(real_value)) + imaginary * math.sin(math.radians(real_value))
            X_k += samples[j] * e

        X_k_list.append(X_k)
        X_k = 0  # Reset sum for each iteration

    # print(index_list)
    # print(X_k_list)

    # Loop over the list to compute the amplitude and the phase shift
    for complex_number in X_k_list:
        # Extract coefficients
        real_part = complex_number.real
        imaginary_part = complex_number.imag

        Amp = np.sqrt(real_part ** 2 + imaginary_part ** 2)
        Amp_list.append(Amp)

        phaseshift = math.atan2(imaginary_part, real_part)
        Phase_shift_list.append(phaseshift)

    # print("Amplitude:" , Amp_list)
    # print('\n')
    # print("Phaseshift: ", Phase_shift_list)

    # Modify signal amplitude & shift
    if amp_value != "":
        if index_value != "":
            Amp_list[int(index_value)] = amp_value

    if phaseshift_value != "":
        if index_value != "":
            Phase_shift_list[int(index_value)] = phaseshift_value

            # Write values to files
    f = open("dft_output_signal", "w", newline="")
    # f.write
    f.write(str(0) + '\n')
    f.write(str(1) + '\n')  # Time domain
    f.write(str(len(samples)) + '\n')
    for i in range(len(samples)):
        f.write(str(Amp_list[i]) + ' ')
        f.write(str(Phase_shift_list[i]) + '\n')
    f.close()

    # Plot signals 
    plot1 = plt.subplot2grid((1, 2), (0, 0))  # plot frequency & amplitude
    plot2 = plt.subplot2grid((1, 2), (0, 1))  # plot frequency & phaseshift

    # Frequency & Amplitude
    plot1.plot(index_list, Amp_list)
    plot1.set_xlabel("Frequency")
    plot1.set_ylabel("Amplitude")

    # frequency & phaseshift
    plot2.plot(index_list, Phase_shift_list)
    plot2.set_xlabel("Frequency")
    plot2.set_ylabel("Phaseshift")

    plt.show()


def IDFT_algorithm():
    # read signals from File
    with open('Files/Input_Signal_IDFT_A,Phase.txt', 'r') as file:
        signal_values = file.readlines()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[0]
    signal_values.pop(0)
    N = signal_values[0]
    signal_values.pop(0)
    # Parse the signal samples
    freq_inputs = []
    for line in signal_values:
        # Convert Amplitude & phase_shift to imaginary numbers & real number
        amplitude, phase_shift = map(lambda x: float(x.rstrip('f')), line.strip().split(','))
        real = amplitude * np.cos(phase_shift)
        imaginary = amplitude * np.sin(phase_shift)
        freq_inputs.append(real + 1j * imaginary)
    print("freq input", freq_inputs)
    # Equation of IDFT
    size = len(freq_inputs)
    result = []
    for k in range(size):
        exp = 0
        j = 0
        summation = 0
        for n in range(size):
            power = 2 * (1j * math.pi * k * n) / size
            # using rule of e power j* theta = cos(theta) + j*sin(theta)
            summation += freq_inputs[n] * np.exp(power)

        summation = complex(round(summation.real, 4), round(summation.imag, 4))
        summation = 1 / size * summation
        summation = int(np.real(summation))
        result.append(summation)

    plot1 = plt.subplot2grid((1, 1), (0, 0))  # plot amplitude

    # Frequency & Amplitude
    plot1.plot(range(size), result)
    plot1.set_xlabel("Time")
    plot1.set_ylabel("Amplitude")
    plt.show()


def remove_dc():
    # read signals from File
    X1.clear()
    Y1.clear()
    with open('Files/DC_component_input.txt', 'r') as file:
        signal_values = file.readlines()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[0]
    signal_values.pop(0)
    N = int(signal_values[0])
    signal_values.pop(0)
    # Parse the signal samples
    sum_all_value = 0
    for line in signal_values:
        values = line.strip().split()
        X1.append(float(values[0]))
        sum_all_value += float(values[1])
        Y1.append(float(values[1]))
    average = sum_all_value / N
    new_Y1 = [round(x - average, 3) for x in Y1]
    print("Values After Removing Dc \n", new_Y1)

    # TEST 
    SignalSamplesAreEqual('Files/DC_component_output.txt', new_Y1)

    # plotting
    plot1 = plt.subplot2grid((1, 2), (0, 0))  # plot Before Removing DC
    plot2 = plt.subplot2grid((1, 2), (0, 1))  # plot After Removing DC

    # Before
    plot1.plot(range(N), Y1)
    plot1.set_xlabel("Time")
    plot1.set_ylabel("Amplitude")
    plot1.set_title("Before Removing Dc")

    # After
    plot2.plot(range(N), new_Y1)
    plot2.set_xlabel("Time")
    plot2.set_ylabel("Amplitude")
    plot2.set_title("After Removing Dc")

    plt.show()


def computing_DCT():
    global DCT_entry
    X1.clear()
    Y1.clear()
    # read signals from File
    with open('Files/DCT_input.txt', 'r') as file:
        signal_values = file.readlines()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[0]
    signal_values.pop(0)
    size_input = int(signal_values[0])
    signal_values.pop(0)
    # Parse the signal samples

    for line in signal_values:
        values = line.strip().split()
        X1.append(float(values[0]))
        Y1.append(float(values[1]))
    y_k_list = []
    for k in range(size_input):
        summation = 0
        for n in range(size_input):
            summation += Y1[n] * math.cos(np.pi / (4 * size_input) * (2 * n - 1) * (2 * k - 1))
        summation = summation * math.sqrt(2 / size_input)
        y_k_list.append(round(summation, 5))
    plot1 = plt.subplot2grid((1, 1), (0, 0))
    plot1.stem(X1, y_k_list)
    plot1.set_xlabel("Time")
    plot1.set_ylabel("Y(K)")
    plt.show()
    print(y_k_list)

    # TEST 
    SignalSamplesAreEqual('Files/DCT_output.txt', y_k_list)

    # Save to file 
    NumberOfCOeff = int(DCT_entry.get())
    if NumberOfCOeff != " " and NumberOfCOeff <= len(y_k_list):
        f = open("DCT_Coefficients", "w", newline="")
        for i in range(NumberOfCOeff):
            f.write(str(y_k_list[i]) + ' ')
            f.write('\n')
        f.close()

def Delaying_or_advancing():
    global K_entry
    K_value = int(K_entry.get())

    X1.clear()
    Y1.clear()
    # read signals from File
    with open('Files/Input_conv_Sig1.txt', 'r') as file:
        signal_values = file.readlines()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[0]
    signal_values.pop(0)
    size_input = int(signal_values[0])
    signal_values.pop(0)
    indices = [-2, -1, 0, 1, 2, 3, 4, 5, 6]
    Y_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1]
    new_indices = [x + K_value for x in indices]
    print("Result of Delaying_or_advancing \n")
    print("indices :- ", indices, "\n")
    print("new_indices :- ", new_indices, "\n")
    print("Y_Samples :- ", Y_samples, "\n")

    ConvTest(indices, Y_samples)


def Smoothing():
    global Window_size_entry
    Window_size_value = int(Window_size_entry.get())
    X1.clear()
    Y1.clear()
    # read signals from File
    with open('Files/Signal2.txt', 'r') as file:
        signal_values = file.readlines()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[0]
    signal_values.pop(0)
    size_input = int(signal_values[0])
    signal_values.pop(0)
    # Parse the signal samples
    Y_smoothing = []
    for line in signal_values:
        values = line.strip().split()
        X1.append(int(values[0]))
        Y1.append(int(values[1]))
    for y in range(len(Y1) - Window_size_value + 1):
        summation = 0
        for i in range(Window_size_value):
            summation += Y1[y + i]
        # average = sum(Y1[y:y + Window_size_value]) / Window_size_value
        average = summation / Window_size_value
        Y_smoothing.append(average)
    print("Result of Smoothing_Signals \n")
    print("indices :- ", X1, "\n")
    print("samples :- ", Y1, "\n")
    print("Smoothing_samples :- ", Y_smoothing, "\n")
    SignalSamplesAreEqual('Files/Signal2.txt', Y1)


def Folding():
    X1.clear()
    Y1.clear()
    # read signals from File
    with open('Files/input_fold.txt', 'r') as file:
        signal_values = file.readlines()
    SignalType = signal_values[0]
    signal_values.pop(0)
    IsPeriodic = signal_values[0]
    signal_values.pop(0)
    size_input = int(signal_values[0])
    signal_values.pop(0)
    # Parse the signal samples
    Y_old = []
    for line in signal_values:
        values = line.strip().split()
        X1.append(int(values[0]))
        Y1.append(int(values[1]))
        Y_old.append(int(values[1]))
    Y1.reverse()
    print("Result of folding_signal \n")
    print("indices :- ", X1, "\n")
    print("samples :- ", Y_old, "\n")
    print("Folding_samples :- ", Y1, "\n")
    return X1, Y_old, Y1


def Delaying_or_advancing_folding_signal():
    global K_entry
    K_value = int(K_entry.get())
    print("Result of Delaying_or_advancing_folding_signal \n")
    indices, Y_old, Y_samples = Folding()
    new_indices = [x + K_value for x in indices]
    print("Result of Delaying_or_advancing")
    print("new_indices :- ", new_indices, "\n")
    Shift_Fold_Signal('Files/input_fold.txt', indices, Y_old)


def Convolve_signal():

    input_indices, input_signal_list = read_signal1()
    filter_indices, filter_list, N = read_signal2()
    
    excpected_indeces = []
    first_index = input_indices[0] + filter_indices[0]
    last_index = input_indices[-1] + filter_indices[-1]

    # Indices 
    for i in range(first_index, last_index + 1):
        excpected_indeces.append(i)


    N = len((excpected_indeces)) # (form first_index to last_index)

    # Check boundaries 
    if (len(input_signal_list) < len(filter_list)):
        for i in range(N - len(input_signal_list)):
            input_signal_list.append(0)
            input_indices.append(0) # For the sake of plotting 

    elif (len(input_signal_list) > len(filter_list)):
        for i in range(N - len(filter_list)):
            filter_list.append(0)
            filter_indices.append(0) 
    
    excpected_samples = np.zeros(N)

    for n in range(len(input_signal_list)):
        for k in range(len(filter_list)):
            if n >= k:
                excpected_samples[n] = excpected_samples[n] + input_signal_list[n - k]*filter_list[k]

    # print('Expected Indices: ',excpected_indeces)
    # print('Expected Samples: ', list(excpected_samples))
    ConvTest(excpected_indeces, excpected_samples)

    # plotting 
    fig, axs = plt.subplots(3)
    fig.suptitle('Input & Convolved signals')
    axs[0].plot(input_indices, input_signal_list)
    axs[0].set_title( 'Input Signal 1')
    axs[1].plot(filter_indices, filter_list)
    axs[1].set_title('Input Signal 2')
    axs[2].plot(excpected_indeces, excpected_samples)
    axs[2].set_title('Convolved Signal')
    plt.show()

############# WINDOWS ###############################

def shifting_window():
    global constant_entry

    signal_generation = tk.Toplevel()
    signal_generation.title('Shifting')
    signal_generation.geometry('400x350')

    constant_label = tk.Label(signal_generation, text='constant value:')
    constant_label.pack()
    constant_entry = tk.Entry(signal_generation)
    constant_entry.pack()

    shifting_button = tk.Button(signal_generation, text='Read', command=shifting)
    shifting_button.pack()

def signal_generation_window():
    global amplitude_entry, sample_freq_entry, analog_freq_entry, phaseshift_entry

    signal_generation = tk.Toplevel()
    signal_generation.title('Signal Generation')
    signal_generation.geometry('400x350')

    amplitude_label = tk.Label(signal_generation, text='Amplitude:')
    amplitude_label.pack()
    amplitude_entry = tk.Entry(signal_generation)
    amplitude_entry.pack()

    sample_freq_label = tk.Label(signal_generation, text='Sample Frequency:')
    sample_freq_label.pack()
    sample_freq_entry = tk.Entry(signal_generation)
    sample_freq_entry.pack()

    analog_freq_label = tk.Label(signal_generation, text='Analog Frequency:')
    analog_freq_label.pack()
    analog_freq_entry = tk.Entry(signal_generation)
    analog_freq_entry.pack()

    phaseshift_label = tk.Label(signal_generation, text='Phase Shift:')
    phaseshift_label.pack()
    phaseshift_entry = tk.Entry(signal_generation)
    phaseshift_entry.pack()

    sin_button = tk.Button(signal_generation, text='Sin', command=show_sin)
    sin_button.pack()

    cos_button = tk.Button(signal_generation, text='Cos', command=show_cos)
    cos_button.pack()

def DCT_window():

    global DCT_entry
    DCT = tk.Toplevel()
    DCT.title('DCT component')
    DCT.geometry('400x350')

    DCT_label = tk.Label(DCT, text='Number of coefficients:')
    DCT_label.pack()

    DCT_entry = Entry(DCT)
    DCT_entry.pack()

    DCT_button = tk.Button(DCT, text='Save to file', command=computing_DCT)
    DCT_button.pack()

def frequency_domain():
    global Amp_entry, Phaseshift_entry, Index_entry, fs_entry
    frequency_domain = tk.Toplevel()
    frequency_domain.title("Frequency Domain")
    frequency_domain.geometry("400x500")

    Amp_label = tk.Label(frequency_domain, text="Amplitude")
    Amp_label.pack()

    Amp_entry = tk.Entry(frequency_domain)
    Amp_entry.pack()

    Phaseshift_label = tk.Label(frequency_domain, text="Phase shift")
    Phaseshift_label.pack()

    Phaseshift_entry = tk.Entry(frequency_domain)
    Phaseshift_entry.pack()

    Index_label = tk.Label(frequency_domain, text="Index")
    Index_label.pack()

    Index_entry = tk.Entry(frequency_domain)
    Index_entry.pack()

    fs_label = tk.Label(frequency_domain, text="Sample Frequency")
    fs_label.pack()

    fs_entry = tk.Entry(frequency_domain)
    fs_entry.pack()

    # Removing Dc
    Read_button = Button(frequency_domain, text="Removing Dc", command=remove_dc)
    Read_button.pack()

    # Button to read the values
    Read_button = Button(frequency_domain, text="Read", command=Modify_values)
    Read_button.pack()

    # Apply DFT
    Read_button = Button(frequency_domain, text="DFT", command=DFT)
    Read_button.pack()

def time_domain():
    global K_entry, Window_size_entry
    time_domain = tk.Toplevel()
    time_domain.title("Time Domain")
    time_domain.geometry("400x500")
    # Window_size of smoothing
    Window_size_label = tk.Label(time_domain, text="Window_size")
    Window_size_label.pack()
    Window_size_entry = tk.Entry(time_domain)
    Window_size_entry.pack()
    # Apply Smoothing
    Smoothing_button = Button(time_domain, text="Smoothing", command=Smoothing)
    Smoothing_button.pack()
    # Apply Sharpening
    Sharpening_button = Button(time_domain, text="Sharpening", command=DerivativeSignal)
    Sharpening_button.pack()
    # K of Delaying_or_advancing
    K_label = tk.Label(time_domain, text="K")
    K_label.pack()
    K_entry = tk.Entry(time_domain)
    K_entry.pack()
    # Apply Delaying_or_advancing
    Delaying_or_advancing_button = Button(time_domain, text="Delaying_or_advancing",
                                          command=Delaying_or_advancing)
    Delaying_or_advancing_button.pack()

    # Apply Delaying_or_advancing_folding_signal
    Delaying_or_advancing_folding_signal_button = Button(time_domain, text="Delaying_or_advancing_folding_signal",
                                                         command=Delaying_or_advancing_folding_signal)
    Delaying_or_advancing_folding_signal_button.pack()
    # Apply Folding
    folding_button = Button(time_domain, text="Folding", command=Folding)
    folding_button.pack()

    # Time domain
    freq_to_time = Button(time_domain, text="IDFT", command=IDFT_algorithm)
    freq_to_time.pack()

my_menu = Menu(root)
root.config(menu=my_menu)
file_menu = Menu(my_menu)
my_menu.add_cascade(label="Arithmetic Operations", menu=file_menu)
file_menu.add_command(label="Add", command=add)
file_menu.add_command(label="Sub", command=sub)
file_menu.add_command(label="Multiplication", command=mutliply_window)
file_menu.add_command(label="Squaring", command=squaring)
file_menu.add_command(label="Shifting", command=shifting_window)
file_menu.add_command(label="Normalization", command=normalize_window)
file_menu.add_command(label="Accumalation", command=accumulate)

ContinuousButton = Button(root, text='Continuous', bg='#DDA0DD', width=20, height=2, command=show_continuous)
ContinuousButton.pack()

DiscreteButton = Button(root, text='Discrete', bg='#DDA0DD', width=20, height=2, command=show_discrete)
DiscreteButton.pack()

# Signal generation button
SignalGereration = Button(root, text='Signal Generation', bg='#DDA0DD', width=20, height=2,
                          command=signal_generation_window)
SignalGereration.pack()

# Quantization button
Quantization = Button(root, text='Quantization', bg='#DDA0DD', width=20, height=2,
                      command=Quantization_window)
Quantization.pack()

# Frequency domain
Frequency_domain = Button(root, text=" Frequency domain", bg='#DDA0DD', width=20, height=2, command=frequency_domain)
Frequency_domain.pack()

# Time domain
time_domain = Button(root, text="Time domain", bg='#DDA0DD', width=20, height=2, command=time_domain)
time_domain.pack()

# DCT Component 
Compute_DCT = Button(root, text='Compute DCT', bg='#DDA0DD', width=20, height=2, command=DCT_window)
Compute_DCT.pack()

# Convolution 
Convolution = Button(root, text="Convolution", bg='#DDA0DD', width=20, height=2, command = Convolve_signal)
Convolution.pack()
root.mainloop()
