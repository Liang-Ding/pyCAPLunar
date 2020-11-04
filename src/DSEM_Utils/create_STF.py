# -------------------------------------------------------------------
# Create Source Time Function.
#
# Author: Liang Ding
# Email: myliang.ding@mail.utoronto.ca
# -------------------------------------------------------------------

from scipy import signal
import numpy as np


def create_STF_Butterworth(dt, highcut, order, wordN, data_length):
    '''
    * Create Source Time Function by using the Butterworth function.

    :param dt:      The sampling interval for the STF.
    :param highcut: The highest frequency.
    :param order:   The order of the Butterworth function.
    :param wordN:   The length of the frequency response.
    :param data_length: The length of the impulse and step response.

    :return: * freq, The frequency contains.
             * gain, The gain for each frequency contains.
             * t_arr, The time axis of the response.
             * impulse_response, The impulse response of Butterworth filter.
             * step_response, The step Response of Butterworth filter.
    '''

    # create butterworth filter
    fs = 1.0 / dt  # in Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    b, a = signal.butter(order, highcut / nyq)

    # frequency response of butterworth filter.
    w, h = signal.freqz(b, a, wordN)
    freq = (fs * 0.5 / np.pi) * w
    gain = abs(h)

    # impluse response of butterworth filter.
    impulse = np.zeros(data_length)
    impulse[int(data_length / 2)] = 1
    impulse_response = signal.filtfilt(b, a, impulse)

    # normalize
    impulse_response = impulse_response/np.max(impulse_response)


    # step response of butterworth filter
    step = np.ones(data_length)
    step_response = signal.lfilter(b, a, step)

    # time axis
    t_arr = dt * np.arange(data_length)

    return freq, gain, t_arr, impulse_response, step_response



def example_create_STF_butterworth():
    dt = 0.01  # sampling rate in second
    highcut = 0.5  # High cut in Hz, = 2s
    wordN = 1024
    data_length = 1000
    order = 3
    freq, gain, t_arr, impulse_response, step_response = create_STF_Butterworth(dt, highcut, order, wordN, data_length)

    import matplotlib.pyplot as plt

    # plot the frequency response of butterworth filter
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes[0, 0].semilogx(freq, gain)
    axes[0, 0].set_title('Frequency response')
    axes[0, 0].set_xlabel('Frequency [Hz]')
    axes[0, 0].set_ylabel('Gain')
    # axes[0, 0].set_xlim([0, 5])
    axes[0, 0].margins(0, 0.1)
    axes[0, 0].grid(which='both', axis='both')
    axes[0, 0].axvline(highcut, color='green')  # cutoff frequency

    axes[0, 1].axis('off')

    # plot the Impulse response of butterworth filter
    t = dt * np.arange(len(impulse_response))
    axes[1, 0].plot(t, impulse_response)
    axes[1, 0].set_title('Impulse response')
    axes[1, 0].set_xlabel('Time [second]')
    axes[1, 0].set_ylabel('Amplitude')

    # plot the step response of Buttwerworth filter
    # axes[1, 1].plot(t, step_response)
    axes[1, 1].plot(t, np.cumsum(impulse_response))
    axes[1, 1].set_title('Step Response (STF for SGT Computation)')
    axes[1, 1].set_xlabel('Time [second]')
    # axes[1, 1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    example_create_STF_butterworth()
    
    
