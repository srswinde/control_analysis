"""Main module."""
from struct import pack, unpack
from collections import namedtuple
import pandas as pd
from io import StringIO
from enum import Enum, auto
from scipy import fftpack, stats, signal
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import numpy as np
from abc import abstractmethod
import math
import datetime
from copy import deepcopy
from serial import Serial
from pathlib import Path
_wfmpre = namedtuple("wfmpre", ['byt_nr',
                                'bit_nr',
                                'encdg',
                                'bn_fmt',
                                'byt_or',
                                'nr_pt',
                                'wfid',
                                'pt_fmt',
                                'xincr',
                                'pt_off',
                                'xzero',
                                'xunit',
                                'ymult',
                                'yzero',
                                'yoff',
                                'yunit',
                                ])
_pre = _wfmpre(
    byt_nr='Preamble byte width of waveform points',
    bit_nr='Preamble bit width of waveform points',
    encdg='the preamble encoding method',
    bn_fmt='preamble binary encoding type',
    byt_or='preamble byte order of waveform points',
    nr_pt='the number of points in the curve transfer to/from the oscilloscope',
    wfid='Query curve identifier',
    pt_fmt='format of curve points',
    xincr='the horizontal sampling interval',
    pt_off='Query trigger offset',
    xzero='returns the time of first point in waveform',
    xunit='horizontal units',
    ymult='horizontal units',
    yzero='offset voltage',
    yoff='vertical position',
    yunit='vertical units',
)


class filepart(Enum):
    other = auto()
    preamble = auto()
    data = auto()
    cursor = auto()


class commer(Serial):

    def __init__(self, port=None, timeout=1.0, *args, **kwargs):

        if port is None:
            port = "com5"

        super().__init__(port, timeout=timeout, *args, **kwargs)

    def converse(self, cmd: str):
        self.flushOutput()
        ecmd = (cmd + '\n').encode()
        self.write(ecmd)
        self.flushInput()
        return self.recv()

    def send(self, msg):
        output = msg + '\n'
        return self.write(output.encode())

    def recv(self, sep=b'\r\n\n\r', length=None):
        if length is not None:
            return self.read(length)

        chunk = b' '
        resp = b''

        while chunk not in sep and chunk != b'':
            chunk = self.read(1)
            resp += chunk

        return resp.decode()

    def wfmpre(self):
        resp = self.converse("wfmpre?")
        return _wfmpre(*resp.strip().split(';'))
        # return self._wfmpre(*resp.split(';'))

    def get_curve(self, channel):
        possibles = ["ch1", "ch2", "ch3", "ch4"]
        # possibles += [1,2,3,4]
        if channel not in possibles:
            raise TypeError(f"{channel}")
        self.send(f"data:source {channel}")

        cursor_str = self.converse("cursor?")
        data_str = self.converse("data?")
        pre = self.wfmpre()

        c.flushInput()
        self.send("curve?")

        resp = b''
        while len(resp) < 10000:
            chunk = self.read(10)
            resp += chunk
            if chunk == b'':
                break

        if len(resp) != 10000:
            raise ValueError("Did not recieve all the curve data! Is correct channel selected on the scope?")

        return waveform(pre, resp, cursor_str, data_str)

    def upload_curve(self, ref, wfm):
        """Specify the reference waveform using DATa:DESTination.
        2. Specify the record length of the reference waveform using WFMPre:NR_Pt.
        3. Specify the waveform data format using DATa:ENCdg.
        4. Specify the number of bytes per data point using DATa:WIDth.
        5. Specify first data point in the waveform record using DATa:STARt.
        6. Transfer waveform preamble information using WFMPRe.
        7. Transfer waveform data to the oscilloscope using CURVe."""

        spots = ('ref1', 'ref2', 'ref3', 'ref4')
        if ref.lower() not in spots:
            raise NameError(f"ref must be one of {spots} not {ref}")

        rawdata = wfm.raw()
        self.send(f"data:destination {ref}")
        self.send(f"wfmpre:nr_pt 10000")
        self.send(f'data:encdg BIN')
        self.send(f'data:width 1')
        self.send(f'data:start 1')

        self.write(b"curve " + rawdata)

    def __del__(self):
        self.close()
        super().__del__()


class waveform:

    def __init__(self, preamble, data, cursor=None, data_str=None):

        if preamble.encdg == "BIN":
            if preamble.bn_fmt == "RI":  # signed
                if preamble.byt_nr == '2':  # short
                    ctype = 'h'  # signed short
                else:  # char
                    ctype = 'b'  # signed char
            else:  # unsigned
                if preamble.byt_nr == '2':  # short
                    ctype = 'H'  # unsigned short
                else:  # char
                    ctype = 'c'  # unsigned char

            if preamble.byt_or == "LSB":
                byte_order = '<'

            else:
                byte_order = '>'

            pack_str = f'{byte_order}{len(data)}{ctype}'
            self._data = unpack(pack_str, data)
        else:
            raise Exception(f"bad Preamble! {preamble}")
        self.series = pd.Series(self._data)
        self.series.index *= float(preamble.xincr)
        self.series.index = pd.to_timedelta(self.series.index, unit="sec")
        self.series = self.series * float(preamble.ymult) + float(preamble.yoff)

        self.cursor = cursor
        self.data_str = data_str
        self.preamble = preamble

    def clean_step(self, sigma=3.0, window=None):
        """Replace datapoints with std>sigma over a window
        with the mean in that window. This operation is not
        reversible"""

        if window is None:
            window = int(0.1 * len(self.series))

        start = 0
        stop = len(self.series)
        step = window

        n_replaced = 0

        for idx, ii in enumerate(range(start, stop, step)):
            sl = self.series.iloc[ii:ii + step, 0]
            z = np.abs(stats.zscore(sl))

            n_replaced += len(sl[z > sigma])
            sl[z > sigma] = sl.mean()

        print(f"Replaced {n_replaced} points ")

    def set_index(self, Type="float"):
        Types = ["time", "float"]

        if Type == "time":

            self.series.index = pd.to_timedelta(self.series.index, unit="sec")
            self.index_type = Type
        elif Type == "float":

            self.series.index = self.series.index.total_seconds()
            self.index_type = Type
        else:
            raise ValueError(f"Type argumnet must be in {Types}")

    def zero_mean(self, inplace=True):
        if inplace:
            self.series = self.series - self.series.mean()
        else:
            return self.series - self.series.mean()

    def dataframe(self):
        return pd.DataFrame(self.series)

    def shift(self, time=0.0):
        news = deepcopy(self.series)
        if type(news.index) == pd.TimedeltaIndex:
            time = pd.Timedelta(seconds=time)
        news.index = news.index + time
        return news

    def raw(self):
        return pack(f'{len(self._data)}b', *self._data)

    def save(self, fname):
        data = StringIO()
        data.write("####### Preamble ##########################\r\n")

        for key, value in self.preamble._asdict().items():
            data.write(f"#\t{key}={value}\r\n")
        data.write("#######  Cursor ##########################\r\n")
        data.write(f"#{self.cursor}\r\n")
        data.write("#######  Data ##########################\r\n")
        data.write(f"#{self.data_str}\r\n")
        data.write("# Delete the above lines to open in excel ######\r\n\r\n")
        pd.DataFrame({"volts [V]": self.series}).to_csv(data, mode='a', index_label='Time [s]')

        data.seek(0)
        with open(fname, 'w') as fd:
            fd.write(data.read())

    def frequency(self, sigma=0):
        X = fftpack.fft(self.series.to_numpy().T[0])
        f_s = len(self.series) / (self.series.index[-1] - self.series.index[0])
        freqs = fftpack.fftfreq(len(X)) * f_s
        f = pd.Series(np.abs(X), index=freqs)
        f = f[f.index > 0]
        f = f[f > f.std() * sigma]
        return f


def fake(tf: signal.TransferFunction=None, noise: float= 0.00, resolution: float= 0.005, t=None):

    class loadedwf(waveform):
        def __init__(self, data):
            self.series = data
            self.series.index = pd.to_timedelta(self.series.index, unit="seconds")

    if tf is None:
        # Under damped 2nd order
        zeta = 0.1*np.random.random()
        w_n = 10*np.random.random()
        tf = signal.TransferFunction([20],[1, zeta*w_n, w_n**2])

    if t is None:
        t = np.linspace(0.0, 100.0, 10000)
    x, y = tf.step(0, N=10000)

    y = np.random.normal(y, noise)
    # add digitization
    digit_sample = [-1, 1]*10
    digit_sample.extend([0]*100) # weight to 0
    digit_sample.extend([-2,2])
    y = y + resolution*np.random.choice(digit_sample, len(y))
    s=pd.Series(y, index=x)

    return loadedwf(s)



def order2(zeta, w_n, k=1):
    return signal.TransferFunction([k], [1, zeta*w_n, w_n**2])

def load(fname):
    class loadedwf(waveform):
        def __init__(self, data):
            self.series = data
            self.series.index = pd.to_timedelta(self.series.index, unit="seconds")
    path = Path(__file__).parent.parent/"waveforms"/fname
    with  path.open() as fd:

        meta = []
        data = StringIO()
        part = filepart.other
        predict = {}
        for rawline in fd:
            line = rawline.strip()

            if line == '':
                pass
            elif line.startswith("#"):

                meta.append(line)
            else:
                data.write(f"{line}\r\n")

            if "####### Preamble " in line:
                part = filepart.preamble

            elif '#######  Cursor ' in line:
                part = filepart.cursor

            elif '#######  Data ' in line:
                part = filepart.data
    data.seek(0)
    series = pd.read_csv(data, index_col="Time [s]")
    return loadedwf(series), meta


class sine_fitter:

    def __init__(self, x, y=None):
        if y is None:
            self.data = x
        else:
            self.data = pd.DataFrame(y, index=x)

    @property
    def x(self):
        return self.data.index.to_numpy()

    @property
    def y(self):
        return self.data.to_numpy().T[0]

    def smooth(self, window=None):
        if window is None:
            # default to 1% of data
            window = int(len(self.data) * 0.01)
        return self.data.rolling(window=window).mean()

    def fit(self, plot=True):
        yoff = float(self.data.mean())
        A = float(self.data.max() - self.data.min())
        phase = math.pi / 4.0
        f = 0
        lowbound = (A - A * 0.01, -1, -math.pi / 4, yoff - 0.01 * yoff)
        upbound = (A + A * 0.01, 1, math.pi / 4, yoff + 0.01 * yoff)
        params, _ = curve_fit(sin_wave, self.x, self.y, p0=(A, f, phase, yoff), bounds=(lowbound, upbound))
        if plot:
            plt.gca().plot(self.x, sin_wave(self.x, *params))
        return params

    def opt(self):
        yoff = float(self.data.mean())
        A = float(self.data.max() - self.data.min())
        phase = math.pi / 4.0
        f = 0.0
        lowbound = (A - A * 0.01, -1, math.pi / 4.5, yoff - 0.001 * yoff)
        upbound = (A + A * 0.01, 1, math.pi / 3.5, yoff + 0.001 * yoff)
        return least_squares(
            sin_wave_res,
            x0=(A, f, phase, yoff),
            bounds=(lowbound, upbound),
            # loss='soft_l1',
            # f_scale=0.1,
            args=(self.x, self.y))

    def plot(self, *args, **kwargs):
        self.data.plot(ax=plt.gca())


def sin_wave(t, A, f, phase, yoff):
    return A * np.sin(2 * math.pi * f * t + phase) + yoff


def sin_wave_res(x, t, y):
    return (x[0] * np.sin(2 * math.pi * t * x[1] + x[2]) + x[3]) - y


def frequency(x, f_s):
    X = fftpack.fft(x)
    freqs = fftpack.fftfreq(len(x)) * f_s
    # p=plt.plot(freqs, np.abs(X), 'r.')
    return X, freqs


def main():
    data = load("132553.csv")[0]
    # sf = sine_fitter(data.index.to_numpy(), data.to_numpy())
    return data


def record(c=None):
    if c is None:
        c = commer('com7', timeout=3)
    now = datetime.datetime.now()
    nowstr = now.strftime("%H%M%S")

    w = c.get_curve("ch1")
    w.save(f"{nowstr}_input.csv")
    w = c.get_curve("ch2")
    w.save(f"{nowstr}_output.csv")

