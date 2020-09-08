"""Main module."""
import json
from struct import pack, unpack
from collections import namedtuple
import pandas as pd
from io import StringIO
from enum import Enum, auto
from scipy import fftpack, stats, signal, special
from scipy.optimize import curve_fit, least_squares
import matplotlib.pyplot as plt
import numpy as np

import math
import datetime
from copy import deepcopy
from serial import Serial
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging
import re
from .appconfig import AppConfig
from uuid import uuid4
from .appconfig import AppConfig; config = AppConfig()
import asyncio
from functools import partial

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






_pre_description = _wfmpre(
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


_data = namedtuple("data", ["encdg", "destination", "source", "start", "stop", "width"])

_vertical = namedtuple("vertical", [
                         'scale',
                         'position',
                         'offset',
                         'coupling',
                         'bandwidth',
                         'deskew',
                         'impedance',
                         'probe',
                         'yunit',
                         'id',
                         'invert',
                         ])

_measurement = namedtuple("meas", [
    'type',
    'units',
    'count',
    'min',
    'max',
    'mean',
    'stddev',
    'source1',
    'source2',
    'delay_direction',
    'edge1',
    'edge2',
    'state',
])



class filepart(Enum):
    other = auto()
    preamble = auto()
    data = auto()
    cursor = auto()



class http:

    """This class encapsulates functions to communicqte with
    the Tektronix oscilloscope over its HTTP/Web interface. It
    is meant to be used as a drop in replacement for the rs232
    class. Unfortunately the HTTP interface was never meant
    to be used as a programming API. You will notice a lot of
    work to make sure we get all the data back from the server.
    """

    def __init__(self, ipaddr: str, port: int=80, chunksize: int=512):
        self.ipaddr = ipaddr
        self.port = port
        self.chunksize = chunksize
        self.url = f"http://{self.ipaddr}:{self.port}/Comm.html"
        self.imageurl = f"http://{self.ipaddr}:{self.port}/Image.png"


        self.cache = {}

        # Here we setup a begin and end tag
        # for communication. Because the messages that
        # come back seem to be split arbitrarily we
        # need to know where the message begins and
        # ends. We will query VAR1 before and VAR2
        # after every msg we send. that way we know
        # we will get a response. :/

        try:
            self.start()
        except RuntimeError as error:
            logging.warning(f"Initial setup failed. API needs to be flushed. This will take a 10 seconds or so.")
            self.hard_flush()
            self.start()


    def start(self):
        rqdata = {
            "COMMAND": ["header off", "math:var1 1.2345"],
            "gpibsend": "Send",
            "name": None
        }

        requests.get(f"http://{self.ipaddr}:{self.port}/Comm.html", rqdata)
        rqdata['COMMAND'] = "math:var2 5.4321"
        requests.get(f"http://{self.ipaddr}:{self.port}/Comm.html", rqdata)

        rqdata['COMMAND'] = "math:var1?"
        self.startstr = requests.get(f"http://{self.ipaddr}:{self.port}/Comm.html", rqdata).text.split('\n')[0]
        rqdata['COMMAND'] = "math:var2?"
        self.endstr = requests.get(f"http://{self.ipaddr}:{self.port}/Comm.html", rqdata).text.split('\n')[0]

        try:
            float(self.startstr)
        except ValueError:

            raise RuntimeError(f"startstr should be convertible to float but it is {self.startstr}")

        try:
            float(self.endstr)
        except ValueError:
            raise RuntimeError(f"startstr should be convertible to float but it is {self.endstr}")

        if float(self.startstr) != 1.2345:
            raise RuntimeError(f"startstr should be 1.2345 but its {self.startstr}")

        if float(self.endstr) != 5.4321:
            raise RuntimeError(f"startstr should be 1.2345 but its {self.endstr}")



    def hard_flush(self):
        """FLush info from the api with the lone `?'"""
        # I have found that sending the `?' alone
        # flushes that http api without adding anything
        # this might work in situations were flush
        # isn't enough. This takes a long time ~ 10 seconds.

        rqdata = {
            "COMMAND": "?",
            "gpibsend": "Send",
            "name": None
        }

        requests.get(self.url, rqdata)

    def converse(self, msg: str):
        """Send a message return a response
        The response is received in the html textarea.
        We use BeautifulSoup to extract it.

        If the the response is more than 80 characters
        or so it is broken up into more than one packet
        To make sure we get the whole message we request
        var1 before and var2 after the request in msg.
        In the response, we throw away everything before
        VAR1 and keep flushing until we see var2.
        """
        self.flush()
        rqdata = {
            "COMMAND": ["math:var1?", msg, "math:var2?"],
            # "COMMAND": [""],
            "gpibsend": "Send",
            "name": None
        }
        resp = requests.post(self.url, rqdata)
        sp = BeautifulSoup(resp.text, features="html.parser")
        dt = sp.find("textarea").contents[0]
        stag = dt.find(self.startstr)
        etag = dt.find(self.endstr)
        etag_count = 1
        if stag == -1: raise ValueError( f"Cannot find starttag in {dt}" )

        clipped = dt[stag + len(self.startstr) + 1:]
        if etag == -1:
            clipped += self.flush()

        elif etag < stag:
            raise ValueError( f"{etag} is greater than {stag}!: ==> {dt}" )

        return clipped[:clipped.find(self.endstr)].strip()

    def flush(self):
        """Send requests for Var2 until
        we see VAR2 in the response. If this takes
        more than max_count iterations
        we throw an error.  We return all the
        data before the VAR2.
        """

        buffer = ""
        dt = ""
        count = 0
        max_count = 10
        while dt != self.startstr:

            rqdata = {
                "COMMAND": "math:var2?",
                "gpibsend": "Send",
                "name": None
            }
            resp = requests.post(self.url, rqdata)
            sp = BeautifulSoup(resp.text, features="html.parser")
            dt = sp.find("textarea").contents[0]
            buffer += dt
            if dt == self.endstr:
                break

            if count >= max_count:
                raise StopIteration(f"Max iter is acheived without self.endstr! {buffer}")

            count += 1

        return buffer

    def wfmpre(self):

        resp = self.converse("wfmpre?")
        preamble = _wfmpre(*resp.strip().split(';'))

        # The http interface seems to ignore
        # some aspects of the preamble. The
        # byt_nr ===2 and bn_fmt === 'RI'
        # regardless of what is stored in the preamble.
        preamble = preamble._replace(byt_nr=2, bn_fmt='RI')

        self.cache['preamble'] = preamble._asdict()
        return preamble

    def data(self):
        """Retrieve 'data' from scope. This
        gives us the format and location
        of waveform data
        """

        resp = self.converse("data?")
        data = _data(*resp.split(';'))
        self.cache['data'] = data
        return data

    def vertical(self, channel):
        possibles = ["ch1", "ch2", "ch3", "ch4"]

        if channel not in possibles:
            raise ValueError(f"channel must be in {possibles} you gave {channel}")

        resp = self.converse(f"{channel}?")
        vertical = _vertical(*resp.split(';'))
        self.cache[channel] = vertical

        return vertical

    def measurement(self, num):
        possibles = [1, 2, 3, 4]
        if num not in possibles:
            raise ValueError(f"Measurement number (num) must be in {possibles} not {num} .")

        resp = self.converse(f"measurement:meas{num}?")
        try:
            return _measurement(*resp.split(';'))
        except Exception as error:
            logging.warning(f"Error collecting measurement: {error}")
            return resp


    def get_curve(self, channel):

        possibles = ["ch1", "ch2", "ch3", "ch4"]
        regex = re.compile(r"#([0-9])")

        if channel not in possibles:
            raise ValueError(f"{channel}")

        rqdata = {
            "command": [f"select:{channel} on", "save:waveform:fileformat internal"],
            "wfmsend": "Get"
        }

        self.converse(f"data:source {channel}")
        vertical = self.vertical(channel)
        data_info = self.data()
        pre = self.wfmpre()

        resp = requests.get(f"http://{self.ipaddr}:{self.port}/getwfm.isf", rqdata, stream=True)

        last_char = b""
        for char in resp.iter_content(1):
            substr = (last_char+char).decode()
            match = regex.match( substr)
            if match:
                next_read = int(match.group(1))
                break

            last_char = char

        length = int(resp.raw.read(next_read))
        logging.debug(f"We have {length} bytes to read")
        data = resp.raw.read(length)

        return waveform(pre, data, vertical, data_info, self.get_raw_image())

    def get_raw_image(self):
        resp = requests.get(f"http://{self.ipaddr}/Image.png")
        return resp.content

class waveform:

    divs = 10
    """This class stores and analyzes waveforms
    from the oscilloscope."""
    def __init__(self, preamble, data, vertical=None, data_info=None, img=None):

        self.config = AppConfig
        if preamble.encdg == "BIN":
            if preamble.bn_fmt == "RI":  # signed
                if preamble.byt_nr in ('2', 2):  # short
                    ctype = 'h'  # signed short
                else:  # char
                    ctype = 'b'  # signed char
            else:  # unsigned
                if preamble.byt_nr in ('2',2):  # short
                    ctype = 'H'  # unsigned short
                else:  # char
                    ctype = 'c'  # unsigned char

            if preamble.byt_or == "LSB":
                byte_order = '<'

            else:
                byte_order = '>'

            pack_str = f'{byte_order}{int(preamble.nr_pt)}{ctype}'
            try:
                self._data = unpack(pack_str, data)
            except Exception as error:
                print(error)
                print(pack_str)
                self.cursor = cursor
                self.data_info = data_info
                self.preamble = preamble
                self.unpack_str = pack_str
                self.raw = data
                self.img = img
                return
        else:
            raise Exception(f"bad Preamble! {preamble}")

        self.series = pd.Series(self._data)
        self.series.index *= float(preamble.xincr)
        #self.series.index = pd.to_timedelta(self.series.index, unit="sec")
        self.series.index.name='microseconds'
        self.series.name='Volts'
        #self.series = self.series * float(preamble.ymult) + float(preamble.yoff)

        self.series *= self.divs/(2**(8*preamble.byt_nr))


        self.data_info = data_info
        self.preamble = preamble
        self.vertical = vertical
        self.unpack_str = pack_str
        self.img = img
        self.raw = data

    @property
    def x(self):
        x = self.series.index.to_numpy()
        if len(x.shape) != 1:
            x = x.reshape((len(x), ))
        if len(x.shape) != 1:
            raise ValueError(f"Can't convert to 1d numpy array x.shape = {x.shape}")

        return x

    @property
    def y(self):
        y = self.series.to_numpy()
        if len(y.shape) != 1:
            y = y.reshape((len(y), ))
        if len(y.shape) != 1:
            raise ValueError(f"Can't convert to 1d numpy array y.shape = {y.shape}")

        return y

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
            raise ValueError(f"Type argument must be in {Types}")

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


    def meta(self):
        return {
            "preamble":self.preamble._asdict(),
            "data": self.data_info._asdict(),
            "vertical": self.vertical._asdict()
        }

    def save(self, name=None, path=Path("waveforms/"), unique=None):


        # get 4 bytes for a unique ID
        if unique is None:
            unique = str(uuid4())[:4]

        if name is None:
            name = datetime.datetime.now().isoformat()

        full_path = path/f"{unique}_{name}"

        full_path.mkdir(parents=True)

        with open(full_path/"meta.json", 'w') as meta_file:
            json.dump(self.meta(), meta_file)

        with open(full_path/"data.csv", 'w') as data_file:
            self.dataframe().to_csv(data_file)

        with open(full_path/"raw.bin", 'wb') as bin_file:
            bin_file.write(self.raw)

        if self.img:
            with open(full_path/"image.png", 'wb') as image_file:
                image_file.write(self.img)

        return f"{unique}_{name}"

    def frequency(self, sigma=0):
        arr = self.x
        if self.index_type != "float":
            self.set_index("float")
        if len(arr.shape) != 1:
            arr = arr.T[0]
        X = fftpack.fft(arr)
        f_s = len(self.series) / (self.series.index[-1] - self.series.index[0])
        freqs = fftpack.fftfreq(len(X)) * f_s
        f = pd.Series(np.abs(X), index=freqs)
        f = f[f.index > 0]
        f = f[f > f.std() * sigma]
        return f

    def sigmoid_fit(self):
        sigmoid = lambda x, x0, y0, rise, amp : y0+amp*special.expit(rise*(x-x0))
        x, y = self.x, self.y
        xi = np.mean(x)
        y1 = np.mean(y)
        ampi = 1.0
        risei = 1.0

        resp, cov = curve_fit(sigmoid, x, y, p0=[xi, yi, ampi, risei])

    def sigmoid_normalize(self, inplace=False):
        """Fit the waveform a sigmoid(logistical) function

        """
        sigmoid = lambda x, x0, y0, rise, amp: y0 + amp * special.expit(rise * (x - x0))
        x, y = self.x, self.y
        xi = np.mean(x)
        yi = np.mean(y)
        ampi = 1.0
        risei = 1.0
        (x0, y0, rise, amp), cov = curve_fit(sigmoid, x, y, p0=[xi, yi, ampi, risei])
        nfit = sigmoid(x, x0, 0, rise, 1)

        idx = np.argmax(nfit > 0.01)
        t0 = self.series.index[idx]

        output = pd.Series((self.series - y0) / amp)

        output.index -= t0
        if inplace:
            self.series = output[0:]
        else:
            return output[0:], (x0, y0, rise, amp), nfit[idx:]

    def analyze(self, peak_width=0.01):
        x, y = self.x, self.y
        an = namedtuple("analysis", "peaks peak_time final_value rise_time rise_time_rate overshoot t_10 t_90")
        peaks, props = signal.find_peaks(y, width=peak_width*len(y))
        final_value = self.series.iloc[int(-peak_width*len(x)):].mean()
        if len(peaks) < 1:
            peak_time = None
            overshoot = None
        else:
            peak_time = self.series.index[peaks[0]]
            overshoot = self.series[peak_time] - final_value
        print(final_value)
        t_10 = self.series[self.series > 0.1*final_value].index[0]
        t_90 = self.series[self.series > 0.9*final_value].index[0]

        rise_time = t_90-t_10
        rise_time_rate = (self.series[t_90]-self.series[t_10])/rise_time

        return an(peaks, peak_time, final_value, rise_time, rise_time_rate, overshoot, t_10, t_90)

    def plot_analysis(self, ax=None, an=None):
        x, y = self.x, self.y
        if ax is None:
            ax = plt.gca()
        if an is None:
            an = self.analyze()

        ax.plot(x, y, label="Wave Form")
        ax.plot(x[an.peaks], y[an.peaks], "y+", label="Peaks")
        ax.plot(x, [an.final_value]*len(x), "r--", label="Final")
        y_10 = self.series[an.t_10]
        y_90 = self.series[an.t_90]
        ax.plot([an.t_10, an.t_90], [y_10, y_90], "g--", label="Rise")
        ax.legend()
        ax.grid()

        return ax


class loadedwf(waveform):

    def __init__(self, data, meta):

        self.preamble = _wfmpre(**meta['preamble'])
        self.data_info = _data(**meta['data'])
        self.vertical = _vertical(**meta['vertical'])

        self.series = data

class curve_job:

    def __init__(self, name, channel, unique=None):

        if unique is None:
            unique = str(uuid4())[:4]

        self.unique = unique
        self.name = name
        self.channel = channel


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

        def __init__(self, data, fname):
            self.series = data
            self.series.index = pd.to_timedelta(self.series.index, unit="seconds")
            self.fname = fname

    path = Path(config["DEFAULT"]['waveform_path'])

    with path.open() as fd:

        meta = []
        data = StringIO()
        part = filepart.other
        predict = {}

    series = pd.read_csv(data, index_col="Time [s]")

    return loadedwf(series, fname), meta

def load_all():
    path = Path(__file__).parent/"waveforms"

    wfs = []
    for fname in path.iterdir():
        try:
            wfs.append(load(fname))
        except Exception as err:
            print(f"Error loading {fname}: {err}")

    return wfs


class MultipleConnectionError(RuntimeError):
    pass



class comm_singleton:
    instance = None

    class http_wrapper:
        http = None

        def __init__(self):
            self.connected = False
            self.session_id = None
            self.session_time = None
            self.time_limit = datetime.timedelta(hours=2)
            self.details = None
            self.current_job = None
            self.last_job = None
            self.error = None

        def start(self, ipaddr, connection_details, port=80):
            self.session_time = datetime.datetime.now()
            if self.session_id is not None:
                raise MultipleConnectionError("Only one connection allowed")

            self.http = http(ipaddr, port)
            self.session_id = uuid4()
            self.connected = True
            self.details = connection_details

        def converse(self, msg: str):
            self.session_time = datetime.datetime.now()
            if self.connected:
                return self.http.converse(msg)
            else:
                raise RuntimeError(f"Must connect before sending msg")

        def current_job_finished(self):

            if self.error:
                raise self.error

            if self.current_job:
                return False
            else:
                jobid = self.last_job
                self.last_job = None
                return jobid

        def clear_error(self):

            self.error = None
            self.last_job = None
            self.current_job = None


        async def get_and_save(self, name, channel, unique=None):
            """Retrieve the waveform from the scope and
            save it to disk. This should be done with
            concurrency. """

            loop = asyncio.get_running_loop()
            self.error = None
            # get 4 bytes for a unique ID
            if unique is None:
                unique = str(uuid4())[:4]

            try:
                self.current_job = unique
                # create partial fxns and run them in the background
                fget_curve = partial(self.http.get_curve, channel)
                curve = await loop.run_in_executor(None, fget_curve)
                fsave = partial(curve.save,
                                name,
                                path=Path(config["DEFAULT"]['waveform_path']),
                                unique=unique)

                dirname = await loop.run_in_executor(None, fsave)

                self.last_job = dirname
                self.current_job = None

            except Exception as error:
                logging.warning(f"There was an error in getting the curve f{error}")
                self.error = error


            return dirname

        def stop(self):
            self.connected = False
            self.session_time = None
            self.session_id = None
            self.details = None

        def state(self):
            self.session_time = datetime.datetime.now()
            return {
                "connected": self.connected,
                "pyobj": str(self.http),
                "session": str(self.session_id),
                "details": self.details
            }

        def has_expired(self):
            if self.session_time is None:
                raise RunTimeError(f"{self.__class__} cannot expire if it has not been started")

            return (datetime.datetime.now()-self.session_time) > self.time_limit

        @property
        def imageurl(self):
            return self.http.imageurl

    def __new__(cls):
        if cls.instance is None:
            cls.instance = cls.http_wrapper()
        return cls.instance





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

