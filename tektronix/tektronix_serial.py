class rs232(Serial):

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

        if channel not in possibles:
            raise TypeError(f"{channel}")
        self.send(f"data:source {channel}")

        vertical = self.vertical(channel)
        data_str = self.converse('data?')
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
            raise ValueError("Did not receive all the curve data! Is correct channel selected on the scope?")

        return waveform(pre, resp, vertical, data_info)

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
