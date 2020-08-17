import io
import json
import asyncio
from scipy import signal
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, namedtuple



import sys
from pathlib import Path

sys.path.insert( 0, str(Path.cwd()) )

try:
    import tornado
except ImportError:
    raise RuntimeError("This example requires tornado.")
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.websocket
import tornado.gen
import tektronix.tektronix as tek
from .appconfig import AppConfig; config = AppConfig()
import matplotlib
from uuid import uuid4


matplotlib.use('webagg')
from matplotlib.backends.backend_webagg_core import (
    FigureManagerWebAgg, new_figure_manager_given_figure,
    FigureCanvasWebAggCore)






class FigureContainer:
    """Contains the figure and their respective managers
    To our eventual detriment we are assuming on plot
    per figure. This should be abandoned ASAP.
    """

    def __init__(self):
        self._figures = OrderedDict()
        self._managers = OrderedDict()

    def add_figure(self, key, figure=None, *args, **kwargs):
        if figure is None:
            figure = Figure(*args, **kwargs)

        self._figures[key] = figure
        self._managers[key] = new_figure_manager_given_figure(id(figure), figure)

        if len(figure.axes) == 0:
            # at least 1 axis per figure
            figure.add_subplot(111)

    def __getitem__(self, key):
        return self._figures[key]

    def iterall(self):
        for key, fig, in self._figures.items():
            man = self._managers[key]
            yield key, fig, man, id(fig)

    def __iter__(self):
        for key, fig, in self._figures.items():
            yield key, fig,

    def cla(self, key):
        fig = self._figures[key]
        fig.axes[0].cla()

    def plot(self, key, x, y, *args, **kwargs):

        if key not in self._figures:
            raise KeyError(f"{key} not a figure in this container")

        ax = self._figures[key].axes[0]

        ax.cla()  # Clear the plot?
        ax.plot(x, y, *args, **kwargs)
        plt.tight_layout(1)
        return ax

    def flush(self, key=None):

        if key is None:
            figs = self._figures.values()
            managers = self._managers.values()

        else:
            figs = [self._figures[key]]
            managers = [self._managers[key]]

        for figure, manager in zip(figs, managers):
            canvas = FigureCanvasWebAggCore(figure)
            manager.canvas = canvas
            manager.canvas.manager = manager
            manager._get_toolbar(canvas)
            manager._send_event("refresh")
            manager.canvas.draw()

    def get_manager(self, key):
        return self._managers[key]

    def get_id(self, key):
        return id(self._figures[key])

    def figure_by_id(self, ID):
        # Maybe this should be a map for
        # faster indexing.

        for figure in self._figures.values():
            if id(figure) == ID:
                return figure
        raise ValueError(f"Figure with id {ID} does not exist")

    def manager_by_id(self, ID):
        # Maybe this should be a map for
        # faster indexing.

        for key, figure in self._figures.items():
            if id(figure) == ID:
                return self._managers[key]
        raise ValueError(f"Figure with id {ID} does not exist")

    def create_figures(self, t=None, s=None):
        """
        Creates a simple example figure.
        """
        figs = [Figure() for a in range(4)]

        (p11, p12, p21, p22) = [fig.add_subplot(111) for fig in figs]

        if t is None:
            t = np.arange(0.0, 3.0, 0.01)
        if s is None:
            s = np.sin(2 * np.pi * t)

        p11.plot(t, s)
        return figs


# The following is the content of the web page.  You would normally
# generate this using some sort of template facility in your web
# framework, but here we just use Python string formatting.
html_content = """

"""
phi = 0


class MyApplication(tornado.web.Application):

    class MainPage(tornado.web.RequestHandler):
        tekcomm = tek.comm_singleton()
        """
        Serves the main HTML page.
        """
        def get(self):
            session_cookie = self.get_secure_cookie('session')
            tek_ip = self.get_argument("tek_ip", None)

            if session_cookie:
                session_cookie = session_cookie.decode()
                if str(self.tekcomm.session_id) == session_cookie:
                    self.render((config['WEB']['template_path']+"/running.html"), tekcomm=self.tekcomm)

                else:
                    self.render((config['WEB']['template_path'] + "/error.html"),
                                error=f"session cookie mismatch {session_cookie} is not {self.tekcomm.session_id}",
                                msg="This means someone else is connected or there is a bug."
                                )

            else:
                if self.tekcomm.session_id is None:
                    if tek_ip is not None:
                        try:
                            self.tekcomm.start(tek_ip, {'client_ip': self.request.remote_ip})
                            self.set_secure_cookie("session", str(self.tekcomm.session_id))
                            self.render((config['WEB']['template_path'] + "/running.html"), tekcomm=self.tekcomm)
                        except Exception as Error:
                            self.render((config['WEB']['template_path'] + "/connect.html"),
                                        tekcomm=None,
                                        error=f"Could not connect to {tek_ip}<br>{Error}",
                                        msg="This is likely caused by wrong IP or other network issues"
                                        )
                    else:
                        self.render((config['WEB']['template_path'] + "/connect.html"),
                                    tekcomm=None,
                                    error="",
                                    msg=""
                                    )
                else:
                    self.write(self.tekcomm.state())



    class TxFunction(tornado.web.RequestHandler):

        def get(self):
            fc = self.application.fig_container
            figure = self.application.fig_container['fig1']
            fc.get_manager('fig2').set_window_title("HEY!!")
            fc.flush('fig2')
            loop = tornado.ioloop.IOLoop.instance()
            # loop.add_callback(self.application.test)
            host = self.request.host
            ws_uri = f"ws://{host}"
            content = html_content % {
                "ws_uri": ws_uri, "fig_id": id(figure)}
            self.render(config['WEB']['template_path']+"/txfxn.html", ws_uri=ws_uri, fc=fc, host=host,
            rproxy="apps/tektronix")

    class dosomething(tornado.web.RequestHandler):

        def get(self):
            global phi
            print(f"{phi}")
            phi += np.pi / 10
            fc = self.application.fig_container

            x = np.linspace(0, 8 * np.pi, 100)
            y = np.sin(x + phi)
            fc.plot('fig1', x, y)
            fc.flush()

            self.write("Success")

    class init_plots(tornado.web.RequestHandler):

        def get(self):
            fc = self.application.fig_container
            fc.get_manager('fig1').set_window_title("Channel 1")
            fc.get_manager('fig2').set_window_title("Channel 2")
            #fc.get_manager('fig3').set_window_title("Channel 3")
            #fc.get_manager('fig4').set_window_title("Channel 4")



            self.write("Plots Inited")

    class step(tornado.web.RequestHandler):

        def post(self):
            holder = namedtuple('data', 'num den noise', defaults=[None, None, 0.01])
            data = json.loads(self.request.body)
            data = holder(**data)
            num = list(map(float, data.num))
            den = list(map(float, data.den))
            noise = float(data.noise)
            if num is None:
                num = [1]
            elif den is None:
                den = [1, 1, 1]
            tf = signal.TransferFunction(num, den)

            t = np.linspace(0, 100, 10000)
            wf = tek.fake(tf, noise=noise)
            x, y = wf.series.index, wf.series
            fc = self.application.fig_container
            yin = np.ones(x.shape)
            yin[0] = 0.0
            ax1 = fc.plot("fig1", x, yin)
            ax1.set_title("Input")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Amplitude")
            ax1.grid()

            ax2 = fc.plot('fig2', x, y)
            ax2.set_title("Output")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Amplitude")
            ax2.grid()

            fc.flush()
            print(tf)
            self.write(json.dumps({"status": 'Success'}))

    class MplJs(tornado.web.RequestHandler):
        """
        Serves the generated matplotlib javascript file.  The content
        is dynamically generated based on which toolbar functions the
        user has defined.  Call `FigureManagerWebAgg` to get its
        content.
        """

        def get(self):
            self.set_header('Content-Type', 'application/javascript')
            js_content = FigureManagerWebAgg.get_javascript()

            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        """
        Handles downloading of the figure in various file formats.
        """

        def get(self, key, fmt):

            fc = self.application.fig_container
            fig = fc[key]
            manager = fc.get_manager(key)
            mimetypes = {
                'ps': 'application/postscript',
                'eps': 'application/postscript',
                'pdf': 'application/pdf',
                'svg': 'image/svg+xml',
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'tif': 'image/tiff',
                'emf': 'application/emf',
                "csv": 'text/csv'
            }
            if fmt != 'csv':
                self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))

                buff = io.BytesIO()
                manager.canvas.figure.savefig(buff, format=fmt)
                self.write(buff.getvalue())
            else:
                self.set_header('Content-Type', mimetypes.get(fmt, 'utf-8'))
                buff = io.StringIO()
                x = fig.axes[0].lines[0].get_xdata()
                y = fig.axes[0].lines[0].get_ydata()
                s = pd.Series(y, index=x)
                s.to_csv(buff)
                buff.seek(0)
                self.write(buff.read())



    class WebSocket(tornado.websocket.WebSocketHandler):
        """
        A websocket for interactive communication between the plot in
        the browser and the server.

        In addition to the methods required by tornado, it is required to
        have two callback methods:

            - ``send_json(json_content)`` is called by matplotlib when
              it needs to send json to the browser.  `json_content` is
              a JSON tree (Python dictionary), and it is the responsibility
              of this implementation to encode it as a string to send over
              the socket.

            - ``send_binary(blob)`` is called to send binary image data
              to the browser.
        """
        supports_binary = True

        def open(self, key):
            # Register the websocket with the FigureManager.
            fc = self.application.fig_container
            manager = fc.get_manager(key)
            manager.add_web_socket(self)
            self.manager = manager
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            # When the socket is closed, deregister the websocket with
            # the FigureManager.
            self.manager.remove_web_socket(self)
            # fc = self.application.fig_container
            # for key, fig, manager, ID in fc.iterall():
            # manager.remove_web_socket(self)

        def on_message(self, message):
            # The 'supports_binary' message is relevant to the
            # websocket itself.  The other messages get passed along
            # to matplotlib as-is.

            # Every message has a "type" and a "figure_id".
            message = json.loads(message)
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                fc = self.application.fig_container
                manager = fc.manager_by_id(message['figure_id'])
                manager.handle_json(message)

        def send_json(self, content):
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            if self.supports_binary:
                self.write_message(blob, binary=True)
            else:
                data_uri = "data:image/png;base64,{0}".format(
                    blob.encode('base64').replace('\n', ''))
                self.write_message(data_uri)


    class Waveform(tornado.web.RequestHandler):

        def get(self, wfname):


            fc = self.application.fig_container
            figure = self.application.fig_container['fig1']
            host = self.request.host
            ws_uri = f"ws://{host}"
            path = Path(config["DEFAULT"]["waveform_path"])/wfname
            data = pd.read_csv(path/"data.csv")
            meta = json.load(open(path/"meta.json"))

            try:
                wf = tek.loadedwf(pd.Series(data), meta)
                errmsg = ""
            except Exception as error:
                print(error)
                #fc = None
                errmsg = str(error)
            #x = wf.x
            ##y = wf.y
            #print(len(x), len(y))
            ax=fc.plot('fig1', data[data.columns[0]], data[data.columns[1]])
            ax.grid()
            ax.set_title("Interactive Plot")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Volts")
            #fc.get_manager('fig2')

            self.render(config['WEB']['template_path'] + '/wf.html',
                        wfdir=wfname,
                        fc=fc,
                        ws_uri=ws_uri,
                        host=host,
                        error=errmsg)



    class ListWaveforms(tornado.web.RequestHandler):

        def get( self ):

            wfpath = Path(config["DEFAULT"]["waveform_path"])
            wfpath.mkdir(parents=True, exist_ok=True)
            names = [ path.name for path in wfpath.iterdir() ]

            self.render(config['WEB']['template_path']+"/waveforms.html", waveforms=names)


    class WaitForWaveform(tornado.web.RequestHandler):
        def get(self):

            self.render(config['WEB']['template_path']+"/saveandwait.html")

    # Begin tektronix api

    class converse(tornado.web.RequestHandler):
        tekcomm = tek.comm_singleton()
        def post( self ):
            arg = self.get_argument("msg")

            # We should probably do some error checking

            self.tekcomm.converse(arg)

    class start(tornado.web.RequestHandler):

        tekcomm = tek.comm_singleton()
        def post( self ):
            ipaddr = self.get_argument( "ipaddr" )
            port = self.get_argument("port", 80)

            # We should probably do some error checking
            #
            self.tekcomm.start(ipaddr, {'client_ip': self.request.remote_ip}, port)
            self.set_secure_cookie("session", self.tekcomm.session_id)
            self.write(self.tekcomm.state())


    class state(tornado.web.RequestHandler):
        tekcomm = tek.comm_singleton()

        def get(self):
            self.write(self.tekcomm.state())

    class save(tornado.web.RequestHandler):

        tekcomm = tek.comm_singleton()


        async def get(self):

            try:
                is_finished = self.tekcomm.current_job_finished()
            except Exception as error:
                self.tekcomm.clear_error()

                await self.render(config['WEB']['template_path'] + "/saveandwait.html",
                                  name=name,
                                  status="Error",
                                  error=str(error))

            if is_finished:

                self.redirect(f"/waveform/{is_finished}")

            elif is_finished is None:
                channel = self.get_argument("channel")
                name = self.get_argument("name")
                unique = str(uuid4())[:4]

                self.current_job = asyncio.create_task(self.tekcomm.get_and_save(name, channel, unique))

                await self.render(config['WEB']['template_path'] + "/saveandwait.html",
                            name=name,
                            status="Working",
                            error="")
                #self.write({"status": "started", "wfname": f"{unique}_{name}"})

            else:
                name = self.get_argument("name")
                await self.render(config['WEB']['template_path'] + "/saveandwait.html",
                                  name=name,
                                  status="Working",
                                  error="")





    async def test(self):
        fc = self.fig_container
        phi = 0

        for a in range(10):
            x = np.linspace(0, 8 * np.pi, 10000)
            y = np.sin(x + phi)
            fc.plot('fig1', x, y)
            fc.flush('fig1')
            phi += np.pi / 100
            await asyncio.sleep(0.01)

        fc.get_manager('fig2').set_window_title("HEY!!")
        x = np.linspace(0, 100, 10000)
        fc.flush('fig2')


    def __init__(self, fc):
        self.fig_container = fc
        # self.figure = figure[1]
        # self.manager = figure._managers[1]
        # self.manager = new_figure_manager_given_figure(id(figure), figure)

        print(str(Path(config['WEB']['favicon_path'])))
        print(Path(config['WEB']['favicon_path']).exists())

        super().__init__([
            # MPL Static files for the CSS and JS
            (r'/_static/(.*)',
             tornado.web.StaticFileHandler,
             {'path': FigureManagerWebAgg.get_static_file_path()}),

            # My static files
            (r'/s/(.*)',
             tornado.web.StaticFileHandler,
             {'path': str(Path(config['WEB']['static_path']))}),


            # Waveforms static path
            (r'/wf/(.*)',
             tornado.web.StaticFileHandler,
             {'path': str(Path(config['WEB']['waveform_path']))}
             ),

            # favicon
            (r'/(favicon.ico)',
             tornado.web.StaticFileHandler,
             {'path': str(Path(config['WEB']['favicon_path']))}),


            # The page that contains all of the pieces
            ('/', self.MainPage),
            #('/', self.TxFunction),

            ('/waveforms.html', self.ListWaveforms),

            (r'/waveform/(.*)', self.Waveform),

            (r'/saveandwait', self.WaitForWaveform),

            ('/mpl.js', self.MplJs),

            # Sends images and events to the browser, and receives
            # events from the browser

            ('/([a-z0-9]+)/ws', self.WebSocket),

            # Handles the downloading (i.e., saving) of static images
            (r'/([a-z0-9]+)/download.([a-z0-9.]+)', self.Download),

            (r'/dosomething', self.dosomething),
            (r'/step', self.step),
            (r'/init', self.init_plots),
            (r'/tek/start', self.start),
            (r'/tek/state', self.state),
            (r'/tek/converse', self.converse),
            (r'/tek/save', self.save)



        ], debug=True,

            cookie_secret=str(uuid4()))

