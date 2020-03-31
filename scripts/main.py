import io

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
import matplotlib

matplotlib.use('webagg')
from matplotlib.backends.backend_webagg_core import (
    FigureManagerWebAgg, new_figure_manager_given_figure,
    FigureCanvasWebAggCore)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, namedtuple
import json
import asyncio
from scipy import signal
import pandas as pd

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

    def create_figures(t=None, s=None):
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
        """
        Serves the main HTML page.
        """

        def get(self):
            fc = self.application.fig_container
            figure = self.application.fig_container['fig1']
            fc.get_manager('fig2').set_window_title("HEY!!")
            fc.flush('fig2')
            loop = tornado.ioloop.IOLoop.instance()
            # loop.add_callback(self.application.test)
            host = self.request.host
            ws_uri = f"ws://{host}/fig1/"
            content = html_content % {
                "ws_uri": ws_uri, "fig_id": id(figure)}
            self.render("index.html", ws_uri=ws_uri, fc=fc, host=host)

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

        super().__init__([
            # Static files for the CSS and JS
            (r'/_static/(.*)',
             tornado.web.StaticFileHandler,
             {'path': FigureManagerWebAgg.get_static_file_path()}),

            # The page that contains all of the pieces
            ('/', self.MainPage),

            ('/mpl.js', self.MplJs),

            # Sends images and events to the browser, and receives
            # events from the browser
            ('/([a-z0-9]+)/ws', self.WebSocket),

            # Handles the downloading (i.e., saving) of static images
            (r'/([a-z0-9]+)/download.([a-z0-9.]+)', self.Download),

            (r'/dosomething', self.dosomething),
            (r'/step', self.step),
            (r'/init', self.init_plots)

        ])


# if __name__ == "__main__":
def main():
    figs = FigureContainer()
    figs.add_figure("fig1", figsize=(6, 3.0))
    figs.add_figure("fig2", figsize=(6, 3.0))
    # figs.add_figure("fig3", figsize=(6, 1.8))
    # figs.add_figure("fig4", figsize=(6, 1.8))

    x = np.linspace(0, np.pi * 8, 10000)
    y = np.sin(x)
    #figs.plot('fig1', x, y, 'r.')
    #figs.plot('fig2', x, y + 10, )
    # figs.plot('fig3', x, np.exp(x))
    # figs.plot('fig4', x, -np.exp(x))

    # figure = create_figures()[0]
    application = MyApplication(figs)

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    loop = tornado.ioloop.IOLoop.instance()
    #    loop.add_callback(application.test)
    loop.start()


main()
