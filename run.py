
# this needs to go first to initialize the
# config stuff.
from tektronix.appconfig import AppConfig
config = AppConfig('config.ini')

from tektronix.api import FigureContainer
from tektronix.api import MyApplication
import tornado




def main():

    figs = FigureContainer()
    figs.add_figure("fig1", figsize=(6, 3.0))
    figs.add_figure("fig2", figsize=(6, 3.0))


    application = MyApplication(figs)
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    loop = tornado.ioloop.IOLoop.instance()
    #    loop.add_callback(application.test)
    loop.start()

if __name__ == '__main__':
    main()
