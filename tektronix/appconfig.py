from configparser import ConfigParser


class AppConfig:
    config = None
    test = False

    def __new__(cls, config_path=None):
        if cls.config is None:

            if config_path is None:
                raise RuntimeError("config_path can't be None on first instantiation")

            cls.config = ConfigParser()
            cls.config.read(str(config_path))

        return cls.config
