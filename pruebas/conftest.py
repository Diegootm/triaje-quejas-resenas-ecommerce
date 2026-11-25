"""Configuración común de pytest."""


def pytest_configure(config):
    # joblib/loky cuenta los núcleos físicos con `wmic`, que Windows 11 ya no trae, y
    # emite este aviso. Es inofensivo (usa los núcleos lógicos). Se filtra solo ese mensaje.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Could not find the number of physical cores:UserWarning",
    )
