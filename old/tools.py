import webbrowser
from read_data import NmeaFile


def open_gmaps(lon, lat):
    lon = NmeaFile._dd_to_dms(lon, ('E', 'W'))
    lat = NmeaFile._dd_to_dms(lat, ('N', 'S'))
    link = "www.google.com/maps/place/{}%C2%B0{}'{:.1f}%22{}+{}%C2%B0{}'{:.1f}%22{}".format(*lon, *lat)
    webbrowser.open_new_tab(link)
