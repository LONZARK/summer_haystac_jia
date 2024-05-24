import pymap3d

print('USING KNOXVILLE COORDINATES FOR ENU')
# central point in knoxville
_lat0 = 35.960443
_lon0 = -83.921263


def get_enu_from_ll(lat, lon):
    """This function takes a lat/lon and returns an East/North combo
    we are assuming height of zero

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    """
    en = pymap3d.enu.geodetic2enu(lat, lon, 0, _lat0, _lon0, 0)
    return en[:2]
