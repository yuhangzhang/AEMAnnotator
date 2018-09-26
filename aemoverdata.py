from io import BytesIO

from PIL import Image
from PIL import ImageQt

import psycopg2
import numpy as np


class AEMOverData():
    def __init__(self):
        # connect to local postgres db, enable postgis extension
        self.conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")
        self.cur = self.conn.cursor()
        self.cur.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL'")


    def getOverview(self):
        self.cur.execute("select st_aspng(rast) from geoannotator.baseimg")
        baseimg = self.cur.fetchall()
        baseimg = BytesIO(baseimg[0][0])
        baseimg = Image.open(baseimg)

        # self.cur.execute(" \
        #     select distinct \
        #             (ST_WorldToRasterCoord(rast, easting_albers::float, northing_albers::float)).*, line \
        #     from geoannotator.resample_reference, \
        #     ( \
        #         select distinct easting_albers, northing_albers, line from geoannotator.aem_albers \
        #     ) as tmp \
        # ")
        self.cur.execute("select * from geoannotator.trace order by columnx, rowy")
        trace = np.array(self.cur.fetchall())

        self.cur.execute(" \
            select distinct \
                (ST_WorldToRasterCoord(rast, x_coor::float, y_coor::float)).* \
            from geoannotator.resample_reference, geoannotator.eggs_dbf \
        ")
        borehole = np.array(self.cur.fetchall())

        return baseimg, trace, borehole

