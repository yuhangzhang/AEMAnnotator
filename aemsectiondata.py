
import numpy as np
from sklearn import preprocessing
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import psycopg2

from metric_learn import LMNN
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


from scipy.interpolate import interp1d


from aemlinestandardisation import AEMLineStandardisation

class AEMSectionData():

    def __init__(self, line):

        self.conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")
        cur = self.conn.cursor()


        cmd = """
            select 
                l.wii_albers, 
                l.ir_grav_al , 
                easting_albers,
                northing_albers,
                elevation,
                conductivity,
                thickness 
            from geoannotator.dbf as l 
            inner join aem.albers as r
            on l.line::float = r.line 
            and l.fiducial::float = r.fiducial
            where l.line::bigint = """+str(line)+""" and r.line = """+str(line)+ """ 
            order by l.fiducial::float
            """
        print(cmd)
        cur.execute(cmd)
        lineattribute = cur.fetchall()

        column_names = ['wii_albers', 'ir_grav_al', 'easting_albers', 'northing_albers', 'elevation', 'conductivity', \
                        'thickness']


        #lineattribute = {name:list(map([idx],lineattribute)) for idx, name in enumerate(column_names)}
        lineattribute = {name:np.array(list(map(lambda x: x[idx], lineattribute))).astype(float) for idx, name in enumerate(column_names)}

        print(lineattribute['easting_albers'][:10])
        self.maxeast = lineattribute['easting_albers'].max()
        self.maxnorth = max(lineattribute['northing_albers'])
        self.mineast = min(lineattribute['easting_albers'])
        self.minnorth = min(lineattribute['northing_albers'])

        print(self.maxeast, self.maxnorth, self.mineast, type(self.minnorth))

        disrange = 500 # borehole within this range will be considered



        cmd= """
            select x_coor, y_coor, dem_fill, round_a_de, 
            case when lag is null then '0' else lag end  as upper,
            value_type
            from
            (
            select x_coor, y_coor, round_a_de, lag(round_a_de,1) over(partition by borehole_n order by round_a_de::float) as lag,
                dem_fill, value_type
                from aem.borehole
                where x_coor::float < """ +str(self.maxeast+disrange) + """ 
                and x_coor::float > """ +str(self.mineast-disrange) + """ 
                and y_coor::float < """ +str(self.maxnorth+disrange) + """
                and y_coor::float > """ +str(self.minnorth-disrange) + """
            )tmp
        """


        print(cmd)
        cur.execute(cmd)
        borehole = cur.fetchall()

        column_names = ['x_coor', 'y_coor', 'dem_fill', 'round_a_de', 'upper', 'value_type']

        borehole = {name:np.array(list(map(lambda x: x[idx], borehole))).astype(float) for idx, name in enumerate(column_names)}


        xy = np.array(list(zip(lineattribute['easting_albers'], lineattribute['northing_albers'])))
        # xy_borehole = list(zip(borehole['x_coor'], borehole['y_coor']))
        # xy_borehole = [tuple(map(float, element)) for element in xy_borehole]
        xy_borehole =np.array(list(zip(borehole['x_coor'], borehole['y_coor'])))

        self.cellsize = {'width': 20.0, 'height': 2.0}

        self.transform = AEMLineStandardisation()
        self.transform.fit(xy)
        self.w = self.transform.transform(xy)/self.cellsize['width']
        rect_w = self.w[:,0].round().astype(int)

        self.w_borehole = self.transform.transform(xy_borehole)/self.cellsize['width']
        rect_w_borehole = self.w_borehole[:,0].round().astype(int)

        h = [list(self.running_middle(element)) for element in lineattribute['thickness']]

        rect_h = np.array(h)
        for idx, ele in enumerate(lineattribute['elevation']):
            rect_h[idx] = ele - rect_h[idx]


        firstdepth = 0
        rect_h_max = rect_h.max()
        rect_h = rect_h_max - rect_h
        rect_h_min = rect_h.min()
        rect_h = ((rect_h - rect_h_min + firstdepth) / self.cellsize['height']).round().astype(int)

        self.width = rect_w.max() + 1
        self.height = rect_h.max() + 1
        print(self.height,self.width,'size')
        im = np.zeros([self.height, self.width, 4])  # conductivity, wii, grav, borehole
        imcount = np.zeros([rect_h.max() + 1, rect_w.max() + 1, 4])  # conductivity, wii, grav, borehole

        #print('assign conductivity')
        conductivity = lineattribute['conductivity']
        for idx, x in enumerate(rect_w):
            for idy, y in enumerate(rect_h[idx]):
                im[y, x, 0] = im[y, x, 0] * imcount[y, x, 0] / (imcount[y, x, 0] + 1) + conductivity[idx][idy] / (
                            imcount[y, x, 0] + 1)
                imcount[y, x, 0] += 1

        wii_max = 5.98
        wii_min = 1
        wii = lineattribute['wii_albers']
        grav= lineattribute['ir_grav_al']
        for idx, x in enumerate(rect_w):
            topwii = float(wii[idx])
            topgrav = float(grav[idx])
            surface = rect_h[idx][0]
            if topwii > 0 and x > 0 and x < im.shape[1]:
                height = surface + round((topwii - wii_min) / (wii_max - wii_min) * 100 / self.cellsize['height'])
                if height<im.shape[0]:
                    im[surface, x, 1] = topwii
                    im[height, x, 1] = wii_min

                im[surface, x, 2] = topgrav
                im[rect_h[idx][-1], x, 2] = topgrav

        boreholewidth = 5
        for idx, x in enumerate(rect_w_borehole):
            y_up = borehole['dem_fill'][idx]-borehole['upper'][idx]
            y_up = rect_h_max - y_up
            y_up = round((y_up - rect_h_min + firstdepth) / self.cellsize['height'])
            y_up = int(y_up)
            if y_up >=  im.shape[0]:
                continue
            y_down = borehole['dem_fill'][idx]-borehole['round_a_de'][idx]
            y_down = rect_h_max - y_down
            y_down = round((y_down - rect_h_min + firstdepth) / self.cellsize['height'])
            y_down = int(y_down)
            if y_down >= im.shape[0]:
                y_down = im.shape[0]-1
            #im[y_up, x - boreholewidth:x + boreholewidth, 3] = 0
            im[y_up+1,x-boreholewidth:x+boreholewidth,3] = np.ceil(borehole['value_type'][idx])
            im[y_down, x-boreholewidth:x+boreholewidth, 3] = np.ceil(borehole['value_type'][idx])
            print(y_up,y_down,x)



        immin = [im[:, :, i][np.nonzero(im[:, :, i])].min() for i in range(im.shape[2])]
        immax = [im[:, :, i][np.nonzero(im[:, :, i])].max() for i in range(im.shape[2])]

        displayim = np.zeros([im.shape[0], im.shape[1], 4])  # conductivity, wii, grav, borehole

        for i in range(im.shape[2]):
            tim = im[:, :, i] > 0
            displayim[:, :, i] = (im[:, :, i] - immin[i]) / (immax[i] - immin[i]) * 155 + 100
            displayim[:, :, i] = displayim[:, :, i] * tim.astype(float)

        print('interpolate')
        self.displayim = {}

        self.displayim['conductivity'] =self.geointerpolation(displayim[:, :, 0].squeeze().astype(np.uint8))
        self.displayim['wii'] = self.geointerpolation(displayim[:, :, 1].squeeze().astype(np.uint8))
        self.displayim['gravity'] = self.geointerpolation(displayim[:, :, 2].squeeze().astype(np.uint8))
        self.displayim['borehole'] = np.zeros([im.shape[0], im.shape[1], 4])
        self.displayim['borehole'][:, :, 0] = self.geointerpolation(displayim[:, :, 3].squeeze())
        self.displayim['borehole'][:, :, 1] = self.displayim['borehole'][:, :, 0]
        self.displayim['borehole'][:, :, 3] = (self.displayim['borehole'][:, :, 1]>0)*255
        self.displayim['borehole'] = self.displayim['borehole'].astype(np.uint8)


        self.channellist = ['conductivity', 'wii', 'gravity', 'borehole']

        self.point = np.hstack([xy.reshape(-1,2), self.w[:,0].reshape(-1,1)])



    @staticmethod
    def running_middle(lst):
        tot = 0
        for item in lst:
            tot += item
            val = tot - item / 2
            yield val

    @staticmethod
    def geointerpolation(img):
        omg = np.zeros(img.shape)
        for idx, col in enumerate(img.T):
            nz = col.nonzero()[0]
            nzv = col[nz]
            if len(nz) > 0:
                omg[:, idx] = np.interp(list(range(len(col))), nz, nzv, left=0, right=0)

        for idx, row in enumerate(omg):
            nz = row.nonzero()[0]
            nzv = row[nz]
            if len(nz) > 0:
                row = np.interp(list(range(len(row))), nz, nzv, left=0, right=0)
        return omg


    def getaffinetransform(self):
        translation = np.zeros([3,3])
        translation[0, 0] = 1
        translation[0,2] = -self.point[0,2]
        translation[1,2] = 0
        translation[1, 1] = 1
        translation[2,2] = 1

        translation2 = np.zeros([3,3])
        translation2[0,2] = self.point[0,0]
        translation2[0, 0] = 1
        translation2[1,2] = self.point[0,1]
        translation2[1, 1] = 1
        translation2[2,2] = 1

        cur = self.conn.cursor()
        cur.execute("""
        select (ST_WorldToRasterCoord(rast, 
            """+str(self.point[0,0])+""", """+str(self.point[0,1])+"""
            )).*
        from geoannotator.resample_reference
        """)
        [x0, y0] = cur.fetchall()[0]
        cur.execute("""
        select (ST_WorldToRasterCoord(rast, 
            """+str(self.point[-1,0])+""", """+str(self.point[-1,1])+"""
            )).*
        from geoannotator.resample_reference
        """)
        [x1, y1] = cur.fetchall()[0]
        scaling = np.zeros([3,3])
        scaling[0,0] = (x1-x0)/(self.point[-1,0]-self.point[0,0])
        scaling[1,1] = (y1-y0)/(self.point[-1,1]-self.point[0,1])
        scaling[2,2] = 1


        translation3 = np.zeros([3,3])
        translation3[0,2] = x0
        translation3[0, 0] = 1
        translation3[1,2] = y0
        translation3[1, 1] = 1
        translation3[2,2] = 1

        print(self.point.shape)
        c = np.linalg.norm(self.point[-1,0:2] - self.point[0 ,0:2] )
        x1 = self.point[-1 ,2] - self.point[0 ,2]
        y1 = ( c** 2 - x1**2 )**0.5
        x1 = x1 / (x1 ** 2 + y1 ** 2) ** 0.5
        y1 = y1 / (x1 ** 2 + y1 ** 2) ** 0.5

        x0 = self.point[-1,0]-self.point[0,0]
        y0 = self.point[-1,1]-self.point[0,1]
        x0 = x0 / (x0 ** 2 + y0 ** 2) ** 0.5
        y0 = y0 / (x0 ** 2 + y0 ** 2) ** 0.5


        cos = x0*x1+y0*y1
        sin = x1*y0-x0*y1
        rotation = np.zeros([3,3])
        rotation[0,0] = cos
        rotation[0,1] = -sin
        rotation[1,0] = sin
        rotation[1,1] = cos
        rotation[2,2] = 1

        print(cos, sin, "cos and sin")

        #return np.matmul(translation2, np.matmul(rotation, translation))
        return np.matmul(translation3, np.matmul(scaling, np.matmul(rotation, translation)))
        #return np.matmul(translation3, np.matmul(scaling, np.matmul(translation2, np.matmul(rotation, translation))))

        #return np.matmul(translation3, np.matmul(scaling, np.matmul(rotation, translation)))


    def getimagetopdown(self):
        maxx = int(self.w[:,0].max())
        miny = int(self.w[:,1].min())
        maxy = int(self.w[:,1].max())

        img = self.transform.cropimage(0, miny, maxx+1, maxy-miny+1, self.cellsize['width'])

        print(img[0:10,0:10,:],'img')

        for i,j in self.w:
            i = int(i)
            j = int(j)-miny
            img[j:j+2, i:i+2, :] = [255, 0, 0]

        print("td size", img.shape)
        return img.astype(np.uint8)

    # visualise the raw points as an RGB image of specified size, each pixel may correspond to multiple points
    def getimageunderground(self, channel):
        if channel is not None:
            print("section size", self.displayim[channel].shape)
            return self.displayim[channel]

    def getlayernames(self):
        return list(self.displayim)