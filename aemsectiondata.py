
import numpy as np
from sklearn import preprocessing
import sklearn.decomposition as sd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import psycopg2
import cv2
from metric_learn import LMNN
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


from scipy.interpolate import interp1d

class AEMSectionData():

    def __init__(self, line):

        conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")
        cur = conn.cursor()


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

        cmd = """
            select x_coor, y_coor, dem_fill, value_type
            from aem.borehole 
            where x_coor::float < """ +str(self.maxeast+disrange) + """ 
            and x_coor::float > """ +str(self.mineast-disrange) + """ 
            and y_coor::float < """ +str(self.maxnorth+disrange) + """
            and y_coor::float > """ +str(self.minnorth-disrange)
        print(cmd)
        cur.execute(cmd)
        borehole = np.array(cur.fetchall()).astype(float)

        xy = np.array(list(zip(lineattribute['easting_albers'], lineattribute['northing_albers'])))
        # xy_borehole = list(zip(borehole['x_coor'], borehole['y_coor']))
        # xy_borehole = [tuple(map(float, element)) for element in xy_borehole]
        xy_borehole = borehole[:,0:2]

        cellsize = {'width': 20.0, 'height': 2.0}

        pca = sd.PCA(n_components=1)
        pca.fit(xy)
        w = pca.transform(xy)
        rect_w = ((w - min(w)) / cellsize['width']).round().astype(int)

        print(xy_borehole.shape)
        w_borehole = pca.transform(xy_borehole)
        rect_w_borehole = ((w_borehole - min(w)) / cellsize['width']).round().astype(int)

        h = [list(self.running_middle(element)) for element in lineattribute['thickness']]
        h_borehole =


        rect_h = np.array(h)
        for idx, ele in enumerate(lineattribute['elevation']):
            rect_h[idx] = ele - rect_h[idx]


        firstdepth = 0
        rect_h_max = rect_h.max()
        rect_h = rect_h_max - rect_h
        rect_h_min = rect_h.min()
        rect_h = ((rect_h - rect_h_min + firstdepth) / cellsize['height']).round().astype(int)

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
                height = surface + round((topwii - wii_min) / (wii_max - wii_min) * 100 / cellsize['height'])
                if height<im.shape[0]:
                    im[surface, x, 1] = topwii
                    im[height, x, 1] = wii_min

                im[surface, x, 2] = topgrav
                im[rect_h[idx][-1], x, 2] = topgrav

        for idx, x in enumerate(rect_w_borehole):



        immin = [im[:, :, i][np.nonzero(im[:, :, i])].min() for i in range(im.shape[2] - 1)]
        immax = [im[:, :, i][np.nonzero(im[:, :, i])].max() for i in range(im.shape[2] - 1)]

        displayim = np.zeros([im.shape[0], im.shape[1], 4])  # conductivity, wii, grav, borehole

        for i in range(3):
            tim = im[:, :, i] > 0
            displayim[:, :, i] = (im[:, :, i] - immin[i]) / (immax[i] - immin[i]) * 155 + 100
            displayim[:, :, i] = displayim[:, :, i] * tim.astype(float)

        print('interpolate')
        self.displayim = {}

        self.displayim['conductivity'] =self.geointerpolation(displayim[:, :, 0].squeeze().astype(np.uint8))
        self.displayim['wii'] = self.geointerpolation(displayim[:, :, 1].squeeze().astype(np.uint8))
        self.displayim['gravity'] = self.geointerpolation(displayim[:, :, 2].squeeze().astype(np.uint8))
        self.displayim['borehole'] = self.geointerpolation(displayim[:, :, 3].squeeze().astype(np.uint8))

        self.channellist = ['conductivity', 'wii', 'gravity', 'borehole']

        self.point = np.hstack([xy.reshape(-1,2), w.reshape(-1,1)])


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



    def getimagetopdown(self, width, height):
        c = np.linalg.norm(self.point[:,0:2] - self.point[0 ,0:2] ,axis=1)
        a = self.point[: ,2] - self.point[0 ,2]
        b = ( c** 2 - a**2 )**0.5

        maxa = a.max()
        mina = a.min()
        maxb = b.max()
        minb = b.min()

        a = ( a - mina ) / (maxa -mina ) *(self.width -1)
        b = ( b - minb ) / (maxa -mina ) *(self.width -1)
        a = a.astype(int)
        b = b.astype(int)

        height = b.max()+1
        print(self.width,height,'here')

        img = np.zeros([height, self.width, 3], dtype=float)
        img.fill(200)

        for i,j in zip(a,b):
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