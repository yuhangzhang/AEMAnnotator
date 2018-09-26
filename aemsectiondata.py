
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


class AEMSectionData():

    def __init__(self, line):

        conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")
        cur = conn.cursor()

        cmd = """with tmp as
                (
                select 
                    fiducial,
                    easting_albers, 
                    northing_albers, 
                    conductivity, 
                    x::float as thickness, 
                    y as ordinal, 
                    elevation::float as elevation
                from 
                    aem_albers as a, 
                    unnest(string_to_array(regexp_replace(a.thickness,'[{}]','','g'),',')) with ordinality  as b(x,y) 
                    where line = '""" + str(line) + """'
                ),
                tmp2 as
                (
                    select 
                        fiducial,
                        easting_albers, 
                        northing_albers, 
                        conductivity, 
                        sum(thickness) over(partition by fiducial order by ordinal) as thickness, 
                        ordinal, 
                        elevation::float as elevation
                    from 
                        tmp
                )
                select 
                    easting_albers, 
                    northing_albers,
                    elevation-thickness as z,
                    (string_to_array(regexp_replace(conductivity,'[{}]','','g'),','))[ordinal] as conductivity1, 
                    (string_to_array(regexp_replace(conductivity,'[{}]','','g'),','))[ordinal] as conductivity2, 
                    (string_to_array(regexp_replace(conductivity,'[{}]','','g'),','))[ordinal] as conductivity3 
                from  
                    tmp2"""


        cmd = "select easting_albers, northing_albers, elevation, conductivity, conductivity, conductivity\
            from geoannotator.point_by_line where line='"+str(line)+"'"

        cur.execute(cmd)
        inputrecord = np.array(cur.fetchall())

        xyz = inputrecord[:, 0:3].astype(float)  # nx3 matrix
        pca = sd.PCA(n_components=1)
        pca.fit(xyz[:, 0:2])
        xy = pca.transform(xyz[:, 0:2])

        self.point = np.hstack([inputrecord, xy.reshape(-1, 1), inputrecord[:, 2].reshape(-1, 1)])
        self.minh = self.point[:, -1].astype(float).min()
        self.minw = self.point[:, -2].astype(float).min()
        self.maxh = self.point[:, -1].astype(float).max()
        self.maxw = self.point[:, -2].astype(float).max()




        self.feature = np.hstack([self.point[: ,3:6],  self.point[: ,2].reshape(-1, 1)])

        self.whitener = preprocessing.StandardScaler().fit(self.feature)
        self.feature = self.whitener.transform(self.feature)

        self.manuallabel = np.zeros([0 ,0])

    def getimagetopdown(self, width, height):


        c = self.point[:, 0:2].astype(float)
        c = np.linalg.norm(c - c[0 ,:] ,axis=1)
        a = self.point[: ,-2].astype(float ) -float(self.point[0 ,-2])
        b = ( c** 2 - a**2 )**0.5

        maxa = a.max()
        mina = a.min()
        maxb = b.max()
        minb = b.min()
        print(maxa, mina, maxb, minb)

        a = ( a - mina ) / (maxa -mina ) *(width -1)
        b = ( b - minb ) / (maxa -mina ) *(width -1)
        a = a.astype(int)
        b = b.astype(int)

        height = b.max()+1

        img = np.zeros([height, width, 3], dtype=float)
        img.fill(200)

        for i in range(len(a)):
            img[b[i]:b[i ] +2 ,a[i]:a[i ] +2 ,: ] =[255 ,0 ,0]

        return img.astype(np.uint8)

    # visualise the raw points as an RGB image of specified size, each pixel may correspond to multiple points
    def getimageunderground(self, width, height):
        self.widthunderground = width
        self.heightunderground = height

        self.manuallabel = np.zeros([height, width], dtype=int)

        img = np.zeros([height, width, 3], dtype=float)

        self.bucket = [[[] for h in range(height)] for w in range(width)]
        self.featurebucket = [[[] for h in range(height)] for w in range(width)]

        # update point on-image coordinates according to image width and height
        self.point[:, -1] = np.round(
            (self.point[:, -1].astype(float) - self.minh) / (self.maxh - self.minh + 1) * (height - 1)
        )
        self.point[:, -2] = np.round(
            (self.point[:, -2].astype(float) - self.minw) / (self.maxw - self.minw + 1) * (width - 1)
        )

        bucket_size = np.zeros([height, width, 3], dtype=np.int)

        # update image
        for i in range(len(self.point)):
            p = self.point[i]
            if 1== 1:
                inversedh = height - int(float(p[-1])) - 1
                originalw = int(float(p[-2]))
                self.bucket[originalw][inversedh].append(p)
                self.featurebucket[originalw][inversedh].append(self.feature[i])
                img[inversedh, originalw, 0] = img[inversedh, originalw, 0] + float(p[3])
                img[inversedh, originalw, 1] = img[inversedh, originalw, 1] + float(p[4])
                img[inversedh, originalw, 2] = img[inversedh, originalw, 2] + float(p[5])
                bucket_size[inversedh, originalw, :] = bucket_size[inversedh, originalw, :] + 1
                #print(inversedh, originalw, p[3])
                #time.sleep(0.5)

        # if a pixel corresponds to no point, make its size=1 for the next step
        for i in range(bucket_size.shape[0]):
            for j in range(bucket_size.shape[1]):
                for k in range(bucket_size.shape[2]):
                    if bucket_size[i, j, k] == 0:
                        bucket_size[i, j, k] = 1

        # the colour of each pixel is the averge colour of all its points
        img = img / bucket_size

        # for better visualisation, we use 3/4 of the colour spectrum to visualise point variance
        # the other 1/4 is used to separate points from no points
        minr = img[:, :, 0][np.nonzero(img[:, :, 0])].min()
        ming = img[:, :, 1][np.nonzero(img[:, :, 0])].min()
        minb = img[:, :, 2][np.nonzero(img[:, :, 0])].min()
        maxr = img[:, :, 0].max()
        maxg = img[:, :, 1].max()
        maxb = img[:, :, 2].max()

        img[:, :, 0][np.nonzero(img[:, :, 0])] = img[:, :, 0][np.nonzero(img[:, :, 0])] + (maxr - minr) * 0.25
        img[:, :, 1][np.nonzero(img[:, :, 1])] = img[:, :, 1][np.nonzero(img[:, :, 1])] + (maxg - ming) * 0.25
        img[:, :, 2][np.nonzero(img[:, :, 2])] = img[:, :, 2][np.nonzero(img[:, :, 2])] + (maxb - minb) * 0.25

        # fill in pixels which has no points
        # linear intepolation along vertical direction
        img_interpolate = self.geointerpolation(img)

        # all pixel is coloured by three uint8 integers between 0 and 255.
        img_interpolate[:, :, 0] = (img_interpolate[:, :, 0]) * 255 / (maxr + (maxr - minr) * 0.25)
        img_interpolate[:, :, 1] = (img_interpolate[:, :, 1]) * 255 / (maxg + (maxg - ming) * 0.25)
        img_interpolate[:, :, 2] = (img_interpolate[:, :, 2]) * 255 / (maxb + (maxb - minb) * 0.25)

        cmap = cm.ScalarMappable(colors.Normalize(
            vmin=(minr + (maxr - minr) * 0.25) / (maxr + (maxr - minr) * 0.25) * 255,
            vmax=255), cmap=plt.get_cmap('jet')
        )

        img_interpolate = cmap.to_rgba(img_interpolate[:, :, 0])[:, :, 0:3] * (img_interpolate > 0) * 255

        cv2.imwrite("ttt.png", img_interpolate.astype(np.uint8))

        #return(img.astype(np.uint8))
        return (img_interpolate.astype(np.uint8))

    # return the raw points corresponding to a pixel in the image as specified by w and h
    # should only be called after calling getimage()
    def getpoint(self, w, h):
        return self.bucket[w][h]

    def getfeature(self, w, h):
        return self.featurebucket[w][h]

    def get_annotated_point(self):
        point = []
        label = []
        for h in range(self.manuallabel.shape[0]):
            for w in range(self.manuallabel.shape[1]):
                if self.manuallabel[h, w] > 0:
                    morepoints = self.getpoint(w, h)
                    point.extend(morepoints)
                    label.extend([self.manuallabel[h, w]] * len(morepoints))
        return point, label

    def get_annotated_feature(self):
        feature = []
        label = []
        for h in range(self.manuallabel.shape[0]):
            for w in range(self.manuallabel.shape[1]):
                if self.manuallabel[h, w] > 0:
                    morefeature = self.getfeature(w, h)
                    feature.extend(morefeature)
                    label.extend([self.manuallabel[h, w]] * len(morefeature))
        return feature, label

    def get_prediction(self, model):
        self.model = model
        X, y = self.get_annotated_feature()

        self.model.fit(np.array(X), y)
        self.prediction = self.model.predict(self.feature)

        model = KNeighborsClassifier()
        model.fit(X, y)
        self.prediction = model.predict(self.feature)

        img = np.zeros([self.heightunderground, self.widthunderground, 4], dtype=np.uint8)

        for i in range(len(self.point)):
            p = self.point[i]
            if 1 == 1:
                inversedh = self.heightunderground - int(float(p[-1])) - 1
                originalw = int(float(p[-2]))
                img[inversedh, originalw, 0] = self.prediction[i]

        img_interpolate = self.geointerpolation_label(img).round().astype(int)

        img = np.zeros([self.heightunderground, self.widthunderground, 4], dtype=np.uint8)

        for i in range(img_interpolate.shape[0]):
            for j in range(img_interpolate.shape[1]):
                if img_interpolate[i, j, 0] > 0:
                    img[i, j, 0:3] = [c * 255 for c in plt.get_cmap('tab10').colors[img_interpolate[i, j, 0] - 1]]

        return img

    @staticmethod
    def geointerpolation(img):
        height = img.shape[0]
        width = img.shape[1]
        if len(img.shape) == 3:
            depth = img.shape[2]
        else:
            depth = 1

        # fill in pixels which has no points
        # linear intepolation along vertical direction
        updis = np.zeros([height, width], dtype=np.int)
        img_interpolate = np.zeros([height, width, depth], dtype=np.float32)

        vboundary = np.zeros([2, width], dtype=np.int)
        vboundary[1, :] = height

        for w in range(img.shape[1]):
            dis = 0
            lastpoint = img[0, w, :]
            for h in range(img.shape[0]):
                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0:
                    dis = dis + 1
                    img_interpolate[h, w, :] = lastpoint
                    updis[h, w] = dis
                else:
                    img_interpolate[h, w, :] = img[h, w, :]
                    dis = 0
                    lastpoint = img[h, w, :]
                    if vboundary[0, w] < h:
                        vboundary[0, w] = h

        for w in range(img.shape[1]):
            dis = 0
            lastpoint = img[-1, w, :]
            for h in reversed(range(img.shape[0])):
                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0 and abs(
                        img_interpolate[h, w, 0]) + abs(img_interpolate[h, w, 1]) + abs(img_interpolate[h, w, 2]) > 0:
                    if abs(lastpoint[0]) + abs(lastpoint[1]) + abs(lastpoint[2]) == 0:
                        img_interpolate[h, w, :] = lastpoint
                    else:
                        img_interpolate[h, w, :] = (img_interpolate[h, w, :] * dis + lastpoint * updis[h, w]) / (
                                dis + updis[h, w])
                    dis = dis + 1
                elif abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0:
                    break
                else:
                    dis = 0
                    lastpoint = img[h, w, :]
                    if vboundary[1, w] > h:
                        vboundary[1, w] = h

        img = img_interpolate
        img_interpolate = np.zeros([height, width, depth], dtype=np.float32)

        # linear intepolation along horizontal direction
        for h in range(img.shape[0]):
            dis = 0
            lastpoint = img[h, 0, :]
            for w in range(img.shape[1]):

                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0 and h <= vboundary[1, w] and h >= \
                        vboundary[0, w]:
                    dis = dis + 1
                    img_interpolate[h, w, :] = lastpoint
                    updis[h, w] = dis
                else:
                    img_interpolate[h, w, :] = img[h, w, :]
                    dis = 0
                    lastpoint = img[h, w, :]

        for h in range(img.shape[0]):
            dis = 0
            dis = 0
            lastpoint = img[h, -1, :]
            for w in reversed(range(img_interpolate.shape[1])):
                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0 and abs(
                        img_interpolate[h, w, 0]) + abs(img_interpolate[h, w, 1]) + abs(img_interpolate[h, w, 2]) > 0:
                    if abs(lastpoint[0]) + abs(lastpoint[1]) + abs(lastpoint[2]) == 0:
                        img_interpolate[h, w, :] = lastpoint
                    elif h <= vboundary[1, w] and h >= vboundary[0, w]:
                        img_interpolate[h, w, :] = (img_interpolate[h, w, :] * dis + lastpoint * updis[h, w]) / (
                                dis + updis[h, w])
                    else:
                        img_interpolate[h, w, :] = 0
                    dis = dis + 1
                elif abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0:
                    break
                else:
                    dis = 0
                    lastpoint = img[h, w, :]

        return np.squeeze(img_interpolate)

    @staticmethod
    def geointerpolation_label(img):
        height = img.shape[0]
        width = img.shape[1]
        if len(img.shape) == 3:
            depth = img.shape[2]
        else:
            depth = 1

        # fill in pixels which has no points
        # linear intepolation along vertical direction
        updis = np.zeros([height, width], dtype=np.int)
        img_interpolate = np.zeros([height, width, depth], dtype=np.float32)

        vboundary = np.zeros([2, width], dtype=np.int)
        vboundary[1, :] = height

        for w in range(img.shape[1]):
            dis = 0
            lastpoint = img[0, w, :]
            for h in range(img.shape[0]):
                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0:
                    dis = dis + 1
                    img_interpolate[h, w, :] = lastpoint
                    updis[h, w] = dis
                else:
                    img_interpolate[h, w, :] = img[h, w, :]
                    dis = 0
                    lastpoint = img[h, w, :]
                    if vboundary[0, w] < h:
                        vboundary[0, w] = h

        for w in range(img.shape[1]):
            dis = 0
            lastpoint = img[-1, w, :]
            for h in reversed(range(img.shape[0])):
                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0 and abs(
                        img_interpolate[h, w, 0]) + abs(img_interpolate[h, w, 1]) + abs(img_interpolate[h, w, 2]) > 0:
                    if abs(lastpoint[0]) + abs(lastpoint[1]) + abs(lastpoint[2]) == 0:
                        img_interpolate[h, w, :] = lastpoint
                    else:
                        if dis < updis[h, w]:
                            img_interpolate[h, w, :] = lastpoint
                    dis = dis + 1
                elif abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0:
                    break
                else:
                    dis = 0
                    lastpoint = img[h, w, :]
                    if vboundary[1, w] > h:
                        vboundary[1, w] = h

        img = img_interpolate
        img_interpolate = np.zeros([height, width, depth], dtype=np.float32)

        # linear intepolation along horizontal direction
        for h in range(img.shape[0]):
            dis = 0
            lastpoint = img[h, 0, :]
            for w in range(img.shape[1]):

                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0 and h <= vboundary[1, w] and h >= \
                        vboundary[0, w]:
                    dis = dis + 1
                    img_interpolate[h, w, :] = lastpoint
                    updis[h, w] = dis
                else:
                    img_interpolate[h, w, :] = img[h, w, :]
                    dis = 0
                    lastpoint = img[h, w, :]

        for h in range(img.shape[0]):
            dis = 0
            dis = 0
            lastpoint = img[h, -1, :]
            for w in reversed(range(img_interpolate.shape[1])):
                if abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0 and abs(
                        img_interpolate[h, w, 0]) + abs(img_interpolate[h, w, 1]) + abs(img_interpolate[h, w, 2]) > 0:
                    if abs(lastpoint[0]) + abs(lastpoint[1]) + abs(lastpoint[2]) == 0:
                        img_interpolate[h, w, :] = lastpoint
                    elif h <= vboundary[1, w] and h >= vboundary[0, w]:
                        if dis < updis[h, w]:
                            img_interpolate[h, w, :] = lastpoint
                    else:
                        img_interpolate[h, w, :] = 0
                    dis = dis + 1
                elif abs(img[h, w, 0]) + abs(img[h, w, 1]) + abs(img[h, w, 2]) == 0:
                    break
                else:
                    dis = 0
                    lastpoint = img[h, w, :]

        return np.squeeze(img_interpolate)
