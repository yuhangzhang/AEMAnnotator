import numpy as np
import psycopg2
from io import BytesIO
from PIL import Image

# given a set of points in 2D, translate and rotate their coordinates such that the first point is at the origin
# and the last point is at (0, d) where d is the distance between the first and the last point
class AEMLineStandardisation():
    def __init__(self):
        pass

    def fit(self, inputpoint):

        translation = np.eye(3)
        translation[0:2,2] = -inputpoint[0,:]

        reversetranslation = np.eye(3)
        reversetranslation[0:2, 2] = inputpoint[0, :]

        x0 = inputpoint[0,0]
        y0 = inputpoint[0,1]
        x1 = inputpoint[-1,0]
        y1 = inputpoint[-1,1]

        d = np.sqrt((y1-y0)**2+(x1-x0)**2)

        x0 = x1 - x0
        y0 = y1 - y0
        x1 = d
        y1 = 0.0

        cosZ = (x0*x1+y0*y1)/(x0*x0+y0*y0)
        sinZ = (x0*y1-x1*y0)/(x0*x0+y0*y0)

        rotation = np.eye(3)
        rotation[0,0] = cosZ
        rotation[0,1] = -sinZ
        rotation[1,0] = sinZ
        rotation[1,1] = cosZ

        self.T = np.matmul(rotation,translation)
        self.reverseT = np.matmul(reversetranslation, rotation.T)

    def transform(self, point):
        point = np.vstack([point.T, np.ones([1,point.shape[0]])])
        return np.matmul(self.T, point).T[:,0:2]

    def inversetransform(self, x, y):
        #print(x,y,np.matmul(self.reverseT, [x,y,1.0])[0:2],'inverse' )
        return np.matmul(self.reverseT, [x,y,1.0])[0:2]

    def cropimage(self, leftx, topy, width, height, cellsize):
        conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")
        cur = conn.cursor()
        cur.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL'")

        cmd="""select 
                st_upperleftx(rast) as leftx, 
                st_upperlefty(rast) as topy,                
                ST_PixelWidth(rast) as pixelwidth,
                ST_PixelHeight(rast) as pixelheight
                from geoannotator.baseimg
        """
        cur.execute(cmd)
        info = cur.fetchall()

        cmd = """select st_aspng(rast) from geoannotator.baseimg"""
        cur.execute(cmd)
        baseimg = cur.fetchall()
        baseimg = BytesIO(baseimg[0][0])
        baseimg = Image.open(baseimg)
        print(baseimg.size, "baseimg")
        baseimg = baseimg.load()

        cmd = """select ST_RasterToWorldCoordX(rast, """+ str(leftx) +"""), ST_RasterToWorldCoordY(rast, """+ str(topy) +""") from  geoannotator.baseimg"""
        cur.execute(cmd)
        corner = cur.fetchall()


        crop = np.zeros([height, width, 3], dtype=np.uint8)


        for i in range(width):
            for j in range(height):
                x,y = self.inversetransform(i*cellsize,j*cellsize)
                #print(x,y,"inverse")
                x = (x-info[0][0])/info[0][2]
                y = (y-info[0][1])/info[0][3]
                #print(x,width,y,height)
                x = round(x)
                y = round(y)
                #if x>=0 and x<width and y>=0 and y<height:
                #print(baseimg[x,y])
                #print(type(baseimg[x,y][0]),"type")
                #print(i,j,x,y)
                crop[j,i,:] = np.array(baseimg[x,y]).astype(np.uint8)


        return crop