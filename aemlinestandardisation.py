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
        print(inputpoint[0],"input")

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

        print(np.matmul(self.T, np.append(inputpoint[0,:],1)),"heer")
        print(np.matmul(self.T, np.append(inputpoint[-1,:],1)),"heer")

    def transform(self, point):
        point = np.vstack([point.T, np.ones([1,point.shape[0]])])
        return np.matmul(self.T, point).T[:,0:2]

    def inversetransform(self, x, y):
        #print(x,y,np.matmul(self.reverseT, [x,y,1.0])[0:2],'inverse' )
        return np.matmul(self.reverseT, [x,y,1.0])[0:2]

    def cropimage(self, minx, miny, maxx, maxy, cellsize):
        width = int((maxx-minx+1.0)/cellsize)
        height = int((maxy-miny+1.0)/cellsize)

        conn = psycopg2.connect(host="localhost", database="DataLake", user="yuhang")
        cur = conn.cursor()
        cur.execute("SET postgis.gdal_enabled_drivers = 'ENABLE_ALL'")

        x1, y1 = self.inversetransform(minx,miny)
        x2, y2 = self.inversetransform(maxx,maxy)
        x3, y3 = self.inversetransform(minx,maxy)
        x4, y4 = self.inversetransform(maxx,miny)

        tmp = [x1,x2,x3,x4]
        xmin = min(tmp)
        xmax = max(tmp)
        tmp = [y1,y2,y3,y4]
        ymin = min(tmp)
        ymax = max(tmp)

        cmd="""select 
                st_upperleftx(rast) as leftx, 
                st_upperlefty(rast) as topy,                
                ST_PixelWidth(rast) as pixelwidth,
                ST_PixelHeight(rast) as pixelheight,
                st_aspng(rast) as png
                from 
                (
                    select st_union(rast) as rast
                    from geoannotator.satellite_tif
                    where   st_upperleftx(rast)<"""+str(xmax)+"""+1000
                    and     st_upperlefty(rast)<"""+str(ymax)+"""+1000
                    and     st_upperleftx(rast)+st_pixelwidth(rast)*st_width(rast)>"""+str(xmin)+"""-1000
                    and     st_upperlefty(rast)+st_pixelheight(rast)*st_height(rast)>"""+str(ymin)+"""-1000
                ) as tmp
        """
        print(cmd)
        cur.execute(cmd)
        info = cur.fetchall()


        baseimg = info[0][4]
        baseimg = BytesIO(baseimg)
        baseimg = Image.open(baseimg)
        baseimg.show()
        print(baseimg.size, "baseimg")
        baseimg = baseimg.load()

        crop = np.zeros([height, width, 3], dtype=np.uint8)

        x, y = self.inversetransform(0,0)

        for i in range(width):
            #print(i,width)
            for j in range(height):
                i2 = i*cellsize+minx
                j2 = j*cellsize+miny
                x,y = self.inversetransform(i2,j2)
                #print(x,y,"raw")
                # print(info[0:4],"info")
                x = (x-info[0][0])/info[0][2]
                y = (y-info[0][1])/info[0][3]

                x = round(np.fabs(x))
                y = round(np.fabs(y))
                if x>width or x<0:
                    continue
                if y>height or y<0:
                    continue

                crop[j,i,:] = np.array(baseimg[x,y]).astype(np.uint8)


        return crop