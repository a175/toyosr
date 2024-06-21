from PIL import Image
import sys
import cv2
import math
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy
import pdf2image


class OneCharRecognizer:
    def __init__(self):
        self.model_file = 'dnn/mnist_100.onnx'
        self.net = None

    def detect_char(self,box_img):
        if self.net == None:
            self.net = cv2.dnn.readNetFromONNX(self.model_file)
        target_img = cv2.resize(box_img,(28,28))
        target_img = cv2.bitwise_not(target_img)
        target_img =  cv2.dnn.blobFromImage(target_img)
        self.net.setInput(target_img)
        pred = numpy.squeeze(self.net.forward())
        ind = numpy.argsort(pred)
        return int(ind[-1])

class OSRbase:
    def __init__(self):
        self.ocr = OneCharRecognizer()

    def detect_position_markers(self,frame):
        """
        returns pair of dict D and list L, D[k1][k2] is rect of qrcode for problem k1 at k2, L is list of strings for all qrcodes. 
        """
        value = decode(frame, symbols=[ZBarSymbol.QRCODE])
        position_markers = {}
        all_strings = []
        if value:
            for qrcode in value:
                key = qrcode.data.decode('utf-8')
                all_strings.append(key)
                k=key
                position_markers[k]={}
                position_markers[k]['x']=qrcode.rect
        all_strings.sort()
        return (position_markers,all_strings)
    
    def detect_char(self,box_img):
        return self.ocr.detect_char(box_img)

    def get_preprocessed_box_img(self,box_img):
        neiborhood = numpy.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],numpy.uint8)
        preprocessed = cv2.dilate(box_img,neiborhood,iterations=1)
        # 収縮
        preprocessed = cv2.erode(preprocessed,neiborhood,iterations=2)
        # 膨張
        preprocessed = cv2.dilate(preprocessed,neiborhood,iterations=2)
        # 収縮
        preprocessed = cv2.erode(preprocessed,neiborhood,iterations=2)
        # 膨張
        preprocessed = cv2.dilate(preprocessed,neiborhood,iterations=2)
        preprocessed = cv2.erode(preprocessed,neiborhood,iterations=2)
        return preprocessed
    
    def get_inside_box_coordinates(self,box_img):
        s = numpy.sum(box_img,axis=1)
        av = numpy.average(s)
        top_l = min( i for i,si in enumerate(s) if si < av)
        top_l = min( i for i,si in enumerate(s) if si > av and top_l < i)
        bottom_l = max( i for i,si in enumerate(s) if si < av)
        bottom_l = max( i for i,si in enumerate(s) if si > av and bottom_l > i)
        box_img = box_img[top_l:bottom_l,:]

        s = numpy.sum(box_img,axis=0)
        av = numpy.average(s)
        left_l = min( i for i,si in enumerate(s) if si < av)
        left_l = min( i for i,si in enumerate(s) if si > av and left_l < i)
        right_l = max( i for i,si in enumerate(s) if si < av)
        right_l = max( i for i,si in enumerate(s) if si > av and right_l > i)

        return (left_l,right_l,top_l,bottom_l)
    
    def get_box_coordinates_h(self,line_img,width):
        s = numpy.sum(line_img,axis=1)
        av = numpy.average(s)
        av2 = numpy.average([si for si in s if si < av])
        index = [ i for i,si in enumerate(s) if si < av2]
        topline_img = line_img[index,:]
        s = numpy.sum(topline_img,axis=0)
        av = numpy.average(s)
        index = [ i for i,si in enumerate(s) if si < av]
        seg = []
        left = None
        right = None
        for i in index:
            if left == None:
                left = i
                right = i
            elif right+1 == i:
                right = i
            else:
                seg.append((left,right))
                left = None
        if left != None:
            seg.append((left,right))
        ans = [s for s in seg if 1.2*width > s[1]-s[0] > 0.8*width]
        print(len(ans),len(seg))
        return ans

    def get_line_coordinates(self,position_markers):
        done=[]
        ans=[]
        for k in position_markers.keys():
            if k in done:
                continue
            (k1,k2)=self.get_key_in_the_same_line(k)
            done.append(k1)
            done.append(k2)
            if k1 == None:
                continue
            if k1 in position_markers:
                w=position_markers[k1]["x"].width
                h=position_markers[k1]["x"].height
                left=w+position_markers[k1]["x"].left
                top=position_markers[k1]["x"].top
                bottom=h+top
                width = max(w,h)
            else:
                left=0
                top=None
                bottom=None
                width = None
            if k2 in position_markers:
                right=position_markers[k2]["x"].left
                y1=position_markers[k2]["x"].top
                if top == None:
                    top = y1
                if top > y1:
                    top = y1
                h=position_markers[k2]["x"].height
                y1 = y1 + h
                if bottom == None:
                    bottom = y1
                if bottom < y1:
                    bottom = y1
                w=position_markers[k2]["x"].width
                w = max(w,h)
                if width == None:
                    width = w
                if width < w:
                    width = w

            else:
                right=-1
            ans.append((top,bottom,left,right,width))
        ans.sort()
        return ans

    def reduct_noise(self,frame):
        # 近傍の定義
        neiborhood = numpy.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],numpy.uint8)
        # 収縮
        img_erode = cv2.erode(frame,neiborhood,iterations=2)
        # 膨張
        img_dilate = cv2.dilate(img_erode,neiborhood,iterations=4)
        # 収縮
        img_erode = cv2.erode(img_dilate,neiborhood,iterations=4)
        # 膨張
        img_dilate = cv2.dilate(img_erode,neiborhood,iterations=2)
        return img_dilate


    def get_key_in_the_same_line(self,key):
        key_in_the_same_line = [('1','2'),('3','4')]
        for p in key_in_the_same_line:
            if key in p:
                return p
        return (None,None)

    def detect_angle(self,frame):
        """
        M
        returns degrees of angle of image if position marker is detected; otherwise None.  the range of degree is depend on the range of output of math.atan2.
        INPUT:
        frame - image
        """
        value = decode(frame, symbols=[ZBarSymbol.QRCODE])
        if value:
            data={}
            for qrcode in value:
                key = qrcode.data.decode('utf-8')
                x_av=0
                y_av=0
                for (x,y) in qrcode.polygon:
                    x_av=x_av+x
                    y_av=y_av+y
                x_av = x_av / len(qrcode.polygon)
                y_av = y_av / len(qrcode.polygon)
                #x, y, w, h = qrcode.rect
                data[key]=((x_av,y_av),qrcode.polygon)
            west_x = 0
            west_y = 0
            east_x = 0
            east_y = 0
            for key in data.keys():
                (k1,k2) = self.get_key_in_the_same_line(key)
                if k1 in data and k2 in data:
                    west_x = data[k1][0][0]
                    west_y = data[k1][0][1]
                    east_x = data[k2][0][0]
                    east_y = data[k2][0][1]
            if east_y-west_y == 0 and east_x-east_y == 0:
                return None
            ans= math.degrees(math.atan2(east_y-west_y,east_x-east_y))
            print(ans)
            return ans
        else:
            return None


class InteractiveOSR(OSRbase):
    def __init__(self):
        super().__init__()
    
    def detect_with_gui(self):
        pass

    def get_detected_answers_for_questions_as_csv_lines(self):
        ans = []
        (answers,strings) = self.get_detected_answers_for_questions()
        s=",".join([si for si in strings if not si.startswith("marker:")])
        for questionid in answers.keys():
            m=",".join(["&".join(qi) for qi in answers[questionid]])
            d=questionid+','+m+','+s
            ans.append(d)
        return ans

    def get_detected_answers_for_questions(self):
        strings = self.detected_strings[:]
        strings.sort()
        ans = {}
        for questionid in self.questions.keys():
            if questionid not in self.detected_data:
                continue
            marked_keys=self.detected_data[questionid]
            ans[questionid]=[[a for (k,a) in qi if k in marked_keys] for qi in self.questions[questionid]]
        return(ans,strings)

    def modify_angle(self,frame,default_rotation_mat,default_rotaion_90):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        degrees=self.detect_angle(img_gray)
        if degrees != None:
            if (45 < degrees % 180 and degrees % 180  < 135) :
                needs_rotate_90=True
                degrees=degrees+90
                (w,h)=img_gray.shape
            else:
                needs_rotate_90=False
                (h,w)=img_gray.shape
            rotation_mat = cv2.getRotationMatrix2D((w/2,h/2), degrees, 1)
        else:
            rotation_mat = default_rotation_mat
            needs_rotate_90 = default_rotaion_90
        if needs_rotate_90:                
            frame=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        (h,w,c)=frame.shape
        return (cv2.warpAffine(frame,rotation_mat,(w,h),borderValue=(255,255,255)),rotation_mat,needs_rotate_90)
    

    def reset_detected_data(self):
        self.detected_strings = []
        self.fixed_keys = {}
        self.detected_data = {}
        self.marking_boxes = {}

    def draw_markers(self,frame,position_markers):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for k in position_markers.keys():
            (x,y,w,h) = position_markers[k]
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            frame=cv2.putText(frame,k,(x,y-6),font,.3,(255,0,0),1,cv2.LINE_AA)
        return frame


class OSR4Png(InteractiveOSR):
    def __init__(self,filename):
        super().__init__()
        image = Image.open(filename)
        self.scannedimages =  [ cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)]

    def normalize_angle(self,img):
        rotation_mat = cv2.getRotationMatrix2D((0,0),0, 1)
        needs_rotate_90 = False
        self.reset_detected_data()
        (img,rotation_mat,needs_rotate_90)=self.modify_angle(img,rotation_mat,needs_rotate_90)
        return img

    def detect_a_line(self,line_img,width):
        tseg = self.get_box_coordinates_h(line_img,width)
        detected_tokens = []
        box_coord = []
        for (left,right) in tseg:
            box_img=line_img[:,left:right]
            (left_l,right_l,top_l,bottom_l)=self.get_inside_box_coordinates(box_img)
            box_img = box_img[top_l:bottom_l,left_l:right_l]
            box_coord.append(((left+left_l,top_l,right_l-left_l,bottom_l-top_l),box_img))

            preprocessed = self.get_preprocessed_box_img(box_img)
            num=self.detect_char(preprocessed)
            detected_tokens.append(num)
            #cv2.imshow('box_img', box_img)
            #cv2.imshow('box_img_x', preprocessed)
            #keyinput= cv2.waitKey(0)
            #if keyinput & 0xFF == ord('q'):
            #    return
        return(detected_tokens,box_coord)

    def detect_a_page(self,img):
        img_c = self.normalize_angle(img)
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        img = self.reduct_noise(img)
        (position_markers,strings)=self.detect_position_markers(img)
        lines=self.get_line_coordinates(position_markers)
        box_coord = []
        detected_data = []
        for (ltop,lbottom,lleft,lright,width) in lines:
            line_img = img[ltop:lbottom,lleft:lright]
            (detected_tokens,bc)=self.detect_a_line(line_img,width)
            detected_data.append(detected_tokens)
            box_coord=box_coord+[ ((l+lleft,t+ltop,w,h),boximg) for ((l,t,w,h),boximg) in bc]
        return (detected_data,box_coord,position_markers,img_c)
    
    def detect_with_gui(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for pagenum,frame in enumerate(self.scannedimages):
            (detected_data,box_coord,position_markers,img)=self.detect_a_page(frame)
            print(detected_data)
            frame_modified = img
            #frame_modified = img_gray
            while True:
                frame = frame_modified.copy()
                s="Page: {:d}".format(pagenum+1)
                frame=cv2.putText(frame,s,(0,30),font,1.0,(255,255,255),4,cv2.LINE_AA)
                frame=cv2.putText(frame,s,(0,30),font,1.0,(64,128,64),2,cv2.LINE_AA)

                for qid in position_markers.keys():
                    hmarkers={}
                    vmarkers={}
                    frame = self.draw_markers(frame,position_markers[qid])
                    pass
                for ((x,y,w,h),img) in box_coord:
                    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

                cv2.imshow('toyomr scan image', frame)
                # quit
                keyinput=cv2.waitKey(1)
                if keyinput & 0xFF == ord('q'):
                    return
                elif keyinput & 0xFF == ord('z'):
                    is_sleeping = not is_sleeping
                elif keyinput & 0xFF == ord(' '):
                    break
                elif keyinput & 0xFF == 27:
                    #ESC
                    return
                elif keyinput & 0xFF == 13:
                    #enter
                    for li in self.get_detected_answers_for_questions_as_csv_lines():
                        print(li)
                    break

    
def main_png():
    if len(sys.argv) < 2:
        usage="{:s} devicenum".format(sys.argv[0])
        print(usage)
        return
    filename = sys.argv[1]
    osr = OSR4Png(filename)
    osr.detect_with_gui()

def main():
    if len(sys.argv) < 2:
        usage="{:s} devicenum".format(sys.argv[0])
        print(usage)
        return
    if sys.argv[1].endswith(".pdf"):
        main_pdf()
    if sys.argv[1].endswith(".png"):
        main_png()
    else:
        main_cap()



if __name__ == "__main__":
    main()


