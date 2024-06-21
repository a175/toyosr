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
    def __init__(self,questions):
        self.target_keys = {}
        for questionid in questions:
            self.target_keys[questionid]=[k for qi in questions[questionid] for (k,a) in qi]
    
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

    def draw_detected_data(self,frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for questionid in self.marking_boxes.keys():
            for k in self.marking_boxes[questionid].keys():
                (x1,x2,y1,y2)=self.marking_boxes[questionid][k]
                if k in self.fixed_keys[questionid]:
                    frame=cv2.line(frame,(x1,y1),(x2,y2),(256,128,128),2)
                if k in self.detected_data[questionid]:
                    frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)
                    frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
                    frame=cv2.putText(frame,"{:s}-{:s}".format(k[0],k[1]),(x1,y1-6),font,.3,(255,0,255),1,cv2.LINE_AA)
                else:
                    frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)
                    frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(0,80,160),1)
        s="/".join([ k for k in self.detected_strings if not k.startswith("marker:")])
        frame=cv2.putText(frame,s,(0,70),font,1.0,(255,255,255),4,cv2.LINE_AA)
        frame=cv2.putText(frame,s,(0,70),font,1.0,(64,64,128),2,cv2.LINE_AA)
        return frame

    def draw_markers(self,frame,position_markers,hmarkers,vmarkers):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for k in position_markers.keys():
            (x,y,w,h) = position_markers[k]
            frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            frame=cv2.putText(frame,k,(x,y-6),font,.3,(255,0,0),1,cv2.LINE_AA)
                
        
        for k in hmarkers.keys():
            for (d,a,b) in hmarkers[k]:
                frame=cv2.line(frame,(d,a),(d,b),(128,128,0),4)
                frame=cv2.putText(frame,k,(d+10,a+6),font,.3,(255,0,255),1,cv2.LINE_AA)
        for k in vmarkers.keys():
            for (d,a,b) in vmarkers[k]:
                frame=cv2.line(frame,(a,d),(b,d),(128,128,0),4)
                frame=cv2.putText(frame,k,(a,d-6),font,.3,(255,0,255),1,cv2.LINE_AA)
        return frame


class OSR4Png(InteractiveOSR):
    def __init__(self,filename,questions):
        super().__init__(questions)
        self.questions = questions
        image = Image.open(filename)
        self.scannedimages =  [ cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)]
        self.ocr = OneCharRecognizer()

    def detect_with_gui(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for pagenum,frame in enumerate(self.scannedimages):
            rotation_mat = cv2.getRotationMatrix2D((0,0),0, 1)
            needs_rotate_90 = False
            self.reset_detected_data()
            (frame,rotation_mat,needs_rotate_90)=self.modify_angle(frame,rotation_mat,needs_rotate_90)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_gray = self.reduct_noise(img_gray)
            (position_markers,strings)=self.detect_position_markers(img_gray)
            #self.update_detected_strings(strings)
            width=0
            n=0
            max_top=0
            max_height=0
            for k in position_markers.keys():
                n=n+1
                width=width+position_markers[k]["x"].width
                if max_height < position_markers[k]["x"].height:
                    max_height = position_markers[k]["x"].height
                if max_top < position_markers[k]["x"].top:
                    max_top = position_markers[k]["x"].top
            frame = frame[0:max_top+13*max_height//10,:]
            img_gray = img_gray[0:max_top+2*max_height,:]
            width=width/n
            
            lines=self.get_line_coordinates(position_markers)
            box_coord = []
            detected_data = []
            for (ltop,lbottom,lleft,lright,width) in lines:
                detected_tokens = []
                print(lleft,lright,ltop,lbottom)
                line_img = img_gray[ltop:lbottom,lleft:lright]
                tseg = self.get_box_coordinates_h(line_img,width)
                for (left,right) in tseg:
                    box_img=line_img[:,left:right]
                    (left_l,right_l,top_l,bottom_l)=self.get_inside_box_coordinates(box_img)
                    box_img = box_img[top_l:bottom_l,left_l:right_l]
                    box_coord.append(((lleft+left+left_l,ltop+top_l,right_l-left_l,bottom_l-top_l),box_img))

                    preprocessed = self.get_preprocessed_box_img(box_img)
                    num=self.detect_char(preprocessed)
                    detected_tokens.append(num)
                    #cv2.imshow('box_img', box_img)
                    #cv2.imshow('box_img_x', preprocessed)
                    #keyinput= cv2.waitKey(0)
                    #if keyinput & 0xFF == ord('q'):
                    #    return
                detected_data.append(detected_tokens)

            print(detected_data)
            frame_modified = frame
            #frame_modified = img_gray
            while True:
                frame = frame_modified.copy()
                s="Page: {:d}".format(pagenum+1)
                frame=cv2.putText(frame,s,(0,30),font,1.0,(255,255,255),4,cv2.LINE_AA)
                frame=cv2.putText(frame,s,(0,30),font,1.0,(64,128,64),2,cv2.LINE_AA)

                for qid in position_markers.keys():
                    hmarkers={}
                    vmarkers={}
                    frame = self.draw_markers(frame,position_markers[qid],hmarkers,vmarkers)
                    pass
                for ((x,y,w,h),img) in box_coord:
                    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

                frame=self.draw_detected_data(frame)
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
    question = [ [(chr(ord("A")+i),"{:d}{:d}".format(j,k))  for k in range(1,5)] for j in range(1,6) for i in range(10)]
    question_a = [[ (qij,"{:d}".format(j+1)) for (j,qij) in enumerate(qi) ] for qi in question]
    a=[["0","1","2","3","4","5","6"],["7","8","9","A"],["B","C","D"],["E","F"]]
    b=[["G","H","I"],["J","K","L","M"],["N","O"],["P","Q","R"]]
    question =[[("Y",aij) for aij in ai] for ai in a]+[ [("Z",bij) for bij in bi]for bi in b]
    question_b = [[ (qij,"{:d}".format(j+1)) for (j,qij) in enumerate(qi) ] for qi in question]
    questions = {"B":question_a,"A":question_b}

    osr = OSR4Png(filename,questions)
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


