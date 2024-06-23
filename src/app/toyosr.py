from PIL import Image
import sys
import cv2
import math
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy
import pdf2image
import pypdf
import tempfile
import io
import zipfile
import os

class OneCharRecognizer:
    def __init__(self,model_file):
        self.model_file = model_file
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

class FilepathManager:
    def __init__(self,root):
        self.root = root
        self.dir_to_check = os.path.join(self.root,"to_check")
        self.pngdir_to_check = os.path.join(self.dir_to_check,"png")
        self.pdfdir_to_distribute = os.path.join(self.root,"pdf_to_distribute")
        self.pdfdir_not_to_distribute = os.path.join(self.root,"pdf_with_bad_id")
        self.pdfdir_with_sid = os.path.join(self.root,"pdf_without_id")


        self.dict_from_sid_to_dir = {"123456789":"__"}
        
    def get_line_png_to_check(self,page,line):
        d = os.path.join(self.pngdir_to_check,str(page))
        ans = os.path.join(d,str(line)+".png")
        return ans

    def is_leagal_sid(self,sid):
        if sid in self.dict_from_sid_to_dir:
            True
        return False

    def get_pdf_to_distribute(self,sid,page_num):
        filename = str(page_num)+".pdf"

        if sid == None:
            ans = self.pdfdir_with_sid
            ans = os.path.join(ans,filename)
            return ans
            
        if self.is_leagal_sid(sid):
            ans = self.pdfdir_to_distribute 
            ans = os.path.join(ans,self.dict_from_sid_to_dir[sid])
            ans = os.path.join(ans,filename)
            return ans
        ans = self.pdfdir_not_to_distribute 
        ans = os.path.join(ans,str(sid)+"_"+str(page_num))
        ans = os.path.join(ans,filename)
        return ans
            
        
class DetectionHint:
    def get_format_name(self,key):
        ans = {}
        if key == ('1','2'):
            return "f1"
        if key == ('3','4'):
            return "f2"
        return ans

    def get_persing_hint_from_format_name(self,name):
        ans = None
        if name == "f1":
            ans = [("s",8,"sid")]
        if name == "f2":
            ans = [("s",6,"qid"),("n",1,"data"),("n",3,"data")]
        return ans

    def get_detection_hint_from_format_name(self, name):
        ans = None
        if name == "f1":
            ans = ["n" for i in range(8)]
        if name == "f2":
            ans = ["n" for i in range(10)]
        return ans
        
    def get_key_in_the_same_line(self,key):
        key_in_the_same_line = [('1','2'),('3','4')]
        for p in key_in_the_same_line:
            if key in p:
                return p
        return (None,None)

    def get_sid_from_detected_data_in_a_page(self, page_data):
        for li in page_data:
            for (datatype,data) in li:
                if datatype == "sid":
                    return data
        return None
                
    def get_qid_from_detected_data_in_a_page(self, page_data):
        for li in page_data:
            for (datatype,data) in li:
                if datatype == "qid":
                    return data
        return None
                
    def get_serialized_data_from_detected_data_in_a_page(self, page_data):
        ans = []
        for li in page_data:
            for (datatype,data) in li:
                if datatype == "sid":
                    continue
                if datatype == "qid":
                    continue
                ans.append((datatype,data))
        return ans
                
class OSRbase:
    def __init__(self):
        self.num_ocr = OneCharRecognizer('dnn/mnist_100.onnx')
        self.detection_hint = DetectionHint()

    def detect_position_markers(self,frame):
        """
        returns dict D such that D[k1] is a list of rect's of qrcode with string k1. 
        """
        value = decode(frame, symbols=[ZBarSymbol.QRCODE])
        position_markers = {}
        if value:
            for qrcode in value:
                key = qrcode.data.decode('utf-8')
                if key not in position_markers:
                    position_markers[key]=[]
                position_markers[key].append(qrcode)
        return position_markers
    
    def detect_char(self,box_img,hint):
        return self.num_ocr.detect_char(box_img)

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
        return ans

    def get_line_coordinates(self,position_markers):
        done=[]
        ans=[]
        for k in position_markers.keys():
            if k in done:
                continue
            (k1,k2)=self.detection_hint.get_key_in_the_same_line(k)
            done.append(k1)
            done.append(k2)
            if k1 == None:
                continue
            if k1 in position_markers:
                w=position_markers[k1][0].rect.width
                h=position_markers[k1][0].rect.height
                left=w+position_markers[k1][0].rect.left
                top=position_markers[k1][0].rect.top
                bottom=h+top
                width = max(w,h)
            else:
                left=0
                top=None
                bottom=None
                width = None
            if k2 in position_markers:
                right=position_markers[k2][0].rect.left
                y1=position_markers[k2][0].rect.top
                if top == None:
                    top = y1
                if top > y1:
                    top = y1
                h=position_markers[k2][0].rect.height
                y1 = y1 + h
                if bottom == None:
                    bottom = y1
                if bottom < y1:
                    bottom = y1
                w=position_markers[k2][0].rect.width
                w = max(w,h)
                if width == None:
                    width = w
                if width < w:
                    width = w

            else:
                right=-1
            format_name = self.detection_hint.get_format_name((k1,k2))
            ans.append((top,bottom,left,right,width,format_name))
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


    def detect_angle(self,frame):
        """
        M
        returns degrees of angle of image if position marker is detected; otherwise None.  the range of degree is depend on the range of output of math.atan2.
        INPUT:
        frame - image
        """
        position_markers = self.detect_position_markers(frame)
        west_x = 0
        west_y = 0
        east_x = 0
        east_y = 0
        for k in position_markers.keys():
            (k1,k2) = self.detection_hint.get_key_in_the_same_line(k)
            if k1 not in position_markers:
                continue
            if k2 not in position_markers:
                continue
            if len(position_markers[k1])!=len(position_markers[k2]):
                continue
            
            for qrcode in position_markers[k1]:
                x_av=0
                y_av=0
                for (x,y) in qrcode.polygon:
                    x_av=x_av+x
                    y_av=y_av+y
                x_av = x_av / len(qrcode.polygon)
                y_av = y_av / len(qrcode.polygon)
            west_x = west_x+x_av
            west_y = west_y+y_av

            for qrcode in position_markers[k2]:
                x_av=0
                y_av=0
                for (x,y) in qrcode.polygon:
                    x_av=x_av+x
                    y_av=y_av+y
                x_av = x_av / len(qrcode.polygon)
                y_av = y_av / len(qrcode.polygon)

            east_x = east_x+x_av
            east_y = east_y+y_av
            
            if east_y-west_y == 0 and east_x-east_y == 0:
                return None
            ans= math.degrees(math.atan2(east_y-west_y,east_x-east_y))
            return ans
        else:
            return None

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

    def normalize_angle(self,img):
        rotation_mat = cv2.getRotationMatrix2D((0,0),0, 1)
        needs_rotate_90 = False
        (img,rotation_mat,needs_rotate_90)=self.modify_angle(img,rotation_mat,needs_rotate_90)
        return img

    def detect_tokens_in_a_line(self,line_img,width,format):
        if format == None:
            format = []
        tseg = self.get_box_coordinates_h(line_img,width)
        format = format+[ None for i in tseg]
        detected_tokens = []
        box_coord = []
        for ((left,right),hint) in zip(tseg,format):
            box_img=line_img[:,left:right]
            (left_l,right_l,top_l,bottom_l)=self.get_inside_box_coordinates(box_img)
            box_img = box_img[top_l:bottom_l,left_l:right_l]
            box_coord.append((left+left_l,top_l,right_l-left_l,bottom_l-top_l))

            preprocessed = self.get_preprocessed_box_img(box_img)
            num=self.detect_char(preprocessed,hint)
            detected_tokens.append(num)
        return(detected_tokens,box_coord)

    def perse_tokens_in_a_line(self,detected_tokens,format):
        detected_data = []
        if format != None:
            i = 0
            for (datatype,datalen,key) in format:
                if i+datalen > len(detected_tokens):
                    break
                if datatype == "s":
                    d=""
                    for di in detected_tokens[i:i+datalen]:
                        d=d+str(di)
                elif datatype == "n":
                    d=0
                    for di in detected_tokens[i:i+datalen]:
                        d=d*10+int(di)
                i=i+datalen
                detected_data.append((key,d))
        return detected_data

    def detect_data_in_a_line(self,line_img,width,format_name):
        format = self.detection_hint.get_detection_hint_from_format_name(format_name)
        (detected_tokens,box_coord)=self.detect_tokens_in_a_line(line_img,width,format)
        format = self.detection_hint.get_persing_hint_from_format_name(format_name)
        detected_data = self.perse_tokens_in_a_line(detected_tokens,format)
        return (detected_data,detected_tokens,box_coord)
    
    def detect_data_in_a_page(self,img):
        img_c = self.normalize_angle(img)
        img = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
        img = self.reduct_noise(img)
        position_markers=self.detect_position_markers(img)
        lines=self.get_line_coordinates(position_markers)
        box_coord = []
        detected_data = []
        for (ltop,lbottom,lleft,lright,width,ditection_hint) in lines:
            line_img = img[ltop:lbottom,lleft:lright]
            (data,detected_tokens,bc)=self.detect_data_in_a_line(line_img,width,ditection_hint)
            detected_data.append(data)
            box_coord=box_coord+[ (l+lleft,t+ltop,w,h) for (l,t,w,h) in bc]
        
        img_info = {}
        img_info["box_coordinates"]=box_coord
        img_info["position_markers"]=position_markers
        img_info["normalized_img"]=img_c
        img_info["line_info"]=lines
        return (detected_data,img_info)

class InteractiveOSR(OSRbase):
    def __init__(self):
        super().__init__()
    
    def detect_with_gui(self):
        pass



class OSR4Pdf(InteractiveOSR):
    def __init__(self,filename):
        super().__init__()

        self.original_pdffile = pypdf.PdfReader(filename)


            #pdfimages = pdf2image.convert_from_path(filename,dpi=600)

        #self.scannedimages =  [ cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR) for image in pdfimages]

    def get_scannedimage_from_pdfpage(self,pdfpage):
        pdfwriter = pypdf.PdfWriter()
        pdfwriter.add_page(pdfpage)
        t = io.BytesIO()
        pdfwriter.write(t)
        t.flush()
        pdfimages = pdf2image.convert_from_bytes(t.getvalue(),dpi=600)
        img = cv2.cvtColor(numpy.asarray(pdfimages[0]),cv2.COLOR_RGB2BGR)
        return img

    def save_partial_image_as_png(self,pngfile,img,top,bottom,left,right):
        mag = 5
        lineimg = img[top:bottom,left:right]
        h=(bottom-top)//mag
        w=(right-left)//mag
        lineimg = cv2.resize(lineimg,(w,h))
        (flag,buffer)=cv2.imencode(".png",lineimg)
        if flag:
            pngfile.write(buffer)

    def save_png_images_of_line(self,logzip,filepathmanager,pagenum,img_info):
        img=img_info["normalized_img"]
        lines_coord=img_info["line_info"]        
        for (linenum,lineinfo) in enumerate(lines_coord):
            (top,bottom,left,right,width,ditection_hint)=lineinfo
            fn=filepathmanager.get_line_png_to_check(pagenum,linenum)
            print(fn)
            with logzip.open(fn,"w") as pngfile:
                self.save_partial_image_as_png(pngfile,img,top,bottom,left,right)
    def save_pdf_to_distribute(self,logzip,pdfpage,filepathmanager,pagenum,sid):
        fn=filepathmanager.get_pdf_to_distribute(sid,pagenum)
        pdfwriter = pypdf.PdfWriter()
        pdfwriter.add_page(pdfpage)
        t = io.BytesIO()
        pdfwriter.write(t)
        t.flush()
        with logzip.open(fn,"w") as f:
            print(fn)
            f.write(t.getvalue())
    def get_table_data(self,detected_data):
        all_qid = []
        for di in detected_data:
            qid = self.detection_hint.get_qid_from_detected_data_in_a_page(di)
            if qid not in all_qid:
                all_qid.append(qid)
        all_qid.sort()
        all_sid = []
        for di in detected_data:
            sid = self.detection_hint.get_sid_from_detected_data_in_a_page(di)
            if sid == None:
                continue
            if sid not in all_sid:
                all_sid.append(sid)
        all_sid.sort()

        data_with_sid={}
        for sid in all_sid:
            data_with_sid[sid]={}
            for qid in all_qid:
                data_with_sid[sid][qid]=[]
        for di in detected_data:
            sid = self.detection_hint.get_sid_from_detected_data_in_a_page(di)
            qid = self.detection_hint.get_qid_from_detected_data_in_a_page(di)
            sdata = self.detection_hint.get_serialized_data_from_detected_data_in_a_page(di)
            if sid == None:
                continue
            data_with_sid[sid][qid].append(sdata)
        max_num_of_qid={}
        for qid in all_qid:
            max_num_of_qid[qid]= max(len(data_with_sid[sid][qid]) for sid in all_sid)

        max_len_of_qid={}
        for qid in all_qid:
            max_len_of_qid[qid]= max(len(di) for sid in all_sid for di in data_with_sid[sid][qid])

        ans = []
        for sid in all_sid:
            line = [sid]
            data = data_with_sid[sid]
            for qid in all_qid:
                for di in data[qid]:
                    for dij in di:
                        line.append(dij)
                    for i in range(max_len_of_qid[qid]-len(di)):
                        line.append(None)
                for j in range(max_num_of_qid[qid]-len(data[qid])):
                    for i in range(max_len_of_qid[qid]):
                        line.append(None)
            ans.append(line)
        top_line = ["id"] 
        for qid in all_qid:
            for j in range(max_num_of_qid[qid]):
                top_line.append(qid)
                for i in range(max_len_of_qid[qid]-1):
                    top_line.append(None)
       
        print(ans,top_line)
        return (ans,top_line)
        
    def detect_with_gui(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        filename = "/tmp/x.zip"
        filepathmanager = FilepathManager("x")
        rawdata = []
        with zipfile.ZipFile(filename,'w') as myzip:            
            for pagenum,pdfpage in enumerate(self.original_pdffile.pages):
                img = self.get_scannedimage_from_pdfpage(pdfpage)
                (detected_data,img_info)=self.detect_data_in_a_page(img)
                rawdata.append(detected_data)
                self.save_png_images_of_line(myzip,filepathmanager,pagenum,img_info)
                sid=self.detection_hint.get_sid_from_detected_data_in_a_page(detected_data)
                self.save_pdf_to_distribute(myzip,pdfpage,filepathmanager,pagenum,sid)
                print(detected_data)
            (tablebody,table_header)=self.get_table_data(rawdata)
            frame_modified = img_info["normalized_img"]

                
            while True:
                position_markers=img_info["position_markers"]
                lines_coord=img_info["line_info"]        
                box_coord=img_info["box_coordinates"]
                frame = frame_modified.copy()
                s="Page: {:d}".format(pagenum+1)
                frame=cv2.putText(frame,s,(0,30),font,1.0,(255,255,255),4,cv2.LINE_AA)
                frame=cv2.putText(frame,s,(0,30),font,1.0,(64,128,64),2,cv2.LINE_AA)

                for k in position_markers.keys():
                    for qrcode in position_markers[k]:
                        (x,y,w,h)=qrcode.rect
                        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                for (top,bottom,left,right,width,ditection_hint) in lines_coord:
                    frame=cv2.rectangle(frame,(left,top),(right,bottom),(155,155,0),1)
                for (x,y,w,h) in box_coord:
                    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

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
                    break

    
def main_pdf():
    if len(sys.argv) < 2:
        usage="{:s} devicenum".format(sys.argv[0])
        print(usage)
        return
    filename = sys.argv[1]
    osr = OSR4Pdf(filename)
    osr.detect_with_gui()

def main():
    if len(sys.argv) < 2:
        usage="{:s} devicenum".format(sys.argv[0])
        print(usage)
        return
    if sys.argv[1].endswith(".pdf"):
        main_pdf()
    else:
        main_cap()



if __name__ == "__main__":
    main()


