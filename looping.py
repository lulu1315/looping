import getopt
import cv2
import numpy as np
import sys

input_root="./images/lionwalk"
extension="png"
loop_start=94
loop_end=120
loop_half_window=8
flowscale=.5 #downscale optical flow input
flowmethod=0
#flowmethod=0 : no opticalflow (blend/fade)
#flowmethod=1 : opencv OpticalFlowFarneback method
#flowmethod=2 : deepmatching/deepflow method     

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in'     and strArgument != '': input_root       = strArgument
	if strOption == '--ext'    and strArgument != '': extension        = strArgument
	if strOption == '--s'      and strArgument != '': loop_start       = int(strArgument)
	if strOption == '--e'      and strArgument != '': loop_end         = int(strArgument)
	if strOption == '--w'      and strArgument != '': loop_half_window = int(strArgument)
	if strOption == '--m'      and strArgument != '': flowmethod       = int(strArgument)
	if strOption == '--fs'     and strArgument != '': flowscale        = float(strArgument)

if (flowmethod == 2):
    from deepmatching import deepmatching as dm
    from deepflow2    import deepflow2 as df
            
#numbers
loop_cut            =loop_start+round((loop_end-loop_start)/2)
loop_out            =loop_end+(loop_cut-loop_start)
Boffset             =loop_start-loop_end
transition_start    =loop_end-loop_half_window
transition_end      =loop_end+loop_half_window

print ("used frames: ",loop_start-loop_half_window+1,"-",loop_end+loop_half_window-1,sep = '')
print ("final loop : ",loop_cut,"-",loop_out,sep = '')
print ("offset     : ",Boffset,sep = '')

#static flags
startatframe1=1

#no flow , just copy
for i in range (loop_cut,transition_start+1):
    Aframe=i
    print("\nprocessing loop frame : ",i-loop_cut+1,sep = '')
    Aimage="%s.%04d.%s"%(input_root,Aframe,extension)
    print("reading : ",Aimage,sep = '')
    imageA=cv2.imread(filename=Aimage,flags=cv2.IMREAD_COLOR)
    if (startatframe1 == 1):
        outputimage="%s_loop_m%d_%d_%d_%d.%04d.%s"%(input_root,flowmethod,loop_start,loop_end,loop_half_window,i-loop_cut+1,extension)
    else:
        outputimage="%s_loop_m%d_%d_%d_%d.%04d.%s"%(input_root,flowmethod,loop_start,loop_end,loop_half_window,i,extension)
    print("writing final frame : ",outputimage)
    cv2.imwrite(filename=outputimage,img=(imageA))
    
for i in range (transition_start + 1,transition_end):
    Aframe=i
    Bframe=Aframe+Boffset
    Aimage="%s.%04d.%s"%(input_root,Aframe,extension)
    Bimage="%s.%04d.%s"%(input_root,Bframe,extension)
    print("\nprocessing loop frame : ",i-loop_cut+1,sep = '')
    #read A/B image
    print("readingA : ",Aimage,sep = '')
    imageA=cv2.imread(filename=Aimage,flags=cv2.IMREAD_COLOR)
    print("readingB : ",Bimage,sep = '')
    imageB=cv2.imread(filename=Bimage,flags=cv2.IMREAD_COLOR)
    #compute fractional value
    frac=(i-transition_start)/(transition_end-transition_start)
    print("fractional : ",frac,sep = '')
    if (flowmethod != 0):
        if (flowmethod == 1): #opencv OpticalFlowFarneback method
            bwA=cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
            bwB=cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
            #resize for optical flow computation
            if (flowscale != 1):
                bwA=cv2.resize(bwA,(0,0),fx=flowscale,fy=flowscale,interpolation = cv2.INTER_AREA);
                bwB=cv2.resize(bwB,(0,0),fx=flowscale,fy=flowscale,interpolation = cv2.INTER_AREA);
                print("flow scale : ",flowscale," [",bwA.shape[1],",",bwA.shape[0],"]",sep = '')
            print("processing backward flow : ",Bframe,"->",Aframe,sep = '')
            previousflow=cv2.calcOpticalFlowFarneback(bwB,bwA, None, 0.5, 3, 30, 3, 5, 1.2, 0)
            print("processing forward flow  : ",Aframe,"->",Bframe,sep = '')
            nextflow=cv2.calcOpticalFlowFarneback(bwA,bwB, None, 0.5, 3, 30, 3, 5, 1.2, 0)
            if (flowscale != 1):
                #resize flow for warping
                previousflow=cv2.resize(previousflow,(0,0),fx=1./flowscale,fy=1./flowscale,interpolation=cv2.INTER_CUBIC);
                nextflow=cv2.resize(nextflow,(0,0),fx=1./flowscale,fy=1./flowscale,interpolation=cv2.INTER_CUBIC);
                previousflow=previousflow*1./flowscale;
                nextflow=nextflow*1./flowscale;
                print("resized flow [",previousflow.shape[1],",",previousflow.shape[0],"]",sep = '')
        if (flowmethod == 2): #deepmatching/deepflow method
            if (flowscale != 1):
                simageA=cv2.resize(imageA,(0,0),fx=flowscale,fy=flowscale,interpolation = cv2.INTER_AREA);
                simageB=cv2.resize(imageB,(0,0),fx=flowscale,fy=flowscale,interpolation = cv2.INTER_AREA);
                print("flow scale : ",flowscale," [",simageA.shape[1],",",simageA.shape[0],"]",sep = '')
                matches=dm.deepmatching(simageB,simageA,'-v')
                previousflow=df.deepflow2(simageB,simageA,matches,'-sintel')
                matches=dm.deepmatching(simageA,simageB,'-v')
                nextflow=df.deepflow2(simageA,simageB,matches,'-sintel')
                #resize flow for warping
                previousflow=cv2.resize(previousflow,(0,0),fx=1./flowscale,fy=1./flowscale,interpolation=cv2.INTER_CUBIC);
                nextflow=cv2.resize(nextflow,(0,0),fx=1./flowscale,fy=1./flowscale,interpolation=cv2.INTER_CUBIC);
                previousflow=previousflow*1./flowscale;
                nextflow=nextflow*1./flowscale;
                print("resized flow [",previousflow.shape[1],",",previousflow.shape[0],"]",sep = '')
            else:
                matches=dm.deepmatching(imageB,imageA,'-v')
                previousflow=df.deepflow2(imageB,imageA,matches,'-sintel')
                matches=dm.deepmatching(imageA,imageB,'-v')
                nextflow=df.deepflow2(imageA,imageB,matches,'-sintel')
        #warping nextflow A -> B
        mapx=np.zeros((nextflow.shape[0], nextflow.shape[1]), dtype=np.float32)
        mapy=np.zeros((nextflow.shape[0], nextflow.shape[1]), dtype=np.float32)
        print("warping Aframe")
        for y in range(nextflow.shape[0]):
            for x in range(nextflow.shape[1]):
                mapx[y,x] = x - (nextflow[y,x][0]*frac)
                mapy[y,x] = y - (nextflow[y,x][1]*frac) 
        warpA = cv2.remap(imageA,mapx,mapy,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        #warping previousflow B -> A   
        mapx=np.zeros((previousflow.shape[0], previousflow.shape[1]), dtype=np.float32)
        mapy=np.zeros((previousflow.shape[0], previousflow.shape[1]), dtype=np.float32)
        print("warping Bframe")
        for y in range(previousflow.shape[0]):
            for x in range(previousflow.shape[1]):
                mapx[y,x] = x - (previousflow[y,x][0]*(1.0-frac))
                mapy[y,x] = y - (previousflow[y,x][1]*(1.0-frac))
        warpB = cv2.remap(imageB,mapx,mapy,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        #final mix
        finalframe=cv2.addWeighted(warpA,1.0-frac,warpB,frac,0.0)
        print("mixing A/B",sep = '')
    else:
        print("no optical flow : blending A/B")
        finalframe=cv2.addWeighted(imageA,1.0-frac,imageB,frac,0.0)
    if (startatframe1 == 1):
        outputimage="%s_loop_m%d_%d_%d_%d.%04d.%s"%(input_root,flowmethod,loop_start,loop_end,loop_half_window,i-loop_cut+1,extension)
    else:
        outputimage="%s_loop_m%d_%d_%d_%d.%04d.%s"%(input_root,flowmethod,loop_start,loop_end,loop_half_window,i,extension)
    print("writing final frame : ",outputimage)
    cv2.imwrite(filename=outputimage,img=(finalframe))
    
#no flow , just copy
for i in range (transition_end,loop_out+1):
    Bframe=i+Boffset
    print("\nprocessing loop frame : ",i-loop_cut+1,sep = '')
    Bimage="%s.%04d.%s"%(input_root,Bframe,extension)
    print("reading : ",Bimage,sep = '')
    imageB=cv2.imread(filename=Bimage,flags=cv2.IMREAD_COLOR)
    if (startatframe1 == 1):
        outputimage="%s_loop_m%d_%d_%d_%d.%04d.%s"%(input_root,flowmethod,loop_start,loop_end,loop_half_window,i-loop_cut+1,extension)
    else:
        outputimage="%s_loop_m%d_%d_%d_%d.%04d.%s"%(input_root,flowmethod,loop_start,loop_end,loop_half_window,i,extension)
    print("writing final frame : ",outputimage)
    cv2.imwrite(filename=outputimage,img=(imageB))

