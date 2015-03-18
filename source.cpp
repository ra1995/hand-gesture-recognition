#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\video\tracking.hpp>
#include<opencv2\ocl\ocl.hpp>

#include<iostream>
#include<vector>
#include<cmath>
#include<ctype.h>
#include<Windows.h>

using namespace std;
using namespace cv;
using namespace cv::ocl;

Mat fimage;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;

void MouseSetup(INPUT *buffer,int x,int y)
{
	buffer->type=INPUT_MOUSE;
	buffer->mi.dx=(x*(0xFFFF/1366));
	buffer->mi.dy=(y*(0xFFFF/768));
	buffer->mi.mouseData=0;
	buffer->mi.dwFlags=MOUSEEVENTF_ABSOLUTE;
	buffer->mi.time=0;
	buffer->mi.dwExtraInfo=0;
}

void MouseClick(INPUT *buffer)
{
	buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTDOWN);
	SendInput(1,buffer,sizeof(INPUT));
	Sleep(10);
	buffer->mi.dwFlags=(MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_LEFTUP);
	SendInput(1,buffer,sizeof(INPUT));
}

static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
        selection &= Rect(0, 0, fimage.cols, fimage.rows);
    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}

int main()
{
	int iLowH = 0;
	int iHighH = 179;

	int iLowS = 30; 
	int iHighS = 255;

	int iLowV = 10;
	int iHighV = 255;

	//Create trackbars in "Central" window
	//namedWindow("Central",CV_WINDOW_AUTOSIZE);
	//namedWindow( "Histogram", 0 );
	namedWindow( "Hand Model", CV_WINDOW_AUTOSIZE );

    setMouseCallback( "Hand Model", onMouse, 0 );

	/*
	cvCreateTrackbar("LowH", "Central", &iLowH, 179); //Hue (0 - 179)
	cvCreateTrackbar("HighH", "Central", &iHighH, 179);

	cvCreateTrackbar("LowS", "Central", &iLowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("HighS", "Central", &iHighS, 255);

	cvCreateTrackbar("LowV", "Central", &iLowV, 255); //Value (0 - 255)
	cvCreateTrackbar("HighV", "Central", &iHighV, 255);
	*/

	VideoCapture cap(0);
	Mat image,trajectory;
	namedWindow("video",CV_WINDOW_AUTOSIZE);

	char ch;
	int flag=0;
	int update_bg_model=-1;
	int largest_contour_index;
	int mouseflag=0;
	int mousecount=0;
	double contour_area;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Point2f tmp1(0,0),tmp2(0,0);
	Moments m;
	
	bool mouse = false;
	bool circ = true;
	bool gest = false;

	Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

	Mat fgimg,fgmask,img,handmask;
	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

	cap>>image;
	trajectory=Mat::zeros(image.rows,image.cols,image.type());
	cv::ocl::MOG2 mog2;
	oclMat dimage(image);
	oclMat dfgimg,dfgmask,dimg;
	dfgimg.create(image.size(),image.type());

	while(true)
	{
		cap>>image;
		
		dimage.upload(image);
		mog2(dimage, dfgmask,update_bg_model);
        mog2.getBackgroundImage(dimg);

		dfgmask.download(fgmask);
        dfgimg.download(fgimg);
        if (!dimg.empty())
            dimg.download(img);

		erode(fgmask, fgmask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
		//dilate( backproj, backproj, getStructuringElement(MORPH_ELLIPSE, Size(10, 10)) ); 
		blur(fgmask,fgmask,Size(10,10));
		threshold(fgmask,fgmask,10,255,CV_THRESH_BINARY);
		fgimg.setTo(Scalar::all(0));
		erode(fgmask, fgmask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
		image.copyTo(fgimg,fgmask);
		fimage=Mat(fgimg).clone();

		cvtColor(fimage, hsv, COLOR_BGR2HSV);
		if( trackObject )
        {
			int _vmin = iLowV, _vmax = iHighV;
			int _smin = iLowS, _smax = iHighS;

            inRange(hsv, Scalar(0, MIN(_smin,_smax), MIN(_vmin,_vmax)),Scalar(179, MAX(_smin,_smax), MAX(_vmin, _vmax)), mask);
            int ch[] = {0, 0};
            hue.create(hsv.size(), hsv.depth());
            mixChannels(&hsv, 1, &hue, 1, ch, 1);

			if( trackObject < 0 )
                {
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);

                    trackWindow = selection;
                    trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, CV_HSV2BGR);

                    for( int i = 0; i < hsize; i++ )
                    {
                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows),
                                   Point((i+1)*binW,histimg.rows - val),
                                   Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }

				calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
				threshold(backproj,backproj,1,255,THRESH_BINARY);
				erode(fgmask, fgmask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)) );
				dilate(fgmask, fgmask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
                
				RotatedRect trackBox;
				if(trackWindow.height>10 && trackWindow.width>10)
				{
				trackBox = CamShift(backproj, trackWindow,TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

				if( backprojMode )
                    cvtColor( backproj, fimage, COLOR_GRAY2BGR );
				rectangle( fimage, trackBox.boundingRect(), Scalar(0,0,255),3);

				Rect myrect=trackBox.boundingRect();
				int pad=myrect.width*2;
				copyMakeBorder(backproj,backproj,pad,pad,pad,pad,BORDER_CONSTANT,Scalar(0,0,0));
				Mat roimask=Mat(backproj,Range(myrect.y-(myrect.width/2)+pad,myrect.y+(myrect.width*1.5)+pad),Range(myrect.x-(myrect.width/2)+pad,myrect.x+myrect.width+(myrect.width/2)+pad));
				backproj=Scalar::all(0);
				roimask=Scalar::all(255);
				int b=backproj.cols;
				int h=backproj.rows;
				backproj=backproj.colRange(Range(pad,b-pad));
				backproj=backproj.rowRange(Range(pad,h-pad));
				
				Mat testing;
				fgmask.copyTo(testing,backproj);
				fgmask=testing.clone();
				
		findContours(fgmask,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
		vector<vector<int> >hull(1);
		vector<vector<Point> >hull_pt(1);
		vector<Vec4i> defects;

		if(!contours.empty())
		{
			largest_contour_index=0;
			contour_area=contours[0].size();
			for(int i=1;i<contours.size();i++)
			{
				if(contours[i].size()>contour_area)
				{
					largest_contour_index=i;
					contour_area=contours[i].size();
				}
			}
			if(contourArea(contours[largest_contour_index])>100)
			{
				m=moments(contours[largest_contour_index]);
				tmp1=tmp2;
				tmp2.x=640-m.m10/m.m00;
				tmp2.y=m.m01/m.m00;
				drawContours(image,contours,largest_contour_index,Scalar(255,0,0),2);
				circle(trajectory,tmp2,5,Scalar(0,0,255),2);
				circle(image,Point2f(640-tmp2.x,tmp2.y),5,Scalar(255,0,0),2);

				if(flag==0)
				{
					flag=1;
					continue;
				}
				if(tmp1.y<tmp2.y)
				{
					line(trajectory,tmp1,tmp2,Scalar(0,255,0),2);
				}
				else line(trajectory,tmp1,tmp2,Scalar(255,0,0),2);

				// Convex Hull
				convexHull( Mat(contours[largest_contour_index]), hull[0], false );
				convexHull( Mat(contours[largest_contour_index]), hull_pt[0], false );
				convexityDefects(contours[largest_contour_index], hull[0], defects);
				drawContours( image, hull_pt, 0, Scalar(0,0,255), 2, 8, vector<Vec4i>(), 0, Point() );

				if(defects.size()>5)
				{
				vector<float> distances,fdistances;
				vector<Vec4i> index,findex;
				vector<Point2f> bounding_ellipse;
				for(int i=0;i<defects.size();i++)
				{
					index.push_back(defects[i]);
					distances.push_back(defects[i][3]);
				}

				for(int i=0;i<6;i++)
				{
					int ptr=distance(distances.begin(),max_element(distances.begin(),distances.end()));
					fdistances.push_back(distances[ptr]);
					findex.push_back(index[ptr]);
					bounding_ellipse.push_back(contours[largest_contour_index][index[ptr][2]]);
					circle(image,contours[largest_contour_index][index[ptr][0]],5,Scalar(0,255,0),2);
					circle(image,contours[largest_contour_index][index[ptr][1]],5,Scalar(0,255,0),2);
					circle(image,contours[largest_contour_index][index[ptr][2]],5,Scalar(0,255,0),2);

					distances.erase(distances.begin()+ptr);
					index.erase(index.begin()+ptr);
				}

				if(circ)
				{
					ellipse(image,fitEllipse(bounding_ellipse),Scalar(0,255,0),2,8);
				}
				if(gest)
				{
					int i,cnt=0;
					for(i=0;i<6;i++)
					{
						//cout<<fdistances[i]<<"\n";
						if(fdistances[i]<3000)
						{
							cnt++;
						}
					}

					if(cnt>3)
					{
						putText(image,"Close",Point(100,100),FONT_HERSHEY_SCRIPT_SIMPLEX,2,Scalar(100,100,100),2);
						if(mouse)
						{
							POINT mypoint;
							GetCursorPos(&mypoint);
							SetCursorPos(mypoint.x+(tmp2.x-tmp1.x),mypoint.y+(tmp2.y-tmp1.y));
							//cout<<cnt<<"\n";
							if(cnt==4)
								mousecount++;
							else mousecount=0;
							if(cnt==4 && mouseflag==0 && mousecount==5)
							{
								INPUT buffer[1];
								GetCursorPos(&mypoint);
								MouseSetup(buffer,mypoint.x,mypoint.y);
								MouseClick(buffer);
								//cout<<"clicked";
								mouseflag=1;
								mousecount=0;
							}
						}
					}

					else
					{
						putText(image,"Open",Point(100,100),FONT_HERSHEY_SCRIPT_SIMPLEX,2,Scalar(100,100,100),2);
						mouseflag=0;
						mousecount=0;
					}
				}

				}
			}
		}
		
		contours.clear();
		hierarchy.clear();
				
				}
				else
				{
					trackObject = 0;
					histimg = Scalar::all(0);
					continue;
				}
		}

		if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(fimage, selection);
            bitwise_not(roi, roi);
        }

		cv::imshow("video",image);
		cv::imshow("trajectory",trajectory);

		cv::imshow("Hand Model", fimage );
        //cv::imshow("Histogram", histimg );

		//if(!img.empty())
          //cv::imshow("mean background image", img );

		ch=waitKey(1);

		switch (ch)
		{
		case 'c':
			trajectory.setTo(Scalar::all(0));
			circ=!circ;
			flag=0;
			break;
		case 's':
			update_bg_model=0;
			break;
		case 'r':
			update_bg_model=-1;
			break;
		case 'm':
			mouse=!mouse;
			break;
		case 'g':
			gest=!gest;
			break;
		case 'q':
			return 0;
			break;
		case 'b':
            backprojMode = !backprojMode;
            break;
		case 't':
			trackObject = 0;
            histimg = Scalar::all(0);
			break;
		case 'h':
            showHist = !showHist;
            if( !showHist )
                destroyWindow( "Histogram" );
            else
                namedWindow( "Histogram", 1 );
            break;
		default:
			break;
		}
	}

	return 0;
}
