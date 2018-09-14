#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>      // std::setprecision
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;

using namespace std;
using namespace cv;

#define PI 3.14159265

int dilation_elem = 0; //variables used for the dilation effect
int dilation_type = 0;
double est = 0;
double a = 0;
int n = 0;
string red = "/Users/Bouda/Desktop/TUEindhoven/Cyclone/red2.txt"; //the text files containing the thresholding data
//string blue = "/Users/Bouda/Desktop/TUEindhoven/Cyclone/blue2.txt";
string green = "/Users/Bouda/Desktop/TUEindhoven/Cyclone/green2.txt";
vector<RotatedRect> finalEllipse(1); //detected ellipse is stored here
VectorXd coef_vec2(6);


int hoop_r=257;                 //radius of the hoop in pixels at 1 meter
int h_bar = 100;            //altitude position of the drone in cm (using barometer)
int delta = 180;            //roll action of the drone
double h_hoop = 1.5;        //altitude position of the hoop in m (known together with GPS setpoints)
double radius_m=0.35;        //radius of the hoop
double drone_center = 15;    //distance between the camera and the center of the drone in cm
double x,y,z;                //Cartesian coordinates
double d_beta, d_alpha, d_omega, h_bar_m, dist_drone, angle_hor, angle_ver;
double omega_est, k_est, ff1, ff2, aa, bb,dist_meas, pix_meter, omega_passed;
double d_before = 2; //distance from hoop in meters
double v_in = 1; // speed at which the drone has to go through the hoop
double d_after = 1;                    // how far after the hoop the drone has to travel
double v_after=1;

double x_avg,y_avg,z_avg,dist_avg,angleV_avg,angleH_avg;

// Methods
MatrixXd equations_f(MatrixXd M_used, VectorXd cond_vec, int j);
MatrixXd statef(double coef, MatrixXd M_full, double t);
MatrixXd mainm(double iteration, double waypoints, double i, double coef, MatrixXd cond_final, double t);
MatrixXd Dimention3(MatrixXd init, MatrixXd p_before_hoop, MatrixXd final, VectorXd hoop_pos, double yaw0, double hoop_orient);

void record_values();

void contours_trackbar(int, void *);
void estimation(double omega_passed, double ell_h, double ell_w);
void display_values();
void full_pos();
void position_calculation();
void angles();
MatrixXd executeVision();
void executePathPlanner(double orientation);
vector<vector<Point> > contouring(Mat& input);

Mat threshold(Mat& input, string file);
Mat downsample(Mat& input, double n);
Mat equalize(Mat& input);
Mat dilation(Mat& input, int dilation_elem, int dilation_type);
Mat contouring(vector<vector<Point> >& input);
Mat combinedEllipse3(vector<vector<Point> >& input, Mat& t);
Mat removeBG(Mat& input, Mat& frame);

MatrixXd equations_f(MatrixXd M_used, VectorXd cond_vec, int j) {
    MatrixXd M_u(6, 6);
    M_u = M_used;
    VectorXd cond_(v6);
    cond_v = cond_vec;

    //std::cout << "Here is the matrix A:\n" << M_u << std::endl;
    //std::cout << "Here is the vector b:\n" << cond_v << std::endl;
    VectorXd x = M_u.fullPivLu().solve(cond_v);
    //std::cout << "The solution is:\n" << x << std::endl;

    coef_vec2 = x;
    MatrixXd M_full(4, 6);
    M_full << x(0),x(1),x(2),x(3),x(4),x(5),  0,5*x(0),4*x(1),3*x(2),2*x(3),x(4),  0,0,20*x(0),12*x(1),6*x(2),2*x(3),  0,0,0,60*x(0),24*x(1),6*x(2);
    return M_full;
}



MatrixXd statef(double coef, MatrixXd M_full, double t) {

    MatrixXd statef(4, 50);

    MatrixXd M_fullM(4, 6);
    M_fullM = M_full;
    //std::cout << M_fullM << std::endl;

    coef = coef / t;

    for (double i = 1.0 / coef; (int)round(i*coef) <= 50; i += (1.0 / coef)) {
        VectorXd m(6);
        m << pow(i,5),pow(i,4),pow(i,3),pow(i,2),pow(i,1),1;
        //std::cout << m << std::endl;


        int n = (int) round(i*coef);
        //std::cout << n << std::endl;
        MatrixXd result(4, 1);
        result = M_fullM * m;
        //std::cout << result << std::endl;

        //populate the nth column with the result of the matrix mult of M_full and t_instant
        for (int j = 0; j < 4; ++j) {
            statef(j,n-1) = result(j);
        }
    }
    return statef;
}


// This method is for path planning
MatrixXd Dimention3(MatrixXd init, MatrixXd p_before_hoop, MatrixXd final, MatrixXd hoop_pos, double yaw0,    double hoop_orient) {

    //the point before the hoop where we still see the hoop
    MatrixXd p33(3,3);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            p33(i,j) = p_before_hoop(i,j);
        }
    }
    p33(2,0) = 0;
    p33(2,1) = 0;
    p33(2,2) = 0;

    MatrixXd final_vec(6, 6);
    final_vec.block<3, 3>(0, 0) = init;
    final_vec.block<3, 3>(0, 3) = p33;
    final_vec.block<3, 3>(3, 0) = p33;
    final_vec.block<3, 3>(3, 3) = final;

    double coef = 50;
    int iteration = 0;
    int t = 0;
    MatrixXd cond_final(6,3);
    int counter = 0;
    int t_max = 0;
    int x = 5;
    int T_max = 35;
    int waypoints = 0;
    double yaw = 0;

    MatrixXd state(12, (int)coef);
    state.block<3, 1>(0, 49) = init.block<3, 1>(0, 0);
    state.block<3, 1>(4, 49) = init.block<3, 1>(0, 1);
    state.block<3, 1>(8, 49) = init.block<3, 1>(0, 2);

    MatrixXd trajectory(12, (int)((waypoints + 2)*coef));

    for (int i = 1; i <= waypoints + 2; i++) {

        if (iteration >= waypoints) {

            cond_final.block<3, 1>(0, 0) = state.block<3, 1>(0, 49);
            cond_final.block<3, 1>(0, 1) = state.block<3, 1>(4, 49);
            cond_final.block<3, 1>(0, 2) = state.block<3, 1>(8, 49);
            cond_final.block<3, 3>(3, 0) = final_vec.block<3, 3>(3, (3 * (i - 1)));

        }
        t = 1;

        state = mainm(iteration, waypoints, i, coef, cond_final,t);
        //        yaw = yaw_math(p_before_hoop(1,1),p_before_hoop(1,2),final(1,1),final(1,2),yaw0,t); need the method


        trajectory.block<12, 50>(0, (int)(iteration * 50)) = state;

        iteration++;
    }



    return trajectory;
}

MatrixXd mainm(double iteration, double waypoints, double i, double coef, MatrixXd cond_final, double t) {

    MatrixXd state(12,(int) coef);

    MatrixXd cond_vec = cond_final;
    int last = 1;
    int l1 = 3;
    int l2 = 3;
    MatrixXd m_used(6,6);
    m_used << 0,0,0,0,0,1, 0,0,0,0,1,0, 0,0,0,2,0,0, 1,1,1,1,1,1, 5,4,3,2,1,0, 20,12,6,2,0,0;

    MatrixXd coef_vec_val(6,3);

    for (int j = 1; j <= 3; j++) {
        MatrixXd M_full(4,6);
        M_full = equations_f(m_used, cond_vec.col(j-1), j);

        state.block<4,50>((j-1)*4,0) = statef(coef,M_full,t);
        coef_vec_val.block<6,1>(0,j-1) = coef_vec2;
    }

    return state;
}

//Dilation is an effect which takes the input image, and grows the highlights of the image, in this case the white pixels, which represent the position of the detected LEDs.
Mat dilation(Mat& input, int dilation_elem, int dilation_size) {
    Mat dilation_dst;
    if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
    else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
    else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; } //The shape of         the kernel applied can be vaied, in our case the ellipse is optimal.
    Mat element = getStructuringElement( dilation_type,
                                         Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                         Point( dilation_size, dilation_size ) );
    //the size of the dilation determined the kernel applied to each pixel, which in         our case determines how much the white pixels are enlarged. This has to be kept     fairly low, as neighbouring LED detections cannot touch, as then they will not be detected.
    dilate( input, dilation_dst, element );
    return dilation_dst;
}


Mat removeBG(Mat& input, Mat& frame) {

    int low_h=30, low_s=30, low_v=30; //threshold value declaration
    int high_h=200, high_s=200, high_v=200;

    Mat junk;
    string line;
    ifstream myfile;
    myfile.open("/Users/Bouda/Desktop/TUEindhoven/Cyclone/junk.txt"); //threshold values loaded from a text file
    if (myfile.is_open()) {
        getline (myfile,line);
        low_v = atoi(line.c_str());
        getline (myfile,line);
        low_s = atoi(line.c_str());
        getline (myfile,line);
        low_h = atoi(line.c_str());
        getline (myfile,line);
        high_v = atoi(line.c_str());
        getline (myfile,line);
        high_s = atoi(line.c_str());
        getline (myfile,line);
        high_h = atoi(line.c_str());

        myfile.close();
    }

    else cout << "Unable to open file";

    inRange(frame,Scalar(low_v, low_s, low_h), Scalar(high_v, high_s, high_h),junk);

    junk = dilation(junk, 2, 2);

    for(int y=0; y<=input.size().height; y++)
    {
        for(int x = 0; x <= input.size().width; x++)
        {
            if (input.at<uchar>(y,x) == 255 && junk.at<uchar>(y,x) == 255) {
                input.at<uchar>(y,x) = 0;
            }
        }
    }
    return input;
}


//Enables downsampling of the input video stream with a factor n
Mat downsample(Mat& input, double n) {
    Mat down;
    n = 1/n;
    resize(input, down, down.size(), n, n, INTER_AREA);
    return down;
}


//Histogram equalization of the input image
Mat equalize(Mat& input) {
    Mat eq, hsv;
    vector<Mat> channels;
    cvtColor(input, eq, CV_BGR2YCrCb); //conversion from BGR to YCrCb colorspace
    split(eq, channels);
    equalizeHist(channels[0], channels[0]); //onlz the first channel is equalized
    merge(channels, eq);
    cvtColor(eq, eq, CV_YCrCb2BGR);

    cvtColor(eq, hsv, COLOR_BGR2HSV); //output image is in HSV colorspace
    return hsv;
}



//Thresholding of the image; The output is a black&white image, where white represents the pixels passing the threshold
Mat threshold(Mat& input, string file) {

    Mat output;
    int low_h=30, low_s=30, low_v=30; //threshold value declaration
    int high_h=200, high_s=255, high_v=255;

    string line;
    ifstream myfile;
    myfile.open(file.c_str()); //threshold values loaded from a text file
    if (myfile.is_open()) {
        getline (myfile,line);
        low_h = atoi(line.c_str());
        getline (myfile,line);
        low_s = atoi(line.c_str());
        getline (myfile,line);
        low_v = atoi(line.c_str());
        getline (myfile,line);
        high_h = atoi(line.c_str());
        getline (myfile,line);
        high_s = atoi(line.c_str());
        getline (myfile,line);
        high_v = atoi(line.c_str());

        myfile.close();
    }

    else cout << "Unable to open file";
    cv::cvtColor( input, output, cv::COLOR_BGR2HSV );
    //thresholding command
    inRange(input,Scalar(low_h, low_s, low_v), Scalar(high_h, high_s, high_v),output);
    return output;
}


//This method applies contouring to the input image, which is a sort of edge detection algorithm, and the output is a collection of vectors.
vector<vector<Point> > contouring(Mat& input) {
    vector<Vec4i> hierarchy;
    vector<vector<Point> > contours;
    findContours( input, contours, hierarchy, CV_RETR_TREE,                   CV_CHAIN_APPROX_SIMPLE,Point(0, 0) );
    return contours;
}


//This method processes the vectors received from the contouring algorithm and finds the hoop bz describing its position and shape by a 2D ellipse.
Mat combinedEllipse3(vector<vector<Point> >& input, Mat& t) {
    Point coordinate;
    vector<Point>  bigEllipse;
    vector<Point>  bigEllipsefilter;
    vector<RotatedRect> bigEllipse2;
    double area [input.size()];
    n = 0;
    Mat drawing1 = Mat::zeros( t.size(), CV_8UC3 );
    vector<RotatedRect> minEllipse( input.size() );
    //contours only of a predetermined size are considered. Thie upper threshold could be lowered considerably, as now our aim is to detect the individual LEDs.
    for( size_t i = 0; i < input.size(); i++ )
    {
        if( input[i].size() < 200 && input[i].size() > 4) {
            //ellipse is fitted over the contours passing the size threshold
            minEllipse[i] = fitEllipse( Mat(input[i]) );
            //the area of the fitted ellipse and the area of the contour are                     calculated
            a = PI * minEllipse[i].size.width/2 * minEllipse[i].size.height/2;
            est = abs( contourArea(input[i] ) - a );
            //Only those ellipses are saved, which had a contour area reasonably                     close to the estimated ellipse area. This makes sure, that the                     detected shape was indeed an ellipse. (This can be done as the dilation kernel shape was an ellipse.)
            if ( est < 5) {
                //center position of the LEDs are saved to an array
                coordinate.x = minEllipse[i].center.x;
                coordinate.y = minEllipse[i].center.y;
                bigEllipse.push_back(coordinate);
            }
        }
    }

    //if at least there are 5 detected LEDs, an ellipse is fitted over their shape, which is the final detected hoop.
    if (bigEllipse.size() > 4) {
        minEllipse[1] = fitEllipse( Mat(bigEllipse) );
        ellipse( drawing1, minEllipse[1], Scalar(0, 0, 255), 2, 8 );
        finalEllipse[1] = minEllipse[1];
    }
    return drawing1;
}



//function for deriving the angles and distance knowing only ellipse's size and orientation
void estimation (double omega_passed, double ell_h, double ell_w) {
    delta = delta-180;             //specific for trackbar input (it cannot go for negative numbers)
    omega_est=(omega_passed-delta)*PI/180;    //rotation of the ellipse in radians

    //ratio k is always minor devided by major axis
    if (ell_h<ell_w){    k_est = ell_h/ell_w; }
    else if (ell_h>=ell_w) {    k_est = ell_w/ell_h;}

    ff1 = pow(k_est*cos(omega_est),2.0)+pow(sin(omega_est),2.0);
    ff2 = pow(k_est*sin(omega_est),2.0)+pow(cos(omega_est),2.0);

    bb = acos(sqrt(ff2)); //reverse beta angle
    double temp = pow(sin(bb),2.0);
    aa = acos(sqrt((ff1 - temp)/(1-temp))); //reverse alpha angle
    // aa = acos(sqrt((((2*ff1 - temp)/(1-temp)) + 1)/2));
    //aa = acos(sqrt((2*ff1 - temp)/(1 - temp)) + 1)); //reverse alpha angle

    if(ell_h<=ell_w){         //account for the constaint on the k raito <1
        dist_meas = (pix_meter*radius_m)/ell_w;
        angle_hor=aa;
        angle_ver=bb;
    }
    else if (ell_h>ell_w){
        dist_meas = (pix_meter*radius_m)/ell_h;
        angle_hor=bb;
        angle_ver=aa;
    }

}

void full_pos(){                    //determine from which position you perceive the hoop (above or below the hoop)
    h_bar_m = h_bar/100;
    if (omega_passed<0 && h_bar_m<h_hoop) {
        y=-y;
        //        z=-z;
    } else if (omega_passed>0 && h_bar_m>h_hoop) {
        y = -y;
    }
    //    } else if (omega_passed>0 && h_bar_m<h_hoop) {
    //        z=-z;
    //    }
}

void position_calculation() {        //convert to cartesian coordinate system
    dist_drone = (dist_meas * 100) + drone_center;
    //    cout << "angle_hor: " << angle_hor << endl;
    //    cout << "angle_ver: " << angle_ver << endl;

    //    z = sin(angle_hor)*dist_drone/100;
    //    x = (cos(angle_ver)*cos(angle_hor)*dist_drone)/100;
    //    y = (sin(angle_ver)*cos(angle_hor)*dist_drone)/100;

//    new
//    x = cos(angle_hor)*sin(angle_ver)*dist_drone/100;
//    y = sin(angle_hor)*dist_drone/100;
//    z = cos(angle_hor)*cos(angle_ver)*dist_drone/100;

    z = cos(angle_ver)*dist_drone/100;
    x = (cos(angle_hor)*sin(angle_ver)*dist_drone)/100;
    y = (sin(angle_ver)*sin(angle_hor)*dist_drone)/100;

    full_pos();
}

void display_values(){
    //cout << "Position of the center of the drone: "<< endl;
    cout << setprecision(2) << fixed  << "x = " << x <<"    y = " << y << "    z = " << z << endl;

    // cout << "Angles and distance: "<< endl;
    // cout <<"dist = " << dist_meas <<"    ang_ver = " << angle_ver << "    ang_hor = " << angle_hor << endl;


}

void record_values(){
    x_avg += x;
    y_avg += y;
    z_avg += z;

    dist_avg += dist_meas;
    angleV_avg += angle_ver;
    angleH_avg += angle_hor;

}

//

void angles(){
    ofstream file;
    file.open("/Users/Bouda/Desktop/TUEindhoven/Cyclone/angles2.txt", ios::out);
    file <<"measured "<< dist_meas << " " << angle_ver << " " << angle_hor << "\n";
    file <<"ellipse par " << finalEllipse[1].size.height<< " " << finalEllipse[1].size.width << "\n";
    file <<"\n";
    file.close();
}

MatrixXd executeVision() {
    pix_meter = hoop_r / radius_m;  //representation of meter in pixels

    int count = 1;
    int numOfEstimates = 0;
    MatrixXd estimatedHoopPos(1,5); //x,y,z,hoop_ang_hor,dist
    x_avg,y_avg,z_avg,dist_avg,angleV_avg,angleH_avg = 0;
    Mat thresh, thresh_red, thresh_green, eq, hsv, down, drawing, withBG, withoutBG;
    vector<vector<Point> > contours;

    //    const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! \
    //            nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! \
    //            videoconvert ! video/x-raw, format=(string)BGR ! \
    //            appsink";

    VideoCapture cap("/Users/Bouda/Desktop/video3.avi");
    //    VideoCapture cap(1);
    //
    //    cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
    //    cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);



    //    cap = VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! \
    nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! \
    videoconvert ! video/x-raw, format=(string)BGR ! \
    appsink");


    if (!cap.isOpened()) {
        cout << "Camera not working" << endl;
    }

    //    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
    //    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(CV_CAP_PROP_SATURATION, 0.75);
    cap.set(CV_CAP_PROP_CONTRAST, 0.75);
    cap.set(CV_CAP_PROP_BRIGHTNESS, 0.3);
    cap.set(CV_CAP_PROP_FOURCC ,CV_FOURCC('M', 'J', 'P', 'G') );
    namedWindow("Final Threshold", CV_WINDOW_AUTOSIZE);
    namedWindow("Detection", CV_WINDOW_AUTOSIZE);
    //namedWindow("Trackbar", CV_WINDOW_AUTOSIZE);
    namedWindow("Frame", 1);
    namedWindow("Down",1);
    namedWindow("With BG", CV_WINDOW_AUTOSIZE);
    namedWindow("Without BG", CV_WINDOW_AUTOSIZE);
    //createTrackbar( " Drone rotation ", "Trackbar", &delta, 360, contours_trackbar); //simulaion
    //createTrackbar( " Drone altitude ", "Trackbar", &h_bar, 500, contours_trackbar);

    std::cout << "got here" << endl;

    while (count < 500) {
        Mat frame;
        Mat kernel;
        int kernel_size = 3 + 2*( 0%5 );
        kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);

        cap >> frame; //passing the captured frame onto a Mat object
        filter2D(frame, frame,-1,kernel, Point(-1,-1), 0, BORDER_DEFAULT);
        imshow("Frame", frame);
        //        double fps = cap.get(CV_CAP_PROP_FPS);
        //        cout << "fps: " << fps << endl;
        count++;


        //now the methods desribed above are called individually
        frame = downsample(frame, 2);
        hsv = equalize(frame);

        //thresholding is done parallel with both red and green LEDs
        thresh_red = threshold(hsv, red);
        thresh_green = threshold(hsv, green);

        //the 2 kinds of thresholded outputs are combined
        bitwise_and(thresh_red, thresh_red, thresh_red, thresh_red);
        bitwise_and(thresh_green, thresh_green, thresh_green, thresh_green);
        add(thresh_red, thresh_green, thresh);

        withBG = thresh;
        thresh = removeBG(thresh, hsv);
        withoutBG = thresh;
        thresh = dilation(thresh, 2, 4);

        //        thresh = downsample(thresh,1);
        //        withBG = downsample(withBG, 1);
        //        withoutBG = downsample(withoutBG, 1);
        //        frame = downsample(frame, 1);
        //        thresh_red = downsample(thresh_red, 1);
        //        thresh_green = downsample(thresh_green, 1);


        contours = contouring(thresh);
        drawing = combinedEllipse3(contours, thresh);

        estimation(finalEllipse[1].angle, finalEllipse[1].size.height / 2, finalEllipse[1].size.width / 2); //
        position_calculation();

        record_values();
        numOfEstimates++;

        //ellipse parameters are only diplayed once in every second
        if (count % 5 == 0) {
            display_values();
        }

        angles();

        //Esc key exits the loop
        char key = (char) waitKey(10);
        if (key == 27) break;
        else if (key == 'p') {
            imwrite("threshold2.jpg", thresh);
            imwrite("detection2.jpg", drawing);
            imwrite("frame2.jpg", frame);
        } else if (key == 'd') {
            cout << "diameter:" << (finalEllipse[1].size.height + finalEllipse[1].size.width) / 2 << endl;
        }

        //drawing the detected ellipse and the thresholded image
        imshow("Detection", drawing);
        thresh = downsample(thresh, 2);
        withoutBG = downsample(withoutBG, 2);
        withBG = downsample(withBG, 2);
        imshow("Final Threshold", thresh);
        // imshow("Saturated", hsv);
        // imshow("Without BG", withoutBG);
        //imshow("With BG", withBG);
        imshow("Frame", frame);
    }
    cap.release();

    estimatedHoopPos << x_avg/numOfEstimates,y_avg/numOfEstimates,z_avg/numOfEstimates, angleH_avg/numOfEstimates, dist_avg/numOfEstimates;
    cout << "hooppos: " << estimatedHoopPos << endl;
    return estimatedHoopPos;
}

void executePathPlanner() {
    Matrix3d init = Matrix3d::Zero();
    MatrixXd R(3,3);
    MatrixXd dist_corr_in(3,1);
    MatrixXd vel_corr_in(3,1);
    MatrixXd dist_corr_fin(3,1);
    MatrixXd vel_corr_fin(3,1);

    MatrixXd distanceBeforeHoop(3,1);
    MatrixXd velocityBeforeHoop(3,1);
    MatrixXd distanceAfterHoop(3,1);
    MatrixXd velocityAfterHoop(3,1);
    MatrixXd hoop_state(1,3);
    double orientation;

    distanceAfterHoop << 0, d_after, 0;
    velocityAfterHoop << 0, v_after, 0;
    MatrixXd hoop_pos(1,3);
    hoop_state = executeVision();
    orientation = hoop_state.coeff(0,3) * M_PI/180;
    //orientation = 30*M_PI/180;
    //hoop_pos << 1,2,3;
    hoop_pos = hoop_state.block<1,3>(0,0);
    distanceBeforeHoop << 0, d_before, 0;
    velocityBeforeHoop << 0, v_in, 0;

    R << cos(orientation), sin(orientation), 0, -sin(orientation), cos(orientation), 0, 0, 0, 1;
    dist_corr_in = R * distanceBeforeHoop;
    vel_corr_in = R * velocityBeforeHoop;
    dist_corr_fin = R * distanceAfterHoop;
    vel_corr_fin = R * velocityAfterHoop;

    MatrixXd p_before_hoop(2, 3);
    MatrixXd p_before_hoop1(1,3);
    MatrixXd p_before_hoop2(1,3);

    p_before_hoop1 << hoop_pos - dist_corr_in.transpose();
    p_before_hoop2 << vel_corr_in.transpose();
    p_before_hoop << p_before_hoop1 , p_before_hoop2;

    MatrixXd final(3, 3);
    MatrixXd final1(1, 3);
    MatrixXd final2(1, 3);
    final1 << hoop_pos + dist_corr_fin.transpose();
    final2 << vel_corr_fin.transpose();
    final << final1, final2, 0, 0, 0;

    double yaw0 = 0;
    double hoop_orient = orientation;
    MatrixXd r = Dimention3(init, p_before_hoop, final, hoop_pos, yaw0, hoop_orient);
    std::cout << r << std::endl;
}

int main() {
    executePathPlanner(); //no idea what the orientation should be rn.
}

//    VideoCapture cap(1); // open the default camera
//    if(!cap.isOpened())  // check if we succeeded
//        return -1;
//
//    namedWindow("edges",1);
//    for(;;)
//    {
//        Mat frame;
//        cap >> frame; // get a new frame from camera
//        //cvtColor(frame, edges, COLOR_BGR2RGB);
//        imshow("edges", frame);
//        if(waitKey(30) >= 0) break;
//    }
//    // the camera will be deinitialized automatically in VideoCapture destructor
//    return 0;


