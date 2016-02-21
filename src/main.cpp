// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <vector>

using namespace cv;
using namespace std;

int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

/**
 * computes the midpoint of a rectangle
 */

//A and B form a line
//C and D form a line
bool lineIntersection(
    double Ax, double Ay,
    double Bx, double By,
    double Cx, double Cy,
    double Dx, double Dy,
    double &X, double &Y) {

    double first_x = Ax >= Bx ? Ax : Bx;
    double first_y = Ax >= Bx ? Ay : By;

    double second_x = Ax >= Bx ? Bx : Ax;
    double second_y = Ax >= Bx ? By : Ay;

    if(abs(first_x - second_x) < 0.005)
        return false;
    double m1 = (first_y - second_y) / (first_x - second_x);
    double a1 = first_y - first_x*m1;

    first_x = Cx >= Dx ? Cx : Dx;
    first_y = Cx >= Dx ? Cy : Dy;

    second_x = Cx >= Dx ? Dx : Cx;
    second_y = Cx >= Dx ? Dy : Cy;

    if(abs(first_x - second_x) < 0.005)
        return false;

    double m2 = (first_y - second_y) / (first_x - second_x);
    double a2 = first_y - first_x*m2; 

    if(abs((m2 - m1)) < 0.005)
        return false;

    double x = (a2 - a1) / (m1 - m2);

    X = x;
    Y = a1 + m1*x;

    return true; 
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares, vector<Point2d> &midpoints )
{
    squares.clear();

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every c olor plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {

                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 ) 
                    {
                        squares.push_back(approx);
                    }
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow(wndname, image);
}

vector<vector<Point> > removeDuplicateRectangles(vector<vector<Point> > &squares, vector<Point2d> &midpoints) {

    vector<vector<Point> > uniqueRectangles;
    for(vector<Point> approx : squares) 
    {
        double x, y;
        lineIntersection(
         (double)approx[0].x, (double)approx[0].y,
         (double)approx[2].x, (double)approx[2].y, 
         (double)approx[1].x, (double)approx[1].y, 
         (double)approx[3].x, (double)approx[3].y, x, y);

        //cout << x << " " << y << endl;

        int errorMarginX = 10;
        int errorMarginY = 10;

        bool duplicate = false;
        for(auto &p : midpoints) 
        {
            if(abs(p.x-(int) x) < errorMarginX && abs(p.y-(int) y) < errorMarginY) 
            {
                duplicate = true;
                break;
            }
        }

        if(!duplicate) 
        {
            midpoints.push_back(Point(x, y));
            uniqueRectangles.push_back(approx);
        }
    }

    return uniqueRectangles;

}

vector<Vec4i> findLines(Mat &src, Mat &dst, Mat &color_dst)
{
  
    Canny( src, dst, 50, 200, 3 );
    cvtColor( dst, color_dst, COLOR_GRAY2BGR );

    vector<Vec4i> lines;
    HoughLinesP( dst, lines, 1, CV_PI/180, 80, 50, 5 );

    return lines;
}

void drawLines(vector<Vec4i> lines, Mat &dst, Mat &color_dst)
{

    for( size_t i = 0; i < lines.size(); i++ )
    {
        line( color_dst, Point(lines[i][0], lines[i][1]),
            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
    }

    namedWindow( "Detected Lines", 1 );
    imshow( "Detected Lines", color_dst );

}

void removeRedundantLines(vector<Vec4i> &lines, vector<vector<Point> > &squares)
{

    auto line = lines.begin();

    int radius = 150;
    while(line != lines.end()) 
    {
        bool removed = false;

        for(auto square : squares) 
        {
            for(int i = 0; i < 4; ++i) {

                if(sqrt(pow(abs(square[i].x - (*line)[0]), 2) + pow(abs(square[i].y - (*line)[1]), 2)) < radius &&
                    sqrt(pow(abs(square[(i+1)%4].x - (*line)[2]), 2) + pow(abs(square[(i+1)%4].y - (*line)[3]), 2)) < radius )
                {

                   line = lines.erase(line);
                   removed = true;
                   break;
                }
            }
        }

        if(!removed)
        {
            ++line;
        }
    }
}

void rescale_points(std::vector<Point2d>* v)
{
    auto iteratorX = std::max_element(v->begin(),
                                      v->end(),
                                      [](Point2d p1, Point2d p2)->bool{return p1.x < p2.x;}
                                      );
    auto iteratorY = std::max_element(v->begin(),
                                      v->end(),
                                      [](Point2d p1, Point2d p2)->bool{return p1.y < p2.y;}
                                      );

    double newMaxX = 100.0;
    double newMaxY = 100.0;

    for (auto &e : *v)
    {
        e.x = (iteratorX->x/(double)e.x) * newMaxX;
        e.y = (iteratorY->y/(double)e.y) * newMaxY;
    }
}

void removeFrameSizeRectangle(int rows, int cols, vector<vector<Point> > &rectangles)
{
    int errorMarginX = 20;
    int errorMarginY = 20;
    for(auto rect = rectangles.begin(); rect != rectangles.end();)
    {
        int maxWidth = 0;
        int maxHeight = 0;
        for(int i = 0; i < 4; ++i) 
        {

            maxWidth = max(abs((*rect)[i].x-(*rect)[(i+1)%4].x), maxWidth);
            maxHeight = max(abs((*rect)[i].y-(*rect)[(i+1)%4].y), maxHeight);

        }

        cout << rows << " " << cols << endl << maxWidth << " " << maxHeight << endl;
        if((maxWidth + errorMarginX) > cols && (maxHeight + errorMarginY) > rows)
        {
            rect = rectangles.erase(rect);
        } else {
            ++rect;
        }
    }
}

pair<pair<int, int>, pair<int, int>> findClosestSide(vector<vector<Point> > &rectangles, vector<Vec4i> &lines)
{

    for(int i = 0; i < lines.size(); ++i) 
    {
        for(int j = 0; j < rectangles.size(); ++j)
        {
            for(int k = j+1; k < rectangles.size(); ++k)
            {
                double X1;
                double Y1;
                double X2;
                double Y2;
                bool flag_one = false;
                bool flag_two = false;


                for(int l = 0; l < 4; ++l)
                {

                    if(lineIntersection(rectangles[j][l].x, rectangles[j][l].y, rectangles[j][(l+1)%4].x, rectangles[j][(l+1)%4].y,
                        lines[i][0], lines[i][1], lines[i][2], lines[i][3], X1, Y1))
                    {
                        flag_one = true;
                        break;
                    }
                }


                if(!flag_one)
                {
                     break;
                }

                for(int l = 0; l < 4; ++l)
                {

                    if(lineIntersection(rectangles[k][l].x, rectangles[k][l].y, rectangles[k][(l+1)%4].x, rectangles[k][(l+1)%4].y,
                        lines[i][0], lines[i][1], lines[i][2], lines[i][3], X2, Y2))
                    {
                        flag_two = true;
                        break;
                    }
                }

                if(!flag_two)
                {
                     break;
                }
                cout << X1 << " " << Y1 << endl;
                cout << X2 << " " << Y2 << endl;
            }

        }
    }

    pair<pair<int, int>, pair<int, int>> ans;
    return ans;

}

int main(int argc, char** argv)
{
    vector<Point2d> midpoints;
    namedWindow( wndname, 1 );
    vector<vector<Point> > squares;

    if (argc != 2)
    {
        std::cerr << "Invalid number of arguments!" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]);
    std::cout << "Read file: " << argv[1] << std::endl;
    if (image.empty())
    {
        std::cerr << "Could not read the provided image file!" << std::endl;
        return -1;
    }

    findSquares(image, squares, midpoints);
    
    squares = removeDuplicateRectangles(squares, midpoints);

    //drawSquares(image, squares);
    
    rescale_points(&midpoints);

    Mat dst, color_dst;
    std::vector<Vec4i> lines = findLines(image, dst, color_dst);

    //removeFrameSizeRectangle(image.rows, image.cols, squares);
    removeRedundantLines(lines, squares);
    drawLines(lines, dst, color_dst);

    int i = 0;
    std::ofstream file("./out.tex");
    file << "\\documentclass[tikz]{standalone}\n";
    file << "\\usepackage{tkz-euclide}\n\\usetkzobj{all}\n\\usetikzlibrary{positioning}\n\\usetikzlibrary{fit}\n\\usetikzlibrary{calc}\n";
    file << "\\begin{document}\n\\begin{tikzpicture}[]\n";
    for(auto p : midpoints) {
        file << "\\node[draw, rectangle, minimum size=3cm] ";
        file << "(" << i++ << ")" << " at (" << p.x << "," << p.y << "){};\n";
    }
    file << "\\end{tikzpicture}\n\\end{document}";

    findClosestSide(squares, lines);
    getchar();

    return 0;
}
