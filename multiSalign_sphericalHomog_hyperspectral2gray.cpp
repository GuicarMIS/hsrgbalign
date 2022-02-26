// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A simple example of using the Ceres minimizer.
//
// Minimize 0.5 (10 - x)^2 using jacobian matrix computed using
// automatic differentiation.

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "gflags/gflags.h"
//#include "pgm_image.h"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <ceres/cubic_interpolation.h>
#include "multiSresidue_sphericalHomog_hyperspectral2gray.h"
#include "RANSAC/sphericalRansac.hpp"

#include "npy.hpp"
#include <array>

DEFINE_string(desired, "", "File of the reference image");
DEFINE_string(input, "", "File of the image to align");
DEFINE_string(hsiazel, "", "File of the spherical coordinates of pixels HSI");
DEFINE_string(hsi_wavelengths, "", "File of the wavelengths captured as HSI");
DEFINE_string(hsi_vis_crf, "", "File of hyperspectral CRF in visible part (if empty, equql contribution of channels is considered to build the monochrome)");
DEFINE_string(output, "", "File to which the aligned image should be written");
DEFINE_string(thetaROI, "", "");
DEFINE_string(thetaKeypoints, "", "");
DEFINE_string(hsiKeypoints, "", "");
DEFINE_string(outputMappingOpenCVTxt, "", "File to which the mapping of HSI indices in the Theta frame should be written (text file, opencv format)");
DEFINE_string(outputMappingNpy, "", "File to which the mapping of HSI indices in the Theta frame should be written (npy file)");
DEFINE_double(intensityFact, 0, "");
DEFINE_bool(directOpt, true, "");
DEFINE_bool(displayUsedArea, false, "");

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

using namespace cv;

#define DI 0

#define PI 3.1416

#define MIN_INDEX 142
#define MAX_INDEX 804
#define MIN_WAVELENGTH 400
#define MAX_WAVELENGTH 700

// Creates the alignment problem.
void CreateProblem(const Mat& desired,
									 const Rect2d& area,
//                   const ceres::BiCubicInterpolator<ceres::Grid2D<double,1750> >& imageInterpolator,
                   const ceres::BiCubicInterpolator<ceres::Grid2D<double,1> >& imageInterpolator,
                   Problem* problem,
                   double *h,
                   double *c,
                   double &au_hsi, double &av_hsi, double &u0_hsi, double &v0_hsi)
{
	//pixel coordinates to spherical
	double au_equi = desired.cols*0.5/PI;
	double av_equi = 0.5*desired.rows/(0.5*PI);
	double u0_equi = desired.cols * 0.5;
	double v0_equi = desired.rows * 0.5;
	
	double Xs, Ys, Zs, phi, theta, sphi;
	//int index;
	
  for (unsigned int vd = DI ; vd < desired.rows-DI ; vd++)
  {
  	phi = (vd-v0_equi) / av_equi; //elevation
  	sphi = sin(phi);
    Zs = sin(phi);//cos(phi);
    
    for (unsigned int ud = DI ; ud < desired.cols-DI ; ud++)
    {
    	if(area.contains(Point(ud,vd)))//)(area[0] < ud) && (ud < (area[0]+area[2])) && (area[1] < vd) && (vd < (area[1]+area[3])))
    	{
		  	theta = (ud-u0_equi)/au_equi; // azimuth
    		Ys = sin(theta)*cos(phi);//sphi*sin(theta);
    		Xs = cos(phi)*cos(theta);//sphi*cos(theta);
		  	/*if(Xs == 0)
		  		continue;
		  		*/
		    //index = desired.LinearIndex(ud,vd);
		    ceres::CostFunction* cost_function = multiSresidue::Create(au_hsi, av_hsi, u0_hsi, v0_hsi,
		        Xs, Ys, Zs, desired.at<double>(vd, ud), imageInterpolator);

		    problem->AddResidualBlock(
		        cost_function, 
		        NULL /* squared loss */, //new ceres::CauchyLoss(0.5), //
		        h,
		        c);//, aligned->MutablePixelFromLinearIndex(index));
		        
      }
    }
  }
    
   problem->SetParameterization(h, new ceres::HomogeneousVectorParameterization(9));
   
   //problem->SetParameterBlockConstant(c);
   //problem->SetParameterBlockConstant(h);

}


void SolveProblem(Problem* problem, double *h) {

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true; 
  options.function_tolerance = 1e-12;
  options.max_num_iterations = 500;
  options.initial_trust_region_radius = 1e1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);
  std::cout << summary.FullReport() << "\n";

  std::cout << "h : " << h[0] << " " << h[1] << " " << h[2] << std::endl;
  std::cout << "    " << h[3] << " " << h[4] << " " << h[5] << std::endl;
  std::cout << "    " << h[6] << " " << h[7] << " " << h[8] << std::endl;
}


int main(int argc, char** argv) {
	bool hsi_crf = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_desired.empty()) {
    std::cerr << "Please provide a reference image file name using -desired.\n";
    return 1;
  }

  if (FLAGS_input.empty()) {
    std::cerr << "Please provide an image file name using -input.\n";
    return 1;
  }

  if (FLAGS_hsiazel.empty()) {
    std::cerr << "Please provide an azimuth elevation file name using -hsiazel.\n";
    return 1;
  }


  if (FLAGS_output.empty()) {
    std::cerr << "Please provide an output file name using -output.\n";
    return 1;
  }
  
  if (FLAGS_hsi_vis_crf.empty()) {
    std::cerr << "No HSI CRF provided.\n";
    hsi_crf = false;
  }  
  
  
  
  int max_index, min_index;
  if (FLAGS_hsi_wavelengths.empty()) {
    std::cerr << "No HSI wavelengths provided.\n";
    max_index = MAX_INDEX;
    min_index = MIN_INDEX;
  }  
  else
  {
  	//load npy format wavelegnths data
  	std::vector<unsigned long> shape;
  	std::vector<int> shapei;
  	bool fortran_order;
  	std::vector<double> data;
  	npy::LoadArrayFromNumpy(FLAGS_hsi_wavelengths, shape, fortran_order, data);
  	
  	max_index = 0;
  	min_index = 0;
  	int i = -1;
  	while( (++i < shape[0]) && (data[i] < MIN_WAVELENGTH) );
  	min_index = i;
  	while( (++i < shape[0]) && (data[i] < MAX_WAVELENGTH) );
  	max_index = i-1;	
  }
  
  int nb_index=max_index-min_index;
  
  std::cout << "wavelength indexes: " << min_index << " " << max_index << " " << nb_index << std::endl;
  
  
  

  // Read the images
  Mat desired_Gray = imread(FLAGS_desired, IMREAD_GRAYSCALE);
  
  //ceres::examples::PGMImage<double> desired(FLAGS_desired);
  if (desired_Gray.cols == 0) {
    std::cerr << "Reading \"" << FLAGS_desired << "\" failed.\n";
    return 3;
  }
  
  //load npy format hyperspectral data
  std::vector<unsigned long> shape;
  std::vector<int> shapei;
  bool fortran_order;
  std::vector<double> data;
  npy::LoadArrayFromNumpy(FLAGS_input, shape, fortran_order, data);

  std::cout << "shape: ";
  for (size_t i = 0; i<shape.size(); i++)
  {
  	shapei.push_back(shape[i]);
    std::cout << shapei[i] << ", ";
  }
  std::cout << std::endl;
  
  std::cout << data.size() << std::endl;
   
  //Mat image = imread(FLAGS_input, IMREAD_COLOR); //, IMREAD_GRAYSCALE);
  Mat image = Mat(shapei, CV_64F, data.data());
  
  //ceres::examples::PGMImage<double> image(FLAGS_input);
  if (image.cols == 0) {
    std::cerr << "Reading \"" << FLAGS_input << "\" failed.\n";
    return 3;
  }
  
  //load hsi CRF
    double fact_c = 1; // // 120 for SouthPortal ; 40 for XVIII ; 40 for III
    
    
    if (FLAGS_intensityFact != 0) {
	  	fact_c = FLAGS_intensityFact;
  }
    
  	double *w = new double[nb_index];
    double *c = new double[nb_index];
    //double affine_photometric_model[2] = {1./1000., 0}; // 1/1000 for SouthPortal ; 1/5000 for XVIII ; 1/5000 for III
    //double affine_photometric_model[2] = {1./(nb_index*0.1), 0}; // 1/1000 for SouthPortal ; 0.2 for XVIII ; 0.2 for III
    double affine_photometric_model[2] = {fact_c*1./(257.*nb_index), 0};
    //double affine_photometric_model[2] = {1./nb_index, 0};
    if(!hsi_crf)
    {
    	for(unsigned int i = 0 ; i < nb_index ; i++)
  			c[i] = affine_photometric_model[0];
  /*
  for(unsigned int i = 0 ; i < 892 ; i++)
  	c[i] = 1/892.0;
  for(unsigned int i = 892 ; i < 1750 ; i++)
  	c[i] = 0;
  */
  	}
  	else
  	{
  		std::ifstream fic(FLAGS_hsi_vis_crf);
  		for(unsigned int i = 0 ; i < nb_index ; i++)
  		{
  			fic >> w[i] >> c[i]; 
  			c[i] *= affine_photometric_model[0];
  		}
  		fic.close();
  	}
  
  
  namedWindow("Input", WINDOW_AUTOSIZE );
  //imshow("Input", image);
  
  namedWindow("Desired", WINDOW_AUTOSIZE );
  imshow("Desired", desired_Gray);

	Rect2d area;
	if (FLAGS_thetaROI.empty()) {
	  std::cerr << "manual selection of an area in teh desired image also visible in the HSI.\n";
	  area = selectROI("Desired", desired_Gray, false, false);
  }
  else
  {
  	std::ifstream fic(FLAGS_thetaROI);
  	fic >> area.x >> area.y >> area.width >> area.height;
  	fic.close();
  }
  //Rect2d area = Rect2d(302, 27, 64, 120);
  std::cout << "area clicked: " << area.x << " " << area.y << " " << area.width << " " << area.height << std::endl;
  
  Mat desired;
  desired_Gray.convertTo(desired, CV_64F);
  //image.convertTo(image, CV_64F);
  
  Mat aligned(desired.rows, desired.cols, CV_64FC1, Scalar(0.0));
  Mat aligned_hsi_inds(desired.rows, desired.cols, CV_32SC1, Scalar(-1));
  //ceres::examples::PGMImage<double> aligned(image.width(), image.height());
  //aligned.Set(0.0);

  // Create the data term
  //double h[8] = {1, 0, 0, 0, 1, 0, 0, 0}; //desired -> current
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> H;
  H.setIdentity();  
  double *h = H.data();
  
  double au_equi = desired.cols*0.5/PI;
	double av_equi = 0.5*desired.rows/(0.5*PI);
	double u0_equi = desired.cols * 0.5;
	double v0_equi = desired.rows * 0.5;

  std::vector<int> shapei_hsi_angles;
  std::vector<double> data_hsi_angles;
  npy::LoadArrayFromNumpy(FLAGS_hsiazel, shape, fortran_order, data_hsi_angles);

	std::cout << "shape hsiazel: ";	
	for (size_t i = 0; i<shape.size(); i++)
  {
  	shapei_hsi_angles.push_back(shape[i]);
    std::cout << shapei_hsi_angles[i] << ", ";
  }
  std::cout << std::endl;
  
  double hsi_azMin = 360;//2*PI;
  double hsi_azMax = -360;//-2*PI;
	double hsi_elMin = 360;//2*PI;
  double hsi_elMax = -360;;//-2*PI;
  double az, el;
  std::vector<double>::iterator it_data_hsi_angles = data_hsi_angles.begin();
  double Ihsi[1750];
  
  Mat hsi_cart(desired.rows, desired.cols, CV_64FC1, Scalar(0.0));
  Mat hsi_inds(desired.rows, desired.cols, CV_32SC1, Scalar(-1));
  double u, v;
  
  
  //find min/max hsi angles
  for (size_t i = 0; i<shapei_hsi_angles[0]; i++)
  {
  	double az_prec = *it_data_hsi_angles*PI/180.;
  	double trigshift = 0.;
  	//std::cout << *it_data_hsi_angles*PI/180. << std::endl;
		for (size_t j = 0; j<shapei_hsi_angles[1]; j++, it_data_hsi_angles++)
		{       
			az = *it_data_hsi_angles*PI/180.;//-PI;
			if(az < az_prec)
				trigshift = 2*PI;
			
			az = az + trigshift;
			az_prec = az;
			
			az -= PI;
			
			if(az > hsi_azMax)
				hsi_azMax = az;
			else
				if(az < hsi_azMin)
					hsi_azMin = az;
					
			it_data_hsi_angles++;
			el = -(*it_data_hsi_angles*PI/180.);
			if(el > hsi_elMax)
				hsi_elMax = el;
			else
				if(el < hsi_elMin)
					hsi_elMin = el;
		}
	}
	
	std::cout << "captured hsi_azMin/max: " << hsi_azMin << " " << hsi_azMax << std::endl;
	
	std::cout << "captured hsi_elMin/max: " << hsi_elMin << " " << hsi_elMax << std::endl;
  
  double *HSI=data.data();
  unsigned int indexInHSI = 0;
  it_data_hsi_angles = data_hsi_angles.begin();
  for (size_t i = 0; i<shapei_hsi_angles[0]; i++)
  {
  	double Xs, Ys, Zs, tmp;
		for (size_t j = 0; j<shapei_hsi_angles[1]; j++, it_data_hsi_angles++, HSI+=1750, indexInHSI++)
		{       
			az = *it_data_hsi_angles*PI/180.-PI;		
			it_data_hsi_angles++;
			el = -(*it_data_hsi_angles*PI/180.);
			
		  
		  //az, el local to the robot eye 
			//u = au_equi*az + u0_equi;
		  //v = av_equi*el + v0_equi;
		  
		  //az, el local to the robot eye tripod
		  az-=(hsi_azMax+hsi_azMin)*0.5;//PI*0.5; 
		  //az, el to Xs, Ys, Zs
		  Zs = sin(el);//sphi;//cos(phi);  
   		Ys = cos(el)*sin(az);//sphi*sin(theta);
   		Xs = cos(el)*cos(az);//sphi*cos(theta);
		  
		  //rotate Xs, Ys, Zs around X of 90 degrees
		  tmp = Zs;
		  Zs = -Ys;
		  Ys = tmp;
		  
		  //Xs, Ys, Zs to az, el
		  az = atan2(Ys, Xs); //Azimuth entre -pi et pi
    	el = asin(Zs);//		  
		  
		  u = au_equi*az + u0_equi;
		  v = av_equi*el + v0_equi;
			//u = au_equi*(el) + u0_equi;
		  //v = av_equi*((-az)) + v0_equi; //+PI*0.75		  
		  
		  if((u > -1) && (u < desired.cols) && (v > -1) && (v < desired.rows))
		  {
		  //std::cout << u << " " << v << std::endl;
		  
		  double brigthness = ((HSI[0+min_index]) * c[0]);  //sqrt of HSI
		    
		    for(unsigned int iChannel = 1 ; iChannel < nb_index ; iChannel++) //visible channels only 390nm <-> index 119 / 700 nm <-> index 804
		    	brigthness += ((HSI[iChannel+min_index]) * c[iChannel]); //sqrt of HSI
		    
		    hsi_cart.at<double>(v, u) = (brigthness>255)?255:brigthness;// * c[0];
		    //hsi_cart.at<double>(v, u) = brigthness;// * c[0];
		    
		    hsi_inds.at<int>(v, u) = indexInHSI;
		    
		  }
  	}
  }
  
  affine_photometric_model[0] = 1;
  
  Mat hsi_cart_Gray;
  hsi_cart.convertTo(hsi_cart_Gray, CV_8UC1);
   namedWindow("hsi_cart_Gray", WINDOW_AUTOSIZE );
   imshow("hsi_cart_Gray", hsi_cart_Gray);
   waitKey(0);
   
   imwrite("hsi_cart.png", hsi_cart);
  
  std::cout << "fov bounds: " << hsi_azMin << " " << hsi_azMax << " " << hsi_elMin << " " << hsi_elMax << std::endl;
  
  /*
  unsigned int totalElements = shapei[0]*shapei[1]*shapei[2];//image.total();//*image.channels(); // Note: image.total() == rows*cols.
  cv::Mat flat_image = image.(1, totalElements); // 1xN mat of 1 channel, O(1) operation
  if(!image.isContinuous()) {
        flat_image = flat_image.clone(); // O(N),
    }
    
    std::cout << "nb data "<< flat_image.total() << std::endl;
  
  ceres::Grid2D<double, 1750> array(flat_image.ptr<double>(0), 0, shapei[0], 0, shapei[1]);
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1750> > imageInterpolator(array);
  */
  unsigned int totalElements = hsi_cart.cols*hsi_cart.rows;//shapei[0]*shapei[1];
  cv::Mat flat_image = hsi_cart.reshape(1, totalElements); // 1xN mat of 1 channel, O(1) operation
  if(!hsi_cart.isContinuous()) {
        flat_image = flat_image.clone(); // O(N),
    }
    
    std::cout << "nb data "<< flat_image.total() << std::endl;
  
  ceres::Grid2D<double, 1> array(flat_image.ptr<double>(0), 0, hsi_cart.rows, 0, hsi_cart.cols);
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > imageInterpolator(array);
  
  it_data_hsi_angles = data_hsi_angles.begin();

	double au_hsi = shapei[1]/((PI/180.)*(hsi_azMax-hsi_azMin));
	double av_hsi = shapei[0]/((PI/180.)*(hsi_elMax-hsi_elMin));
	double u0_hsi = shapei[1] * 0.1;//-au_hsi*(PI/180.)*(*it_data_hsi_angles-180);//
	it_data_hsi_angles = data_hsi_angles.end();
	it_data_hsi_angles--;
	double v0_hsi = shapei[0] * 0.1;//-av_hsi*(PI/180.)*(*it_data_hsi_angles);//0;
	
	std::cout << "hsi camera: " << au_hsi << " " << av_hsi << " " << u0_hsi << " " << v0_hsi << std::endl;
	
	//Estimate initial H from clicked points in Equirectangular image and hyperspectral image corners (assuming the clicked points match -> need to show the hyperspectral image first TODO)
	//pixel coordinates to spherical
	
	//with these u0, v0: Xs points "forward", Zs points "downward", Ys points "to the right"
	double Xs, Ys, Zs, phi, theta, sphi;
	double Xsd, Ysd, Zsd, phid, thetad, sphid;
	
  //clicked points in Equirectangular image  transformation to cartesian spherical
  //display images and get clicks
  
  //double u_clicked[4] = {area.x,             area.x, area.x+area.width, area.x+area.width};
  //double v_clicked[4] = {area.y+area.height, area.y,            area.y, area.y+area.height};
  double *u_clicked;//[4] = {302, 308, 366, 365};
  double *v_clicked;//[4] = {133, 49,  51, 134};
  double trapeze = shapei[0]*0.4;
	//double u_hsi[4] = {0., shapei[1]-1, shapei[1]-1, 0.};
  //double v_hsi[4] = {0.,        0.+shapei[0]*0.3, shapei[0]*0.8, shapei[0]-1};
  double *u_hsi;//[4] = {464, 572, 573, 463};
  double *v_hsi;//[4] = {135, 155, 186, 205};
  int nbBestMatch = 4;
  
  bool matchesFromFile = true;
  
	if (FLAGS_thetaKeypoints.empty()) {
	  std::cerr << "***AKAZE-DESCRIPTOR_KAZE_UPRIGHT*** descriptor in Theta image (thus HSI too) for ***BruteForce*** matching.\n";
		
		matchesFromFile = false;
  }
  else
  {
  		u_clicked = new double[nbBestMatch];
  		v_clicked = new double[nbBestMatch];
	  	std::ifstream fic(FLAGS_thetaKeypoints);
  		for(unsigned int i = 0 ; i < 4 ; i++)
  		{
  			fic >> u_clicked[i] >> v_clicked[i]; 
  		}
  		fic.close();	
  }
  
	if (FLAGS_hsiKeypoints.empty()) {
	 	std::cerr << "***AKAZE-DESCRIPTOR_KAZE_UPRIGHT*** descriptor in HSI (thus Theta image too) for ***BruteForce*** matching.\n";
		
		matchesFromFile = false;
  }
  else
  {
  		u_hsi = new double[nbBestMatch];
  		v_hsi = new double[nbBestMatch];
	  	std::ifstream fic(FLAGS_hsiKeypoints);
  		for(unsigned int i = 0 ; i < 4 ; i++)
  		{
  			fic >> u_hsi[i] >> v_hsi[i]; 
  		}
  		fic.close();	
  }
  
  // keypoint  for img1 and img2
      std::vector<KeyPoint> keyImg1, keyImg2;
  //matches
  	std::vector<DMatch> bestMatches;
  	// matched points
    std::vector<Point2f> matchImg1, matchImg2;
    
  if(!matchesFromFile)
  {
  	Ptr<Feature2D> b; // features
  	
  	Ptr<DescriptorMatcher> descriptorMatcher;
    
    // Match between img1 and img2
    std::vector<DMatch> matches;
    // Descriptor for img1 and img2
    Mat descImg1, descImg2;
        
  	b = AKAZE::create(AKAZE::DESCRIPTOR_KAZE_UPRIGHT);
  	
  	//detect and compute descriptors in one step
    b->detectAndCompute(hsi_cart_Gray, Mat(),keyImg1, descImg1,false);
    //detect and compute descriptors in one step
    b->detectAndCompute(desired_Gray, Mat(),keyImg2, descImg2,false);
    
    //matching
    descriptorMatcher = DescriptorMatcher::create("BruteForce");
    
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
    // Keep best matches only
    // We sort distance between descriptor matches
    Mat index;
    int nbMatch=int(matches.size());
    Mat tab(nbMatch, 1, CV_32F);
    for (int i = 0; i<nbMatch; i++)
    {
        tab.at<float>(i, 0) = matches[i].distance;
    }
    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    
    nbBestMatch = 60;
    nbBestMatch = (nbBestMatch<=nbMatch)?nbBestMatch:nbMatch;
    for (int i = 0; i<nbBestMatch; i++)
    {
        bestMatches.push_back(matches[index.at<int>(i, 0)]);
    }
    Mat result;
    drawMatches(hsi_cart_Gray, keyImg1, desired_Gray, keyImg2, bestMatches, result);
    namedWindow("60 best matches", WINDOW_AUTOSIZE);
    imshow("60 best matches", result);
		imwrite("60BestMatches.png", result);
		
		u_clicked = new double[nbBestMatch];
  	v_clicked = new double[nbBestMatch];
  	u_hsi = new double[nbBestMatch];
  	v_hsi = new double[nbBestMatch];
  	
  	std::vector<DMatch>::iterator it;
  	unsigned int i = 0;
  	Point2f matchPoint;
    for (it = bestMatches.begin(); it != bestMatches.end(); ++it, i++)
    {
    		u_clicked[i] = keyImg2[it->trainIdx].pt.x;
    		v_clicked[i] = keyImg2[it->trainIdx].pt.y;
    		
    		matchPoint.x = u_clicked[i];
    		matchPoint.y = v_clicked[i];
    		matchImg2.push_back(matchPoint);
    		
    		u_hsi[i] = keyImg1[it->queryIdx].pt.x;
    		v_hsi[i] = keyImg1[it->queryIdx].pt.y;
    		
    		matchPoint.x = u_hsi[i];
    		matchPoint.y = v_hsi[i];
    		matchImg1.push_back(matchPoint);
    }
		
		/*
		//select the four matches among the 20 bests that are the best spread
		//1) compute centroid in Theta image
		std::vector<DMatch>::iterator it;
		double u_clicked_c = 0., v_clicked_c = 0.;
    for (it = bestMatches.begin(); it != bestMatches.end(); ++it)
    {
    		u_clicked_c += keyImg2[it->trainIdx].pt.x;
    		v_clicked_c += keyImg2[it->trainIdx].pt.y;
    }
    u_clicked_c /= bestMatches.size();
    v_clicked_c /= bestMatches.size();
    
    
    //2) sort the matches in descending order of the distance of Theta image keypoints to their centroids
    Mat tab_dist(nbBestMatch, 1, CV_32F);
    int i = 0;
    double dx, dy;
    for (it = bestMatches.begin(); it != bestMatches.end(); ++it, i++)
    {
    	dx = keyImg2[it->trainIdx].pt.x-u_clicked_c;
    	dy = keyImg2[it->trainIdx].pt.y-v_clicked_c;
      tab_dist.at<float>(i, 0) = dx*dx + dy*dy;
    }
    
    Mat index_dist;
    sortIdx(tab_dist, index_dist, SORT_EVERY_COLUMN + SORT_DESCENDING);
    
    std::vector<DMatch> matchesForHomog;
    int iInd = 0;
    for (int i = 0; i<4; i++)
    {
    		bool keepSearching = true;
    		while(keepSearching)
    		{
		  		if(i > 0)
		  		{
		  			dx = keyImg2[bestMatches[index_dist.at<int>(iInd, 0)].trainIdx].pt.x - u_clicked[i-1];
		  			dy = keyImg2[bestMatches[index_dist.at<int>(iInd, 0)].trainIdx].pt.y - v_clicked[i-1];
		  			if((dx*dx + dy*dy) > 100) // 10*10
		  				keepSearching = false;
		  			else
		  				if(iInd < (bestMatches.size()-1))
		  					iInd++;
		  				else
		  					keepSearching = false;
		  		}
		  		else
		  			keepSearching = false;
    		}
    		
    		matchesForHomog.push_back(bestMatches[index_dist.at<int>(iInd, 0)]);
    
        u_clicked[i] = keyImg2[bestMatches[index_dist.at<int>(iInd, 0)].trainIdx].pt.x;
        v_clicked[i] = keyImg2[bestMatches[index_dist.at<int>(iInd, 0)].trainIdx].pt.y;
        
        u_hsi[i] = keyImg1[bestMatches[index_dist.at<int>(iInd, 0)].queryIdx].pt.x;
        v_hsi[i] = keyImg1[bestMatches[index_dist.at<int>(iInd, 0)].queryIdx].pt.y;
        
        iInd++;
    }
    
    Mat result4;
    drawMatches(hsi_cart_Gray, keyImg1, desired_Gray, keyImg2, matchesForHomog, result4);
    namedWindow("kept matches for Homog", WINDOW_AUTOSIZE);
    imshow("kept matches for Homog", result4);
		imwrite("keptMatchesForHomog.png", result4);
		*/
		
		
  }
  
  
  Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> XsTheta, XsHsi;
  XsTheta = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>::Zero(3, nbBestMatch);
  XsHsi = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>::Zero(3, nbBestMatch);
  /*
  Eigen::Matrix<double, 8, 8, Eigen::RowMajor> A;
  
  Eigen::Matrix<double, 8, 1> b;
  
  for(unsigned int i = 0 ; i < 4 ; i++)
  {
  	std::cout << "clicked: " << u_clicked[i] << " " << v_clicked[i] << std::endl;	
    phi = (v_clicked[i]-v0_equi) / av_equi;//elev
  	sphi = sin(phi);
    XsTheta(2,i) = sphi;//cos(phi);  
 
   	theta = (u_clicked[i]-u0_equi) / au_equi;//az
   	XsTheta(1,i) = cos(phi)*sin(theta);//sphi*sin(theta);
   	XsTheta(0,i) = cos(phi)*cos(theta);//sphi*cos(theta);
   	//XsTheta(2,i) = cos(phi)*cos(theta);//cos(phid);  
    //XsTheta(1,i) = sin(phi);//sphid*sin(thetad);
   	//XsTheta(0,i) = cos(phi)*sin(theta);//sphid*cos(thetad);
   	
   	std::cout << "hsi: " << u_hsi[i] << " " << v_hsi[i] << std::endl;	
   	//phid = (v_hsi[i]-v0_hsi)/av_hsi;
   	phid = (v_hsi[i]-v0_equi)/av_equi;
   	std::cout << "phid: " << phid << std::endl;
   	sphid = sin(phid);
    
    
    //thetad = (u_hsi[i]-u0_hsi)/au_hsi;
    thetad = (u_hsi[i]-u0_equi)/au_equi;
      	
    std::cout << "thetad: " << thetad << std::endl;
      
    XsHsi(2,i) = sin(phid);cos(phid)*cos(thetad);////cos(phid);  
    XsHsi(1,i) = cos(phid)*sin(thetad);//sin(phid);//sphid*sin(thetad);
   	XsHsi(0,i) = cos(phid)*cos(thetad);//cos(phid)*sin(thetad);//sphid*cos(thetad);
   	
   	A(2*i,0) = XsTheta(0,i)*XsHsi(2,i); A(2*i,1) = XsTheta(1,i)*XsHsi(2,i); A(2*i,2) = XsTheta(2,i)*XsHsi(2,i); 
   	A(2*i+1,3) = -XsTheta(0,i)*XsHsi(2,i); A(2*i+1,4) = -XsTheta(1,i)*XsHsi(2,i); A(2*i+1,5) = -XsTheta(2,i)*XsHsi(2,i); 
   	A(2*i,6) = -XsTheta(0,i)*XsHsi(0,i); A(2*i,7) = -XsTheta(1,i)*XsHsi(0,i); 
   	A(2*i+1,6) = XsTheta(0,i)*XsHsi(1,i); A(2*i+1,7) = XsTheta(1,i)*XsHsi(1,i); 
   	
   	b(2*i) = XsHsi(0,i)*XsTheta(2,i);
   	b(2*i+1) = -XsHsi(1,i)*XsTheta(2,i);
  }
  
  std::cout << A << std::endl;
  std::cout << b << std::endl;
  
  //Eigen::Matrix<double, 8, 1> h_ = (A.transpose()*A).inverse()*A.transpose()*b;
  Eigen::Matrix<double, 8, 1> h_ = A.inverse()*b;
   memcpy(h, h_.data(), 8*sizeof(double));
  std::cout << "other H: " << std::endl << h_ << std::endl;
  h[8] = 1;
  */
  
  //Eigen::Matrix<double, 3*nbBestMatch, 9, Eigen::RowMajor> A = Eigen::Matrix<double, 3*nbBestMatch, 9, Eigen::RowMajor>::Zero(3*nbBestMatch,9);
  
  Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor> A;
  A = Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>::Zero(3*nbBestMatch,9);
  
  // spherical keypoints for img1 and img2
  std::vector<cv::Point3f> XsImg1, XsImg2;
    
  for(unsigned int i = 0 ; i < nbBestMatch ; i++)
  {
  	std::cout << "clicked: " << u_clicked[i] << " " << v_clicked[i] << std::endl;	
    phi = (v_clicked[i]-v0_equi) / av_equi;//elev
  	sphi = sin(phi);
    XsTheta(2,i) = sin(phi);//sphi;//cos(phi);  
 
   	theta = (u_clicked[i]-u0_equi) / au_equi;//az
   	XsTheta(1,i) = cos(phi)*sin(theta);//sphi*sin(theta);
   	XsTheta(0,i) = cos(phi)*cos(theta);//sphi*cos(theta);
   	
   	XsImg1.push_back(cv::Point3f(XsTheta(0,i), XsTheta(1,i), XsTheta(2,i)));
   	
   	/*XsTheta(2,i) = cos(phi)*cos(theta);//cos(phid);  
    XsTheta(1,i) = sin(phi);//sphid*sin(thetad);
   	XsTheta(0,i) = cos(phi)*sin(theta);//sphid*cos(thetad);
   	*/
   	std::cout << "hsi: " << u_hsi[i] << " " << v_hsi[i] << std::endl;	
   	
   	std::cout << "theta phi before: " << theta << " " << phi << std::endl;
   	theta = atan2(XsTheta(1,i), XsTheta(0,i)); //Azimuth entre -pi et pi
    phi = asin(XsTheta(2,i));//
   	std::cout << "theta phi after: " << theta << " " << phi << std::endl;
   	
   	std::cout << "recomputed u, v : " << au_equi*theta+u0_equi << " " << av_equi*phi+v0_equi << std::endl;	
   	
   	//phid = (v_hsi[i]-v0_hsi)/av_hsi;
   	phid = (v_hsi[i]-v0_equi)/av_equi;
   	std::cout << "phid: " << phid << std::endl;
   	sphid = sin(phid);
    
    
    //thetad = (u_hsi[i]-u0_hsi)/au_hsi;
    thetad = (u_hsi[i]-u0_equi)/au_equi;
      	
    std::cout << "thetad: " << thetad << std::endl;
      
    XsHsi(2,i) = sin(phid);//cos(phid)*cos(thetad);////cos(phid);  
    XsHsi(1,i) = cos(phid)*sin(thetad);//sin(phid);//sphid*sin(thetad);
   	XsHsi(0,i) = cos(phid)*cos(thetad);//cos(phid)*sin(thetad);//sphid*cos(thetad);
   	
   	XsImg2.push_back(cv::Point3f(XsHsi(0,i), XsHsi(1,i), XsHsi(2,i)));
   	
   	A(3*i,3) = -XsTheta(0,i)*XsHsi(2,i); A(3*i,4) = -XsTheta(1,i)*XsHsi(2,i); A(3*i,5) = -XsTheta(2,i)*XsHsi(2,i); A(3*i,6) = XsTheta(0,i)*XsHsi(1,i); A(3*i,7) = XsTheta(1,i)*XsHsi(1,i); A(3*i,8) = XsTheta(2,i)*XsHsi(1,i); 
   	A(3*i+1,0) = XsTheta(0,i)*XsHsi(2,i); A(3*i+1,1) = XsTheta(1,i)*XsHsi(2,i); A(3*i+1,2) = XsTheta(2,i)*XsHsi(2,i); A(3*i+1,6) = -XsTheta(0,i)*XsHsi(0,i); A(3*i+1,7) = -XsTheta(1,i)*XsHsi(0,i); A(3*i+1,8) = -XsTheta(2,i)*XsHsi(0,i); 
   	A(3*i+2,0) = -XsTheta(0,i)*XsHsi(1,i); A(3*i+2,1) = -XsTheta(1,i)*XsHsi(1,i); A(3*i+2,2) = -XsTheta(2,i)*XsHsi(1,i); A(3*i+2,3) = XsTheta(0,i)*XsHsi(0,i); A(3*i+2,4) = XsTheta(1,i)*XsHsi(0,i); A(3*i+2,5) = XsTheta(2,i)*XsHsi(0,i);
  }
  
  std::cout << "A: " << A << std::endl;
  
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd V=svd.matrixV();
  
  std::cout << "V: " << V << std::endl;
  
  for(unsigned int i = 0; i<9 ; i++)
  	h[i] = V(i,8);
 
  std::cout << "other H: " << std::endl << V << std::endl;

  /*
  h[0] = 1;
  h[4] = cos(-0.5*PI);
  h[5] = -sin(-0.5*PI);
  h[7] = sin(-0.5*PI);
  h[8] = cos(-0.5*PI);
  
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R;
  R(0,0) = cos(-0.25*PI);
  R(0,1) = -sin(-0.25*PI);
  R(1,0) = sin(-0.25*PI);
  R(1,1) = cos(-0.25*PI);
  R(2,2) = 1;
  
  H = R*H;
  H /= sqrt(h[0]*h[0]+h[1]*h[1]+h[2]*h[2]);//H(2,2);
  */
  std::cout << "XsHsi: " << std::endl << XsHsi << std::endl;
  std::cout << "XsTheta: " << std::endl << XsTheta << std::endl;
  
  //H = XsHsi*(XsTheta.transpose()*(XsTheta*XsTheta.transpose()).inverse());
  std::cout << "H: " << std::endl << H << std::endl;
  
   if(!matchesFromFile)
   {
		//WATCH OUT: matchImg1: hsi ; XsImg1: theta...
		//Mat Hr = findHomography(matchImg2, matchImg1, RANSAC); 
		Mat InliersMask;
		Mat Hr = findSphericalHomography(XsImg1, XsImg2, RANSAC, 0.005, InliersMask, 2000, 0.995); 

		//Extract inliers
		std::vector<DMatch> inliersH;

		unsigned int i = 0;
		std::vector<DMatch>::iterator it;
		for (it = bestMatches.begin(); it != bestMatches.end(); ++it, i++)
		{
			if(InliersMask.at<unsigned char>(i))
			{
				inliersH.push_back(*it);
			}
		}


		Mat result;
		drawMatches(hsi_cart_Gray, keyImg1, desired_Gray, keyImg2, inliersH, result);
		namedWindow("InliersH", WINDOW_AUTOSIZE);
		imshow("InliersH", result);
		imwrite("InliersH.png", result);


		std::cout << "H ransac: " << std::endl << Hr << std::endl;

		for(unsigned int i = 0; i<9 ; i++)
			h[i] = Hr.at<double>(i);
    }
   
   std::cout << "H ransac copied: " << std::endl << H << std::endl;
  
  Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> XsTheta2Hsi, XsHsi2Theta; 
  XsTheta2Hsi = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>::Zero(3, nbBestMatch);
  XsHsi2Theta = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>::Zero(3, nbBestMatch);
  XsTheta2Hsi = H*XsTheta;
  for(unsigned int i = 0 ; i < nbBestMatch ; i++)
  {
  	double rho = sqrt(XsTheta2Hsi(0,i)*XsTheta2Hsi(0,i) + XsTheta2Hsi(1,i)*XsTheta2Hsi(1,i) + XsTheta2Hsi(2,i)*XsTheta2Hsi(2,i));
  	for(unsigned int j = 0 ; j < 3 ; j++)
  		XsTheta2Hsi(j,i) /= rho;
  }
  
  XsHsi2Theta = H.inverse()*XsHsi;
  for(unsigned int i = 0 ; i < nbBestMatch ; i++)
  {
  	double rho = sqrt(XsHsi2Theta(0,i)*XsHsi2Theta(0,i) + XsHsi2Theta(1,i)*XsHsi2Theta(1,i) + XsHsi2Theta(2,i)*XsHsi2Theta(2,i));
  	for(unsigned int j = 0 ; j < 3 ; j++)
  		XsHsi2Theta(j,i) /= rho;
  }
  
  std::cout << "H*XsTheta: " << std::endl << XsTheta2Hsi << std::endl;
   std::cout << "Hinv*XsHsi: " << std::endl <<  XsHsi2Theta << std::endl;
   
     waitKey(0);
   
  //H /= H(2,2);
  
  /*
  H <<     -1.38805, -0.0246192,   -1.30031,
 0.0562629,   -2.04767,  0.0154979,
   -1.0625, -0.0144028,          1;
  */ 
  /* 
  H <<   -1.14997,  0.0373661,    1.32381,
  0.267258,    2.39974,  -0.265374,
   1.09094, -0.0101195,          1;
*/

/*
	//eighth
	H <<    0.380807,  -0.0111283,    0.518252,
   0.417945,  -0.0276997,   -0.247773,
  0.0234423,    0.590669, 0.000442264;
*/
/*
	//quarter
	H <<      0.378492,  -0.00441199,     0.523159,
    0.417921,   -0.0108133,    -0.244614,
   0.0250558,     0.589736, -0.000943719;
*/

  ceres::Problem problem;
  //CreateProblem(desired, area, imageInterpolator, &problem, h, c, au_equi, av_equi, u0_equi, v0_equi);//, au_hsi, av_hsi, u0_hsi, v0_hsi);
  CreateProblem(desired, area, imageInterpolator, &problem, h, affine_photometric_model, au_equi, av_equi, u0_equi, v0_equi);//, au_hsi, av_hsi, u0_hsi, v0_hsi);

  std::cout << "Hinit = " << std::endl << H << std::endl;

	if(FLAGS_directOpt)
  	SolveProblem(&problem, h);

  std::cout << "H = " << std::endl << H << std::endl;
  
  //std::cout << "c = " << std::endl << c[0] << " " << c[1] << " " << c[2] << std::endl;
  std::cout << "affine_photometric_model = " << affine_photometric_model[0] << " " << affine_photometric_model[1] << std::endl;

  ceres::Grid2D<double, 1> arrayDes(desired.ptr<double>(0), 0, desired.rows, 0, desired.cols);
  ceres::BiCubicInterpolator<ceres::Grid2D<double, 1> > desiredInterpolator(arrayDes);

  int index;
  double ud, vd;
  double I_aligned[1750];
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> iH = H.transpose();//inverse();
  double *ih = iH.data();
  
	//int index;
  //H.setIdentity();  
  for (unsigned int v = 0 ; v < aligned.rows ; v++)
  {
  	phi = (v-v0_equi) / av_equi;//elev
  	sphi = sin(phi);
    Zs = sin(phi);//sphi;//cos(phi);  
    for (unsigned int u = 0 ; u < aligned.cols ; u++)
    {
		  //if(area.contains(Point(u,v)))//)(area[0] < ud) && (ud < (area[0]+area[2])) && (area[1] < vd) && (vd < (area[1]+area[3])))
		  {
		  	theta = (u-u0_equi)/au_equi;//az
		  	Ys = sin(theta)*cos(phi);//sphi*sin(theta);
    		Xs = cos(phi)*cos(theta);//sphi*cos(theta);
		  	
		  	/*Zs = cos(phi)*cos(theta);//cos(phid);  
    		Ys = sin(phi);//sphid*sin(thetad);
   			Xs = cos(phi)*sin(theta);//sphid*cos(thetad);
		  	*/
		  	Xsd = (Xs*h[0] + Ys*h[1] + Zs*h[2]);
		    Ysd = (Xs*h[3] + Ys*h[4] + Zs*h[5]);
		    Zsd = (Xs*h[6] + Ys*h[7] + Zs*h[8]); 
		    
		    double rho = sqrt(Xsd*Xsd + Ysd*Ysd + Zsd*Zsd);
		    
		    Xsd /= rho;
		    Ysd /= rho;
		    Zsd /= rho;
		    
		    //thetad = atan2(Xsd, Zsd); //Azimuth entre -pi et pi
		  	//phid = asin(Ysd);//atan2(Ysd, sqrt(Xsd*Xsd + Zsd*Zsd)); //Elevation entre -pi/2 et pi/2 // 
		  	thetad = atan2(Ysd, Xsd); //Azimuth entre -pi et pi
    		phid = asin(Zsd);//
    		//phid = atan2(Zsd, sqrt(Xsd*Xsd + Ysd*Ysd)); //Elevation entre -pi/2 et pi/2
	 
	 
	 /*
		  	ud = au_hsi*thetad + u0_hsi;
		  	vd = av_hsi*phid + v0_hsi;
		 */
		 
		 		ud = au_equi*thetad + u0_equi;
		  	vd = av_equi*phid + v0_equi;	
		  	
		  	//std::cout << ud << " " << vd << std::endl;
		  	
		  	//if((ud > -1) && (ud < shapei[1]) && (vd > -1) && (vd < shapei[0]))
		  	if((ud > -1) && (ud < desired.cols) && (vd > -1) && (vd < desired.rows))
		  	{ 
		  	/*
		    imageInterpolator.Evaluate(vd, ud, I_aligned );
		    double brigthness = I_aligned[119];
		    
		    for(unsigned int iChannel = 120 ; iChannel < 805 ; iChannel++)
		       brigthness += I_aligned[iChannel];
		       
		    aligned.at<double>(v, u) = brigthness * c[0];
		    */
		    if(hsi_cart.at<double>((int)vd, (int)ud) != 0)
				{
		    	//aligned.at<double>(v, u) = affine_photometric_model[0]*(hsi_cart.at<double>((int)vd, (int)ud))+affine_photometric_model[1];
					aligned.at<double>(v, u) = hsi_cart.at<double>((int)vd, (int)ud);
					aligned_hsi_inds.at<int>(v, u) = hsi_inds.at<int>((int)vd, (int)ud);//for mapping export
				}
		    /*if(aligned.at<double>(v, u) > 255)
		    	aligned.at<double>(v, u) = 255;
		    	*/
		    }
		  }
      
    }
  }
  
 /*
  double minVal; 
double maxVal; 
Point minLoc; 
Point maxLoc;

	cv::minMaxLoc( aligned, &minVal, &maxVal, &minLoc, &maxLoc );
 	double alpha8 = 255.0/maxVal;
 	
 	std::cout << "min val: " << minVal << std::endl;
	std::cout << "max val: " << maxVal << std::endl;

  
  aligned.convertTo(aligned, CV_8UC1, alpha8, 0);
  */
  
  aligned.convertTo(aligned, CV_8UC1);
  
  /*
  int slice = 500;
  double I_aligned[1750];
  for (unsigned int v = 0 ; v < shapei[0] ; v++)
  {

    for (unsigned int u = 0 ; u < shapei[1] ; u++)
    {
    	
    	
      imageInterpolator.Evaluate(v, u, I_aligned );
      

      //arrayDes.GetValue(vd, ud, &I_aligned );
      aligned.at<double>(v, u) = I_aligned[slice] / (25.);//I_aligned;
      //aligned.MutablePixelFromLinearIndex(aligned.LinearIndex(u,v))[0] = I_aligned; //desired.PixelFromLinearIndex(desired.LinearIndex(ud,vd));//
    }
  }
  
  aligned.convertTo(aligned, CV_8UC1);
  imshow("Image", aligned);
  
  waitKey(0);
	*/
	
  if (!FLAGS_output.empty()) {
  	if(FLAGS_displayUsedArea)
  		cv::rectangle(aligned,area,Scalar(255,0,0),1,8,0);
  		
    CHECK(imwrite(FLAGS_output, aligned))
        << "Writing \"" << FLAGS_output << "\" failed.";
  }
  
  if (!FLAGS_outputMappingOpenCVTxt.empty()) {
    // Declare what you need
		cv::FileStorage file(FLAGS_outputMappingOpenCVTxt, cv::FileStorage::WRITE);
		
		// Write to file!
		file << "hsiIndicesInThetaFrame" << aligned_hsi_inds;
  } 
  
  if (!FLAGS_outputMappingNpy.empty()) {
		std::vector<int> aligned_hsi_inds_vec(aligned_hsi_inds.rows*aligned_hsi_inds.cols*aligned_hsi_inds.channels());
    aligned_hsi_inds_vec.assign(aligned_hsi_inds.begin<int>(), aligned_hsi_inds.end<int>());
  	std::array<long unsigned, 2> aligned_hsi_inds_shape {{(long unsigned)aligned_hsi_inds.rows, (long unsigned)aligned_hsi_inds.cols}};
  	
		npy::SaveArrayAsNumpy(FLAGS_outputMappingNpy, false, aligned_hsi_inds_shape.size(), aligned_hsi_inds_shape.data(), aligned_hsi_inds_vec);
		
		std::vector<int> data;
		npy::LoadArrayFromNumpy(FLAGS_outputMappingNpy, shape, fortran_order, data);
		std::cout << "npy data storage check: " << data[0] << std::endl;
  } 
  
  delete [] c;
  delete [] w;
  delete [] u_clicked;
  delete [] v_clicked;
  delete [] u_hsi;
  delete [] v_hsi;

  return 0;
}

/*
// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 0.5;
  const double initial_x = x;

  // Build the problem.
  Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, &x);

  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
*/
