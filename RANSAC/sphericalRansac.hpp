#ifndef SphericalHomographyEstimatorCallback_H_
#define SphericalHomographyEstimatorCallback_H_

//#include <opencv2/calib3d.hpp>
#define __OPENCV_BUILD
#include "precomp.hpp"
#include "rho.h"
#undef __OPENCV_BUILD
//#include "/home/guillaume/Develop/libraries/opencv/modules/calib3d/src/precomp.hpp"

using namespace cv;

/**
 * This class estimates a homography \f$H\in \mathbb{R}^{3\times 3}\f$
 * between spherical \f$\mathbf{X}_S \in \mathbb{R}^3\f$ and
 * \f$\mathbf{X}_S* \in \mathbb{R}^3\f$ using DLT (direct linear transform)
 * with algebraic distance.
 *
 * \f[
 *   \mathbf{X}_S* x H \mathbf{X}_S = 0
 * \f]
 *
 */
class SphericalHomographyEstimatorCallback CV_FINAL : public PointSetRegistrator::Callback //HomographyEstimatorCallback //
{
public:
    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const CV_OVERRIDE
    {
    	/*
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        if( haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count) ) //may not need to be updated?
            return false;

				//Check if below valid for sphere

        // We check whether the minimal set of points for the homography estimation
        // are geometrically consistent. We check if every 3 correspondences sets
        // fulfills the constraint.
        //
        // The usefulness of this constraint is explained in the paper:
        //
        // "Speeding-up homography estimation in mobile devices"
        // Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
        // Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
        if( count == 4 )
        {
            static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
            const Point2f* src = ms1.ptr<Point2f>();
            const Point2f* dst = ms2.ptr<Point2f>();
            int negative = 0;

            for( int i = 0; i < 4; i++ )
            {
                const int* t = tt[i];
                Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
                Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);

                negative += determinant(A)*determinant(B) < 0;
            }
            if( negative != 0 && negative != 4 )
                return false;
        }
        */

        return true;
    }

    /**
     * Normalization method for 2D plane(TO DO ON SPHERE):
     *  - $x$ and $y$ coordinates are normalized independently
     *  - first the coordinates are shifted so that the average coordinate is \f$(0,0)\f$
     *  - then the coordinates are scaled so that the average L1 norm is 1, i.e,
     *  the average L1 norm of the \f$x\f$ coordinates is 1 and the average
     *  L1 norm of the \f$y\f$ coordinat es is also 1.
     *
     * @param _m1 source points containing (X_S,Y_S,Z_S), depth is CV_32F with 1 column 3 channels or
     *            3 columns 1 channel
     * @param _m2 destination points containing (X_S*,Y_S*,Z_S*), depth is CV_32F with 1 column 3 channels or
     *            3 columns 1 channel
     * @param _model, CV_64FC1, 3x3 
     */
    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const CV_OVERRIDE //_m1: M
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(3);
        const Point3f* M = m1.ptr<Point3f>();
        const Point3f* m = m2.ptr<Point3f>();

				/*
        double LtL[9][9], W[9][1], V[9][9];
        Mat _LtL( 9, 9, CV_64F, &LtL[0][0] );
        Mat matW( 9, 1, CV_64F, W );
        Mat matV( 9, 9, CV_64F, V );
        Mat _H0( 3, 3, CV_64F, V[8] );
        Mat _Htemp( 3, 3, CV_64F, V[7] );
        Point2d cM(0,0), cm(0,0), sM(0,0), sm(0,0);

				//centroid computation
        for( i = 0; i < count; i++ )
        {
            cm.x += m[i].x; cm.y += m[i].y;
            cM.x += M[i].x; cM.y += M[i].y;
        }

        cm.x /= count;
        cm.y /= count;
        cM.x /= count;
        cM.y /= count;

				//center
        for( i = 0; i < count; i++ )
        {
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }

				//scale
        if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
            return 0;
        sm.x = count/sm.x; sm.y = count/sm.y;
        sM.x = count/sM.x; sM.y = count/sM.y;

				//centering and scaling transformations
        double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
        double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
        Mat _invHnorm( 3, 3, CV_64FC1, invHnorm );
        Mat _Hnorm2( 3, 3, CV_64FC1, Hnorm2 );
        
        //solve for the homography
        _LtL.setTo(Scalar::all(0));
        */
        
        
  
  			Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor> A;
  			std::cout << "OK -2" << std::endl;
  			A = Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>::Zero(3*count,9);
  			std::cout << "OK -1" << std::endl;
    
				for(unsigned int i = 0 ; i < count ; i++)
				{				 	
				std::cout << "OK " << i << std::endl;
				 	A(3*i,3) = -M[i].x*m[i].z; A(3*i,4) = -M[i].y*m[i].z; A(3*i,5) = -M[i].z*m[i].z; A(3*i,6) = M[i].x*m[i].y; A(3*i,7) = M[i].y*m[i].y; A(3*i,8) = M[i].z*m[i].y; 
				 	A(3*i+1,0) = M[i].x*m[i].z; A(3*i+1,1) = M[i].y*m[i].z; A(3*i+1,2) = M[i].z*m[i].z; A(3*i+1,6) = -M[i].x*m[i].x; A(3*i+1,7) = -M[i].y*m[i].x; A(3*i+1,8) = -M[i].z*m[i].x; 
				 	A(3*i+2,0) = -M[i].x*m[i].y; A(3*i+2,1) = -M[i].y*m[i].y; A(3*i+2,2) = -M[i].z*m[i].y; A(3*i+2,3) = M[i].x*m[i].x; A(3*i+2,4) = M[i].y*m[i].x; A(3*i+2,5) = M[i].z*m[i].x;
				}
  
  			std::cout << "A: " << A << std::endl;
  
  			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  			Eigen::MatrixXd V=svd.matrixV();
  
  			std::cout << "V: " << V << std::endl;
  
  
  			Mat _H0( 3, 3, CV_64F );	
  			double *h = _H0.ptr<double>(0);
  			for(unsigned int i = 0; i<9 ; i++)
  				h[i] = V(i,8);
  	
  			
  			std::cout << "copy OK" << std::endl;
  	
  			_H0.convertTo(_model, _H0.type());
  	
        std::cout << "convert OK" << std::endl;
        
        /*
        for( i = 0; i < count; i++ )
        {
            double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
            double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
            double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
            double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
            int j, k;
            for( j = 0; j < 9; j++ )
                for( k = j; k < 9; k++ )
                    LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
        }
        completeSymm( _LtL );

        eigen( _LtL, matW, matV );
        _Htemp = _invHnorm*_H0;
        _H0 = _Htemp*_Hnorm2;
        
        _H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(2,2) );
				*/

        return 1;
    }

    /**
     * Compute the reprojection error.
     * m2 = normalize(H*m1)
     * @param _m1 depth CV_32F, 1-channel with 3 columns or 3-channel with 1 column
     * @param _m2 depth CV_32F, 1-channel with 3 columns or 3-channel with 1 column
     * @param _model CV_64FC1, 3x3
     * @param _err, output, CV_32FC1, square of the L2 norm
     */
    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const CV_OVERRIDE
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        int i, count = m1.checkVector(3);
        const Point3f* M = m1.ptr<Point3f>();
        const Point3f* m = m2.ptr<Point3f>();
        const double* H = model.ptr<double>();
        float Hf[] = { (float)H[0], (float)H[1], (float)H[2], (float)H[3], (float)H[4], (float)H[5], (float)H[6], (float)H[7], (float)H[8] };

        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();
        
  			for(unsigned int i = 0 ; i < count ; i++)
  			{
  				float Xsh = Hf[0]*M[i].x + Hf[1]*M[i].y + Hf[2]*M[i].z;
  				float Ysh = Hf[3]*M[i].x + Hf[4]*M[i].y + Hf[5]*M[i].z;
  				float Zsh = Hf[6]*M[i].x + Hf[7]*M[i].y + Hf[8]*M[i].z;
  				
  			
  				float inv_rho = 1.0f/sqrt(Xsh*Xsh + Ysh*Ysh + Zsh*Zsh);  				
  					
  				float dXs = Xsh*inv_rho - m[i].x;
          float dYs = Ysh*inv_rho - m[i].y;
          float dZs = Zsh*inv_rho - m[i].z;
          err[i] = dXs*dXs + dYs*dYs + dZs*dZs;
  			}
  
				/*
        for( i = 0; i < count; i++ )
        {
            float ww = 1.f/(Hf[6]*M[i].x + Hf[7]*M[i].y + 1.f);
            float dx = (Hf[0]*M[i].x + Hf[1]*M[i].y + Hf[2])*ww - m[i].x;
            float dy = (Hf[3]*M[i].x + Hf[4]*M[i].y + Hf[5])*ww - m[i].y;
            err[i] = dx*dx + dy*dy;
        }
        */
    }
};



cv::Mat findSphericalHomography( InputArray _points1, InputArray _points2,
                            int method, double ransacReprojThreshold, OutputArray _mask, 
                            const int maxIters, const double confidence)
{
    CV_INSTRUMENT_REGION();

    const double defaultRANSACReprojThreshold = 3;
    bool result = false;

    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat src, dst, H, tempMask;
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        Mat& p = i == 1 ? points1 : points2;
        Mat& m = i == 1 ? src : dst;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            if( npoints < 0 )
                CV_Error(Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if( npoints == 0 )
                return Mat();
            //convertPointsFromHomogeneous(p, p); //not for spherical homography
        }
        p.reshape(3, npoints).convertTo(m, CV_32F); //3 for spherical homography instead of 2 
    }

    CV_Assert( src.checkVector(2) == dst.checkVector(2) );

    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    Ptr<PointSetRegistrator::Callback> cb = makePtr<SphericalHomographyEstimatorCallback>();

    if( method == 0 || npoints == 4 )
    {
        tempMask = Mat::ones(npoints, 1, CV_8U);
        result = cb->runKernel(src, dst, H) > 0;
    }
    else if( method == RANSAC )
        result = createRANSACPointSetRegistrator(cb, 4, ransacReprojThreshold, confidence, maxIters)->run(src, dst, H, tempMask);
    else if( method == LMEDS )
        result = createLMeDSPointSetRegistrator(cb, 4, confidence, maxIters)->run(src, dst, H, tempMask);
    /*else if( method == RHO )
        result = createAndRunRHORegistrator(confidence, maxIters, ransacReprojThreshold, npoints, src, dst, H, tempMask);
        */
    else
        CV_Error(Error::StsBadArg, "Unknown estimation method");

		/*
    //TO DO: later
    if( result && npoints > 4 && method != RHO)
    {
        compressElems( src.ptr<Point3f>(), tempMask.ptr<uchar>(), 1, npoints );
        npoints = compressElems( dst.ptr<Point3f>(), tempMask.ptr<uchar>(), 1, npoints );
        if( npoints > 0 )
        {
            Mat src1 = src.rowRange(0, npoints);
            Mat dst1 = dst.rowRange(0, npoints);
            src = src1;
            dst = dst1;
            if( method == RANSAC || method == LMEDS )
                cb->runKernel( src, dst, H );
            Mat H8(8, 1, CV_64F, H.ptr<double>());
            LMSolver::create(makePtr<SphericalHomographyEstimatorCallback>(src, dst), 10)->run(H8);
        }
    }
    */

    if( result )
    {
        if( _mask.needed() )
            tempMask.copyTo(_mask);
    }
    else
    {
        H.release();
        if(_mask.needed() ) {
            tempMask = Mat::zeros(npoints >= 0 ? npoints : 0, 1, CV_8U);
            tempMask.copyTo(_mask);
        }
    }

    return H;
}

#endif //SphericalHomographyEstimatorCallback
