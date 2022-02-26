#ifndef multiSresidue_multiChannel_H_
#define multiSresidue_multiChannel_H_

#include <ceres/cubic_interpolation.h>

struct multiSresidue
{
    /// Set the constants of the problem
    multiSresidue(const double & au, const double & av,  const double & u0, const double & v0,
    						const double & Xsd, const double & Ysd,  const double & Zsd,
//               const double & desired, const ceres::BiCubicInterpolator<ceres::Grid2D<double,1750> > & image) : au(au), av(av), u0(u0), v0(v0), Xsd(Xsd), Ysd(Ysd), Zsd(Zsd), image(image), I_observed(desired)
               const double & desired, const ceres::BiCubicInterpolator<ceres::Grid2D<double,1> > & image) : au(au), av(av), u0(u0), v0(v0), Xsd(Xsd), Ysd(Ysd), Zsd(Zsd), image(image), I_observed(desired)
    {

    }

    template <typename T>
    bool operator()(/*const T* const Q, */const T* const h, const T* const c, T* residual) const
   // bool operator()(/*const T* const Q, */const T* const h, T* residual) const
    {
      T I_predicted[1750];
      
      T Xs = (Xsd*h[0] + Ysd*h[1] + Zsd*h[2]);
      T Ys = (Xsd*h[3] + Ysd*h[4] + Zsd*h[5]);
      T Zs = (Xsd*h[6] + Ysd*h[7] + Zsd*h[8]); // *T(1.0)
      
      //std::cout << Xs << std::endl;
      
      T rho = sqrt(Xs*Xs + Ys*Ys + Zs*Zs);
      Xs /= rho;
      Ys /= rho;
      Zs /= rho;
      
      //T theta = atan2(Ys,Xs); //Azimuth entre -pi et pi
    	//T phi = atan2(sqrt(Xs*Xs + Ys*Ys), Zs); //Elevation entre -pi/2 et pi/2
    	T theta = atan2(Ys, Xs); //Azimuth entre -pi et pi
    	T phi = asin(Zs);
 
    	
    	//std::cout << phi << std::endl;
    	
    	T u = au*theta + u0;
    	T v = av*phi + v0;
    	
    	//std::cout << v << std::endl;
      
      image.Evaluate(v, u, I_predicted ); // row, col, interpolated
      
      T I_b_pred = I_predicted[0];/*[119];
      
      for(unsigned int iChannel = 120 ; iChannel < 805 ; iChannel++)
         I_b_pred += I_predicted[iChannel];			
			*/
			residual[0] = c[0]*(I_b_pred)+c[1] - I_observed;
			
      return true;
    }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const double & au, const double & av,  const double & u0, const double & v0,
   							const double & Xsd, const double & Ysd,  const double & Zsd,
//               const double & desired, const ceres::BiCubicInterpolator<ceres::Grid2D<double,1750> > & image) {
               const double & desired, const ceres::BiCubicInterpolator<ceres::Grid2D<double,1> > & image) {
//     return (new ceres::AutoDiffCostFunction<multiSresidue, 1, 9, 1>(
     return (new ceres::AutoDiffCostFunction<multiSresidue, 1, 9, 2>(
                 new multiSresidue(au, av, u0, v0, Xsd, Ysd, Zsd, desired, image)));
   }

		double au, av, u0, v0;
    double Xsd, Ysd, Zsd;
    double I_observed;

//    const ceres::BiCubicInterpolator< ceres::Grid2D<double,1750> > & image;
    const ceres::BiCubicInterpolator< ceres::Grid2D<double,1> > & image;
};

#endif  // multiSresidue_multiChannel_H_
