#ifndef UKF_HPP
#define UKF_HPP

#include <Eigen/Dense>

class UkfBase
{

};

class Ukf
{
public:
   Ukf(const long n_x);
   Ukf(const long n_x, const long n_x_aug);
   virtual ~Ukf();

   void generateSigmaPoints(const Eigen::VectorXd &x, const Eigen::MatrixXd & P, Eigen::MatrixXd & XSigmaPoints);
   void generateAugumentedSigmaPoints(const Eigen::VectorXd &x, const Eigen::MatrixXd & P, const Eigen::VectorXd &std, Eigen::MatrixXd & XSigmaPoints);

   template<typename MODEL_T>
   inline void predict(const Eigen::MatrixXd & XSigmaPoints, Eigen::VectorXd &x_pred, Eigen::MatrixXd & P_pred)
   {
      weightedMean(XSigmaPoints, x_pred);
      predictCovar<MODEL_T>(XSigmaPoints, x_pred, P_pred);
   }


   void weightedMean(const Eigen::MatrixXd & XSigmaPoints, Eigen::VectorXd &x_pred);

   void calculateCrosscorrelationMatix(
         const Eigen::MatrixXd & XSigmaPointsPred,
         const Eigen::MatrixXd & ZSigmaPointsPred,
         const Eigen::VectorXd & x_mean,
         const Eigen::VectorXd & z_mean,
         Eigen::MatrixXd & T);

   void calculateKalmanGain(const Eigen::MatrixXd & T, const Eigen::MatrixXd & S, Eigen::MatrixXd & K);
   void updateState(const Eigen::MatrixXd & K, const Eigen::VectorXd & z, const Eigen::VectorXd & z_pred, Eigen::VectorXd & x_pred);
   void updateCovariance(const Eigen::MatrixXd & K, const Eigen::MatrixXd & S, Eigen::MatrixXd & P_pred);

   long getNx() const {return m_n_x;}
   long getNxAug() const {return m_n_x_aug;}
   long getNSigmaPoints() const {return m_nSigmaPoints; }

protected:
   template<typename MODEL_T>
   void predictCovar(const Eigen::MatrixXd & SigmaPoints, const Eigen::VectorXd & vector, Eigen::MatrixXd & Covariance)
   {
      Eigen::MatrixXd SigmaPointsDiff = SigmaPoints.colwise() - vector;

      for (long colIdx = 0; colIdx < m_nSigmaPoints; ++colIdx)
      {
         // TODO move down
         MODEL_T::normalize(colIdx, SigmaPointsDiff);

         Eigen::MatrixXd cov_pred_sigmaPoint = (SigmaPointsDiff.col(colIdx).array() * m_weights(colIdx)).matrix() *
               SigmaPointsDiff.col(colIdx).transpose();

         Covariance += cov_pred_sigmaPoint;
      }
   }

   //set state dimension
   const long m_n_x;

   //set augumented state dimension
   const long m_n_x_aug;

   //define spreading parameter
   const double m_lambda;

   const long m_nSigmaPoints;

   // weights for prediction
   const Eigen::ArrayXd m_weights;

   static constexpr double TWO_PI = 2 * M_PI;

private:
   const Eigen::ArrayXd calculateWeights();

};

#endif // UKF_HPP
