#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

// ES
double corotationalCurveHingeBendingEnergy(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& iniPos,
    const Eigen::MatrixXd& curPos,
    double YoungsModulus, double PoissonsRatio, double thickness,
    const std::vector<Eigen::Matrix2d>& abars,
    const SecondFundamentalFormDiscretization& sff,
    Eigen::VectorXd* derivative, 
    std::vector<Eigen::Triplet<double> >* hessian,
    bool isLocalProj,
    bool isParallel)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = curPos.rows();
    int nedgedofs = sff.numExtraDOFs();
    int dofs_ = 3 * nverts + nedgedofs * nedges;
    Eigen::VectorXd disp(3 * nverts); 
    for (int i = 0; i < nverts; i++)
        disp.segment<3>(3 * i) = curPos.row(i) - iniPos.row(i);

    if (derivative)
    {
        derivative->resize(dofs_);
        derivative->setZero();
    }
    if (hessian)
    {
        hessian->clear();
    }

    double bendingEnergy = 0.0;
    std::vector<Eigen::Matrix<double, 3, 1>> eleOrderInterFs(nedges * 4, Eigen::Matrix<double, 3, 1>::Zero());
    Eigen::VectorXd globalBendingForces = Eigen::VectorXd::Zero(dofs_);
    std::vector<Eigen::Triplet<double>> bendingStiffK;
    bendingStiffK.clear();
    bendingStiffK.resize(nedges * 12 * 12); 
    const double bendingRigidity = std::pow(thickness, 3) / 12.0 * YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
    if (false) // parallel version should be improved for stability for this model
    {
        auto eleVInds = std::vector<Eigen::Vector4i>(nedges);
        auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i) 
            {
                int adjFace0 = mesh.edgeFace(i, 0);
                int adjFace1 = mesh.edgeFace(i, 1);
                Eigen::Vector4i elemVInd;
                elemVInd(0) = mesh.edgeVertex(i, 0);
                elemVInd(1) = mesh.edgeVertex(i, 1);
                elemVInd(2) = mesh.edgeOppositeVertex(i, 0);
                elemVInd(3) = mesh.edgeOppositeVertex(i, 1);
                eleVInds[i] = elemVInd;
                if (adjFace0!=-1 && adjFace1!=-1) 
                {
                    Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
                    Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
                    Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
                    Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
                    Eigen::Matrix<double, 3, 4> elePatchRestPos;
                    elePatchRestPos << X1, X2, X3, X4;
                    Eigen::Matrix<double, 12, 1> X_e = Eigen::Matrix<double, 12, 1>::Zero();
                    X_e.block<3, 1>(0, 0) = X1; X_e.block<3, 1>(3, 0) = X2; X_e.block<3, 1>(6, 0) = X3; X_e.block<3, 1>(9, 0) = X4;
                    Eigen::Matrix<double, 3, 1> XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                    Eigen::Matrix<double, 3, 1> XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                    const Eigen::Matrix<double, 3, 1> p0 = X3 - XP, q0 = X4 - XQ;
                    Eigen::Matrix<double, 3, 1> n0 = (p0/p0.norm()+q0/q0.norm()).normalized();
                    if (n0.norm() == 0)// initial flat
                        n0 = ((X2 - X1).cross(X3 - X1).normalized()+(X4 - X1).cross(X2 - X1).normalized())/2;
                    const Eigen::Matrix<double, 3, 1> local_x_axis = (X2 - X1).normalized();
                    const Eigen::Matrix<double, 3, 1> local_y_axis = n0.cross(local_x_axis);
                    Eigen::Matrix<double, 3, 3> local_exyzT;
                    local_exyzT << local_x_axis.transpose(),
                                   local_y_axis.transpose(),
                                   n0.transpose();
                    Eigen::Matrix<double, 3, 4> matX1;
                    matX1 << X1, X1, X1, X1;
                    const Eigen::Matrix<double, 3, 4> localXYZ = local_exyzT * (elePatchRestPos - matX1);
                    X1 = localXYZ.col(0); X2 = localXYZ.col(1); X3 = localXYZ.col(2); X4 = localXYZ.col(3);
                    XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                    XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                    const double restLenEdge12 = (X2 - X1).norm();
                    const double restHeight1 = (X3 - XP).norm();
                    const double restHeight2 = (X4 - XQ).norm();
                    const double P2 = (X2 - XP).norm();
                    const double P1 = restLenEdge12 - P2;
                    const double Q2 = (X2 - XQ).norm();
                    const double Q1 = restLenEdge12 - Q2;
                    const double L1 = - (P2/restHeight1 + Q2/restHeight2)/restLenEdge12;
                    const double L2 = - (P1/restHeight1 + Q1/restHeight2)/restLenEdge12;
                    const double L3 = 1/restHeight1;
                    const double L4 = 1/restHeight2;
                    Eigen::Matrix<double, 3, 12> L = Eigen::Matrix<double, 3, 12>::Zero();
                    L.block<3, 3>(0, 0) = L1 * Eigen::Matrix3d::Identity();
                    L.block<3, 3>(0, 3) = L2 * Eigen::Matrix3d::Identity();
                    L.block<3, 3>(0, 6) = L3 * Eigen::Matrix3d::Identity();
                    L.block<3, 3>(0, 9) = L4 * Eigen::Matrix3d::Identity();
                    const double restStencilArea = 0.5 * restLenEdge12 * (restHeight1 + restHeight2); 
                    const double theta0 = n0.transpose() * L * X_e;
                    
                    const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
                    const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
                    const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
                    const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
                    Eigen::Matrix<double, 12, 1> x_e = Eigen::Matrix<double, 12, 1>::Zero();
                    x_e.block<3, 1>(0, 0) = x1; x_e.block<3, 1>(3, 0) = x2; x_e.block<3, 1>(6, 0) = x3; x_e.block<3, 1>(9, 0) = x4;
                    const Eigen::Matrix<double, 3, 1> e = (x2 - x1).normalized();
                    const double curLenEdge12 = (x2 - x1).norm();
                    const Eigen::Matrix<double, 3, 1> xp = (x2 - x1) * (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
                    const Eigen::Matrix<double, 3, 1> xq = (x2 - x1) * (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
                    const Eigen::Matrix<double, 3, 1> p = x3 - xp, q = x4 - xq;
                    const double s_p = (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm(), s_q = (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm();
                    Eigen::Matrix<double, 3, 1> n = (p/p.norm()+q/q.norm()).normalized();
                    if (p.cross(q).norm() == 0)
                        n = (x2 - x1).cross(x3 - x1).normalized();
                    const double theta = n.transpose() * L * x_e;
                    const Eigen::Matrix<double, 3, 3> dpTdx1 = -(1-s_p)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + p*e.transpose()/curLenEdge12;
                    const Eigen::Matrix<double, 3, 3> dpTdx2 = -s_p*(Eigen::Matrix3d::Identity()-e*e.transpose()) - p*e.transpose()/curLenEdge12;
                    const Eigen::Matrix<double, 3, 3> dpTdx3 = Eigen::Matrix3d::Identity() - e*e.transpose();
                    const Eigen::Matrix<double, 3, 3> dpTdx4 = Eigen::Matrix3d::Zero();
                    const Eigen::Matrix<double, 3, 3> dqTdx1 = -(1-s_q)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + q*e.transpose()/curLenEdge12;
                    const Eigen::Matrix<double, 3, 3> dqTdx2 = -s_q*(Eigen::Matrix3d::Identity()-e*e.transpose()) - q*e.transpose()/curLenEdge12;
                    const Eigen::Matrix<double, 3, 3> dqTdx3 = Eigen::Matrix3d::Zero();
                    const Eigen::Matrix<double, 3, 3> dqTdx4 = Eigen::Matrix3d::Identity() - e*e.transpose();
                    Eigen::Matrix<double, 3, 12> grad_p = Eigen::Matrix<double, 3, 12>::Zero();
                    grad_p.block<3, 3>(0, 0) = dpTdx1.transpose(); grad_p.block<3, 3>(0, 3) = dpTdx2.transpose(); grad_p.block<3, 3>(0, 6) = dpTdx3.transpose(); grad_p.block<3, 3>(0, 9) = dpTdx4.transpose();
                    Eigen::Matrix<double, 3, 12> grad_q = Eigen::Matrix<double, 3, 12>::Zero();
                    grad_q.block<3, 3>(0, 0) = dqTdx1.transpose(); grad_q.block<3, 3>(0, 3) = dqTdx2.transpose(); grad_q.block<3, 3>(0, 6) = dqTdx3.transpose(); grad_q.block<3, 3>(0, 9) = dqTdx4.transpose();
                    Eigen::Matrix<double, 3, 12> grad_normalized_p = (Eigen::Matrix3d::Identity() - p.normalized()*p.normalized().transpose()) * grad_p / p.norm();
                    Eigen::Matrix<double, 3, 12> grad_normalized_q = (Eigen::Matrix3d::Identity() - q.normalized()*q.normalized().transpose()) * grad_q / q.norm();
                    Eigen::Matrix<double, 3, 12> grad_n = (Eigen::Matrix3d::Identity() - n*n.transpose()) * (grad_normalized_p + grad_normalized_q) / (p/p.norm()+q/q.norm()).norm();
                    if (p.cross(q).norm() == 0)
                    if ((x2 - x1).cross(x3 - x1).dot(x4 - x1) == 0)
                        grad_n = (Eigen::Matrix3d::Identity() - n*n.transpose()) * (grad_normalized_p + grad_normalized_q) / ((x2 - x1).cross(x3 - x1)).norm();                        

                    Eigen::Matrix<double, 1, 12> grad_theta = x_e.transpose()*L.transpose()*grad_n + n.transpose()*L;
                    Eigen::Matrix<double, 1, 12> grad_theta_cubic = x_e.transpose()*L.transpose()*grad_n;
                    Eigen::Matrix<double, 1, 12> grad_theta_qudratic = n.transpose()*L;
                    // curvature change
                    const double curvature = 2 * (theta - theta0) / (restHeight1 + restHeight2);
                    // bending moment
                    const double moment = bendingRigidity * curvature;
                    // bending energy
                    bendingEnergy += 0.5 * restStencilArea * curvature * moment;

                    // gradient of curvature
                    Eigen::Matrix<double, 12, 1> pCurvaturepUi = 2 * grad_theta.transpose() / (restHeight1 + restHeight2);
                    // gradient of bending moment
                    Eigen::Matrix<double, 12, 1> pMomentpUi = bendingRigidity * pCurvaturepUi;
                    
                    // Hessian of bending energy
                    Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
                    // if (isLocalProj) 
                    //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);

                    // gradient of bending energy
                    Eigen::Matrix<double, 12, 1> eleBendingForce = restStencilArea * curvature * pMomentpUi; 
                    
                    for (int k = 0; k < 4; ++k)
                        globalBendingForces.segment<3>(3 * elemVInd[k]) += eleBendingForce.segment<3>(3 * k);
                    const int offset = i * 4 * 3 * 4 * 3;
                    for (int k = 0; k < 4; ++k)
                        for (int d = 0; d < 3; ++d)
                            for (int s = 0; s < 4; ++s)
                                for (int t = 0; t < 3; ++t){
                                    bendingStiffK[ offset + 
                                    + k * 3 * 4 * 3
                                    + d * 4 * 3
                                    + s * 3
                                    + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, eleBendingStiffMatrix(s * 3 + t, k * 3 + d)); 
                                }
                }
            } 
        }; 
        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nedges);
        tbb::parallel_for(rangex, computeBending);
    }
    else // serial version
    {
        auto eleVInds = std::vector<Eigen::Vector4i>(nedges);
        for (int i = 0; i < nedges; i++) 
        {
            int adjFace0 = mesh.edgeFace(i, 0);
            int adjFace1 = mesh.edgeFace(i, 1);
            Eigen::Vector4i elemVInd;
            elemVInd(0) = mesh.edgeVertex(i, 0);
            elemVInd(1) = mesh.edgeVertex(i, 1);
            elemVInd(2) = mesh.edgeOppositeVertex(i, 0);
            elemVInd(3) = mesh.edgeOppositeVertex(i, 1);
            eleVInds[i] = elemVInd;
            if (adjFace0!=-1 && adjFace1!=-1) 
            {
                Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
                Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
                Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
                Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
                Eigen::Matrix<double, 3, 4> elePatchRestPos;
                elePatchRestPos << X1, X2, X3, X4;
                Eigen::Matrix<double, 12, 1> X_e = Eigen::Matrix<double, 12, 1>::Zero();
                X_e.block<3, 1>(0, 0) = X1; X_e.block<3, 1>(3, 0) = X2; X_e.block<3, 1>(6, 0) = X3; X_e.block<3, 1>(9, 0) = X4;
                Eigen::Matrix<double, 3, 1> XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                Eigen::Matrix<double, 3, 1> XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                const Eigen::Matrix<double, 3, 1> p0 = X3 - XP, q0 = X4 - XQ;
                Eigen::Matrix<double, 3, 1> n0 = (p0/p0.norm()+q0/q0.norm()).normalized();
                if (n0.norm() == 0)// initial flat
                    n0 = ((X2 - X1).cross(X3 - X1).normalized()+(X4 - X1).cross(X2 - X1).normalized())/2;
                const Eigen::Matrix<double, 3, 1> local_x_axis = (X2 - X1).normalized();
                const Eigen::Matrix<double, 3, 1> local_y_axis = n0.cross(local_x_axis);
                Eigen::Matrix<double, 3, 3> local_exyzT;
                local_exyzT << local_x_axis.transpose(),
                                local_y_axis.transpose(),
                                n0.transpose();
                Eigen::Matrix<double, 3, 4> matX1;
                matX1 << X1, X1, X1, X1;
                const Eigen::Matrix<double, 3, 4> localXYZ = local_exyzT * (elePatchRestPos - matX1);
                X1 = localXYZ.col(0); X2 = localXYZ.col(1); X3 = localXYZ.col(2); X4 = localXYZ.col(3);
                XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                const double restLenEdge12 = (X2 - X1).norm();
                const double restHeight1 = (X3 - XP).norm();
                const double restHeight2 = (X4 - XQ).norm();
                const double P2 = (X2 - XP).norm();
                const double P1 = restLenEdge12 - P2;
                const double Q2 = (X2 - XQ).norm();
                const double Q1 = restLenEdge12 - Q2;
                const double L1 = - (P2/restHeight1 + Q2/restHeight2)/restLenEdge12;
                const double L2 = - (P1/restHeight1 + Q1/restHeight2)/restLenEdge12;
                const double L3 = 1/restHeight1;
                const double L4 = 1/restHeight2;
                Eigen::Matrix<double, 3, 12> L = Eigen::Matrix<double, 3, 12>::Zero();
                L.block<3, 3>(0, 0) = L1 * Eigen::Matrix3d::Identity();
                L.block<3, 3>(0, 3) = L2 * Eigen::Matrix3d::Identity();
                L.block<3, 3>(0, 6) = L3 * Eigen::Matrix3d::Identity();
                L.block<3, 3>(0, 9) = L4 * Eigen::Matrix3d::Identity();
                const double restStencilArea = 0.5 * restLenEdge12 * (restHeight1 + restHeight2); 
                const double theta0 = n0.transpose() * L * X_e;
                
                const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
                const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
                const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
                const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
                Eigen::Matrix<double, 12, 1> x_e = Eigen::Matrix<double, 12, 1>::Zero();
                x_e.block<3, 1>(0, 0) = x1; x_e.block<3, 1>(3, 0) = x2; x_e.block<3, 1>(6, 0) = x3; x_e.block<3, 1>(9, 0) = x4;
                const Eigen::Matrix<double, 3, 1> e = (x2 - x1).normalized();
                const double curLenEdge12 = (x2 - x1).norm();
                const Eigen::Matrix<double, 3, 1> xp = (x2 - x1) * (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
                const Eigen::Matrix<double, 3, 1> xq = (x2 - x1) * (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
                const Eigen::Matrix<double, 3, 1> p = x3 - xp, q = x4 - xq;
                const double s_p = (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm(), s_q = (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm();
                Eigen::Matrix<double, 3, 1> n = (p/p.norm()+q/q.norm()).normalized();
                if (p.cross(q).norm() == 0)// p and q are parallel
                    n = (x2 - x1).cross(x3 - x1).normalized();
                const double theta = n.transpose() * L * x_e;
                // compute the gradient of theta (fold angle)
                const Eigen::Matrix<double, 3, 3> dpTdx1 = -(1-s_p)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + p*e.transpose()/curLenEdge12;
                const Eigen::Matrix<double, 3, 3> dpTdx2 = -s_p*(Eigen::Matrix3d::Identity()-e*e.transpose()) - p*e.transpose()/curLenEdge12;
                const Eigen::Matrix<double, 3, 3> dpTdx3 = Eigen::Matrix3d::Identity() - e*e.transpose();
                const Eigen::Matrix<double, 3, 3> dpTdx4 = Eigen::Matrix3d::Zero();
                const Eigen::Matrix<double, 3, 3> dqTdx1 = -(1-s_q)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + q*e.transpose()/curLenEdge12;
                const Eigen::Matrix<double, 3, 3> dqTdx2 = -s_q*(Eigen::Matrix3d::Identity()-e*e.transpose()) - q*e.transpose()/curLenEdge12;
                const Eigen::Matrix<double, 3, 3> dqTdx3 = Eigen::Matrix3d::Zero();
                const Eigen::Matrix<double, 3, 3> dqTdx4 = Eigen::Matrix3d::Identity() - e*e.transpose();
                Eigen::Matrix<double, 3, 12> grad_p = Eigen::Matrix<double, 3, 12>::Zero();
                grad_p.block<3, 3>(0, 0) = dpTdx1.transpose(); grad_p.block<3, 3>(0, 3) = dpTdx2.transpose(); grad_p.block<3, 3>(0, 6) = dpTdx3.transpose(); grad_p.block<3, 3>(0, 9) = dpTdx4.transpose();
                Eigen::Matrix<double, 3, 12> grad_q = Eigen::Matrix<double, 3, 12>::Zero();
                grad_q.block<3, 3>(0, 0) = dqTdx1.transpose(); grad_q.block<3, 3>(0, 3) = dqTdx2.transpose(); grad_q.block<3, 3>(0, 6) = dqTdx3.transpose(); grad_q.block<3, 3>(0, 9) = dqTdx4.transpose();
                Eigen::Matrix<double, 3, 12> grad_normalized_p = (Eigen::Matrix3d::Identity() - p.normalized()*p.normalized().transpose()) * grad_p / p.norm();
                Eigen::Matrix<double, 3, 12> grad_normalized_q = (Eigen::Matrix3d::Identity() - q.normalized()*q.normalized().transpose()) * grad_q / q.norm();
                Eigen::Matrix<double, 3, 12> grad_n = (Eigen::Matrix3d::Identity() - n*n.transpose()) * (grad_normalized_p + grad_normalized_q) / (p/p.norm()+q/q.norm()).norm();
                if (p.cross(q).norm() == 0)// p and q are parallel
                if ((x2 - x1).cross(x3 - x1).dot(x4 - x1) == 0)
                    grad_n = (Eigen::Matrix3d::Identity() - n*n.transpose()) * (grad_normalized_p + grad_normalized_q) / ((x2 - x1).cross(x3 - x1)).norm();                        

                Eigen::Matrix<double, 1, 12> grad_theta = x_e.transpose()*L.transpose()*grad_n + n.transpose()*L;
                Eigen::Matrix<double, 1, 12> grad_theta_cubic = x_e.transpose()*L.transpose()*grad_n;
                Eigen::Matrix<double, 1, 12> grad_theta_qudratic = n.transpose()*L;
                // curvature change
                const double curvature = 2 * (theta - theta0) / (restHeight1 + restHeight2);
                // bending moment
                const double moment = bendingRigidity * curvature;
                // bending energy
                bendingEnergy += 0.5 * restStencilArea * curvature * moment;

                // gradient of curvature
                Eigen::Matrix<double, 12, 1> pCurvaturepUi = 2 * grad_theta.transpose() / (restHeight1 + restHeight2);
                // gradient of bending moment
                Eigen::Matrix<double, 12, 1> pMomentpUi = bendingRigidity * pCurvaturepUi;
                
                // Hessian of bending energy
                Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
                // if (isLocalProj) 
                //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);

                // gradient of bending energy
                Eigen::Matrix<double, 12, 1> eleBendingForce = restStencilArea * curvature * pMomentpUi; 
                
                for (int k = 0; k < 4; ++k)
                    globalBendingForces.segment<3>(3 * elemVInd[k]) += eleBendingForce.segment<3>(3 * k);
                const int offset = i * 4 * 3 * 4 * 3;
                for (int k = 0; k < 4; ++k)
                    for (int d = 0; d < 3; ++d)
                        for (int s = 0; s < 4; ++s)
                            for (int t = 0; t < 3; ++t){
                                bendingStiffK[ offset + 
                                + k * 3 * 4 * 3
                                + d * 4 * 3
                                + s * 3
                                + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, eleBendingStiffMatrix(s * 3 + t, k * 3 + d)); 
                            }
            } 
        } 
    }

    if(derivative){
        (*derivative) = globalBendingForces;
    }

    if(hessian){
        (*hessian).resize(nedges * 12 * 12);
        (*hessian) = bendingStiffK;
    }

    return bendingEnergy;
}



// // QTS
// #include "ElasticEnergy.h"
// #include "../Common/CommonFunctions.h"
// #include <tbb/tbb.h>
// #include <iostream>

// Eigen::Matrix<double, 3, 9> grad_triangle_normal(const Eigen::Matrix<double, 3, 1>& node4, const Eigen::Matrix<double, 3, 1>& node5, const Eigen::Matrix<double, 3, 1>& node6,
//                              const Eigen::Matrix<double, 3, 1>& nodalU4, const Eigen::Matrix<double, 3, 1>& nodalU5, const Eigen::Matrix<double, 3, 1>& nodalU6) 
// {
//     const double x4 = node4(0); const double y4 = node4(1); const double z4 = node4(2);
//     const double x5 = node5(0); const double y5 = node5(1); const double z5 = node5(2);
//     const double x6 = node6(0); const double y6 = node6(1); const double z6 = node6(2);
//     const double u4 = nodalU4(0); const double v4 = nodalU4(1); const double w4 = nodalU4(2);
//     const double u5 = nodalU5(0); const double v5 = nodalU5(1); const double w5 = nodalU5(2);
//     const double u6 = nodalU6(0); const double v6 = nodalU6(1); const double w6 = nodalU6(2);

//     const double nx = (y5+v5-y4-v4)*(z6+w6-z4-w4)-(z5+w5-z4-w4)*(y6+v6-y4-v4);
//     const double ny = -((x5+u5-x4-u4)*(z6+w6-z4-w4)-(z5+w5-z4-w4)*(x6+u6-x4-u4));
//     const double nz = (x5+u5-x4-u4)*(y6+v6-y4-v4)-(y5+v5-y4-v4)*(x6+u6-x4-u4);
//     const double len_n = sqrt(std::pow(nx, 2) + std::pow(ny, 2) + std::pow(nz, 2));
    
//     double len_n_u4 = -(ny*((z6+w6-z4-w4)-(z5+w5-z4-w4)) +nz*(-(y6+v6-y4-v4)+(y5+v5-y4-v4)))/ std::pow(len_n, 3);
//     double len_n_v4 = -(nx*(-(z6+w6-z4-w4)+(z5+w5-z4-w4))+nz*(-(x5+u5-x4-u4)+(x6+u6-x4-u4)))/ std::pow(len_n, 3);
//     double len_n_w4 = -(nx*(-(y5+v5-y4-v4)+(y6+v6-y4-v4))+ny*((x5+u5-x4-u4)-(x6+u6-x4-u4))) / std::pow(len_n, 3);
//     double len_n_u5 = -(-ny*(z6+w6-z4-w4)+nz*(y6+v6-y4-v4))/std::pow(len_n, 3);
//     double len_n_v5 = -( nx*(z6+w6-z4-w4)-nz*(x6+u6-x4-u4))/std::pow(len_n, 3);
//     double len_n_w5 = -(-nx*(y6+v6-y4-v4)+ny*(x6+u6-x4-u4))/std::pow(len_n, 3);
//     double len_n_u6 = -( ny*(z5+w5-z4-w4)-nz*(y5+v5-y4-v4))/std::pow(len_n, 3);
//     double len_n_v6 = -(-nx*(z5+w5-z4-w4)+nz*(x5+u5-x4-u4))/std::pow(len_n, 3);
//     double len_n_w6 = -( nx*(y5+v5-y4-v4)-ny*(x5+u5-x4-u4))/std::pow(len_n, 3);

//     const double nx_u4 = nx*len_n_u4;
//     const double nx_v4 = (-(z6+w6-z4-w4)+(z5+w5-z4-w4))/len_n + nx*len_n_v4;
//     const double nx_w4 = (-(y5+v5-y4-v4)+(y6+v6-y4-v4))/len_n + nx*len_n_w4;
//     const double nx_u5 = nx*len_n_u5;
//     const double nx_v5 = (z6+w6-z4-w4) /len_n+nx*len_n_v5;
//     const double nx_w5 = -(y6+v6-y4-v4)/len_n+nx*len_n_w5;
//     const double nx_u6 = nx*len_n_u6;
//     const double nx_v6 = -(z5+w5-z4-w4)/len_n+nx*len_n_v6;
//     const double nx_w6 = (y5+v5-y4-v4) /len_n+nx*len_n_w6;
//     const double ny_u4 = ((z6+w6-z4-w4)-(z5+w5-z4-w4))/len_n+ny*len_n_u4;
//     const double ny_v4 = ny*len_n_v4;
//     const double ny_w4 = ((x5+u5-x4-u4)-(x6+u6-x4-u4))/len_n+ny*len_n_w4;
//     const double ny_u5 = -(z6+w6-z4-w4)/len_n+ny*len_n_u5;
//     const double ny_v5 = ny*len_n_v5;
//     const double ny_w5 = (x6+u6-x4-u4)/len_n+ny*len_n_w5;
//     const double ny_u6 = (z5+w5-z4-w4)/len_n+ny*len_n_u6;
//     const double ny_v6 = ny*len_n_v6;
//     const double ny_w6 = -(x5+u5-x4-u4)/len_n+ny*len_n_w6;
//     const double nz_u4 = (-(y6+v6-y4-v4)+(y5+v5-y4-v4))/len_n+nz*len_n_u4;
//     const double nz_v4 = (-(x5+u5-x4-u4)+(x6+u6-x4-u4))/len_n+nz*len_n_v4;
//     const double nz_w4 = nz*len_n_w4;
//     const double nz_u5 = (y6+v6-y4-v4) /len_n+nz*len_n_u5;
//     const double nz_v5 = -(x6+u6-x4-u4)/len_n+nz*len_n_v5;
//     const double nz_w5 = nz*len_n_w5;
//     const double nz_u6 = -(y5+v5-y4-v4)/len_n+nz*len_n_u6;
//     const double nz_v6 = (x5+u5-x4-u4) /len_n+nz*len_n_v6;
//     const double nz_w6 = nz*len_n_w6;

//     Eigen::Matrix<double, 3, 9> grad_n;
//     grad_n << nx_u4, nx_v4, nx_w4, nx_u5, nx_v5, nx_w5, nx_u6, nx_v6, nx_w6,
//               ny_u4, ny_v4, ny_w4, ny_u5, ny_v5, ny_w5, ny_u6, ny_v6, ny_w6,
//               nz_u4, nz_v4, nz_w4, nz_u5, nz_v5, nz_w5, nz_u6, nz_v6, nz_w6;

//     return grad_n;
// }

// double corotationalCurveHingeBendingEnergy(
//     const MeshConnectivity& mesh,
//     const Eigen::MatrixXd& iniPos,
//     const Eigen::MatrixXd& curPos,
//     double YoungsModulus, double PoissonsRatio, double thickness,
//     const std::vector<Eigen::Matrix2d>& abars,
//     const SecondFundamentalFormDiscretization& sff,
//     Eigen::VectorXd* derivative, // positions, then thetas
//     std::vector<Eigen::Triplet<double> >* hessian,
//     bool isLocalProj,
//     bool isParallel)
// {
//     int nfaces = mesh.nFaces();
//     int nedges = mesh.nEdges();
//     int nverts = curPos.rows();
//     int nedgedofs = sff.numExtraDOFs();
//     int dofs_ = 3 * nverts + nedgedofs * nedges;
//     Eigen::VectorXd disp(3 * nverts); // displacement
//     for (int i = 0; i < nverts; i++)
//         disp.segment<3>(3 * i) = curPos.row(i) - iniPos.row(i);

//     if (derivative)
//     {
//         derivative->resize(dofs_);
//         derivative->setZero();
//     }
//     if (hessian)
//     {
//         hessian->clear();
//     }

//     double bendingEnergy = 0.0;
//     std::vector<Eigen::Matrix<double, 3, 1>> eleOrderInterFs(nedges * 4, Eigen::Matrix<double, 3, 1>::Zero());
//     Eigen::VectorXd globalBendingForces = Eigen::VectorXd::Zero(dofs_);
//     std::vector<Eigen::Triplet<double>> bendingStiffK;
//     bendingStiffK.clear();
//     bendingStiffK.resize(nedges * 12 * 12); 
//     const double bendingRigidity = std::pow(thickness, 3) / 12.0 * YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
//     if (false) // parallel version should be improved for stability for this model
//     {
//         auto eleVInds = std::vector<Eigen::Vector4i>(nedges);
//         auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
//         {
//             for (uint32_t i = range.begin(); i < range.end(); ++i) // loop over all the edges
//             {
//                 int adjFace0 = mesh.edgeFace(i, 0);
//                 int adjFace1 = mesh.edgeFace(i, 1);
//                 Eigen::Vector4i elemVInd;
//                 elemVInd(0) = mesh.edgeVertex(i, 0);
//                 elemVInd(1) = mesh.edgeVertex(i, 1);
//                 elemVInd(2) = mesh.edgeOppositeVertex(i, 0);
//                 elemVInd(3) = mesh.edgeOppositeVertex(i, 1);
//                 eleVInds[i] = elemVInd;
//                 if (adjFace0!=-1 && adjFace1!=-1) 
//                 {
//                     const Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
//                     const Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
//                     const Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
//                     const Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
//                     Eigen::Matrix<double, 12, 1> X_e = Eigen::Matrix<double, 12, 1>::Zero();
//                     X_e.block<3, 1>(0, 0) = X1; X_e.block<3, 1>(3, 0) = X2; X_e.block<3, 1>(6, 0) = X3; X_e.block<3, 1>(9, 0) = X4;
//                     const double restLenEdge12 = (X2 - X1).norm();
//                     const Eigen::Matrix<double, 3, 1> XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
//                     const Eigen::Matrix<double, 3, 1> XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
//                     const double restHeight1 = (X3 - XP).norm();
//                     const double restHeight2 = (X4 - XQ).norm();
//                     const double P2 = (X2 - XP).norm();
//                     const double P1 = restLenEdge12 - P2;
//                     const double Q2 = (X2 - XQ).norm();
//                     const double Q1 = restLenEdge12 - Q2;
//                     const double L1 = - (P2/restHeight1 + Q2/restHeight2)/restLenEdge12;
//                     const double L2 = - (P1/restHeight1 + Q1/restHeight2)/restLenEdge12;
//                     const double L3 = 1/restHeight1;
//                     const double L4 = 1/restHeight2;
//                     Eigen::Matrix<double, 3, 12> L = Eigen::Matrix<double, 3, 12>::Zero();
//                     L.block<3, 3>(0, 0) = L1 * Eigen::Matrix3d::Identity();
//                     L.block<3, 3>(0, 3) = L2 * Eigen::Matrix3d::Identity();
//                     L.block<3, 3>(0, 6) = L3 * Eigen::Matrix3d::Identity();
//                     L.block<3, 3>(0, 9) = L4 * Eigen::Matrix3d::Identity();
//                     const double restStencilArea = 0.5 * restLenEdge12 * (restHeight1 + restHeight2); 
//                     const Eigen::Matrix<double, 3, 1> p0 = X3 - XP, q0 = X4 - XQ;
//                     const double s_P = (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm(), s_Q = (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm();
//                     Eigen::Matrix<double, 3, 1> n0;
//                     Eigen::Matrix<double, 3, 1> e0 = X2 - X1, e1 = X3 - X1, e2 = X4 - X1;
//                     if (e0.cross(e1).dot(e2) ==0)
//                         n0 = (X2 - X1).cross(X3 - X1).normalized();
//                     else
//                         n0 = (p0/p0.norm()+q0/q0.norm()).normalized();
//                     const double theta0 = n0.transpose() * L * X_e;
                    
//                     const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
//                     const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
//                     const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
//                     const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
//                     Eigen::Matrix<double, 12, 1> x_e = Eigen::Matrix<double, 12, 1>::Zero();
//                     x_e.block<3, 1>(0, 0) = x1; x_e.block<3, 1>(3, 0) = x2; x_e.block<3, 1>(6, 0) = x3; x_e.block<3, 1>(9, 0) = x4;
//                     const Eigen::Matrix<double, 3, 1> e = (x2 - x1).normalized();
//                     const double curLenEdge12 = (x2 - x1).norm();
//                     const Eigen::Matrix<double, 3, 1> xp = (x2 - x1) * (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
//                     const Eigen::Matrix<double, 3, 1> xq = (x2 - x1) * (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
//                     const Eigen::Matrix<double, 3, 1> p = x3 - xp, q = x4 - xq;
//                     const double s_p = (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm(), s_q = (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm();
//                     Eigen::Matrix<double, 3, 1> n;
//                     if ((x2 - x1).cross(x3 - x1).dot(x4 - x1) ==0)
//                         n = (x2 - x1).cross(x3 - x1).normalized();
//                     else
//                         n = (p/p.norm()+q/q.norm()).normalized();
//                     const double theta = n.transpose() * L * x_e;
//                     // gradient of p and q
//                     const Eigen::Matrix<double, 3, 3> dpTdx1 = -(1-s_p)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + p*e.transpose()/curLenEdge12;
//                     const Eigen::Matrix<double, 3, 3> dpTdx2 = -s_p*(Eigen::Matrix3d::Identity()-e*e.transpose()) - p*e.transpose()/curLenEdge12;
//                     const Eigen::Matrix<double, 3, 3> dpTdx3 = Eigen::Matrix3d::Identity() - e*e.transpose();
//                     const Eigen::Matrix<double, 3, 3> dpTdx4 = Eigen::Matrix3d::Zero();
//                     const Eigen::Matrix<double, 3, 3> dqTdx1 = -(1-s_q)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + q*e.transpose()/curLenEdge12;
//                     const Eigen::Matrix<double, 3, 3> dqTdx2 = -s_q*(Eigen::Matrix3d::Identity()-e*e.transpose()) - q*e.transpose()/curLenEdge12;
//                     const Eigen::Matrix<double, 3, 3> dqTdx3 = Eigen::Matrix3d::Zero();
//                     const Eigen::Matrix<double, 3, 3> dqTdx4 = Eigen::Matrix3d::Identity() - e*e.transpose();
//                     Eigen::Matrix<double, 3, 12> grad_p = Eigen::Matrix<double, 3, 12>::Zero();
//                     grad_p.block<3, 3>(0, 0) = dpTdx1.transpose(); grad_p.block<3, 3>(0, 3) = dpTdx2.transpose(); grad_p.block<3, 3>(0, 6) = dpTdx3.transpose(); grad_p.block<3, 3>(0, 9) = dpTdx4.transpose();
//                     Eigen::Matrix<double, 3, 12> grad_q = Eigen::Matrix<double, 3, 12>::Zero();
//                     grad_q.block<3, 3>(0, 0) = dqTdx1.transpose(); grad_q.block<3, 3>(0, 3) = dqTdx2.transpose(); grad_q.block<3, 3>(0, 6) = dqTdx3.transpose(); grad_q.block<3, 3>(0, 9) = dqTdx4.transpose();
//                     Eigen::Matrix<double, 3, 12> grad_normalized_p = (Eigen::Matrix3d::Identity() - p.normalized()*p.normalized().transpose()) * grad_p;
//                     Eigen::Matrix<double, 3, 12> grad_normalized_q = (Eigen::Matrix3d::Identity() - q.normalized()*q.normalized().transpose()) * grad_q;
//                     // gradient of n
//                     Eigen::Matrix<double, 3, 12> grad_n = Eigen::Matrix<double, 3, 12>::Zero();
//                     if ((x2 - x1).cross(x3 - x1).dot(x4 - x1) ==0)
//                     {
//                         Eigen::Matrix<double, 3, 1> u1 = x1 - X1, u2 = x2 - X2, u3 = x3 - X3;
//                         Eigen::Matrix<double, 3, 9> grad_T1 = grad_triangle_normal(X1, X2, X3, u1, u2, u3);
//                         grad_n.template block<3, 9>(0, 0) = grad_T1;
//                     }
//                     else
//                         grad_n = (Eigen::Matrix3d::Identity() - n*n.transpose()) * (grad_normalized_p + grad_normalized_q);
//                     Eigen::Matrix<double, 3, 1> curDelx = L * x_e;
//                     // gradient of theta
//                     Eigen::Matrix<double, 1, 12> grad_theta = grad_n.row(0)*curDelx(0) + grad_n.row(1)*curDelx(1) + grad_n.row(2)*curDelx(2) + n.transpose()*L;
                    
//                     // curvature change
//                     const double curvature = 2 * (theta - theta0) / (restHeight1 + restHeight2);
//                     // bending moment
//                     const double moment = bendingRigidity * curvature;
//                     // bending energy
//                     bendingEnergy += 0.5 * restStencilArea * curvature * moment;

//                     // gradient of curvature
//                     Eigen::Matrix<double, 12, 1> pCurvaturepUi = 2 * grad_theta.transpose() / (restHeight1 + restHeight2);
//                     // gradient of bending moment
//                     Eigen::Matrix<double, 12, 1> pMomentpUi = bendingRigidity * pCurvaturepUi;
//                     // gradient of bending energy
//                     Eigen::Matrix<double, 12, 1> eleBendingForce = restStencilArea * curvature * pMomentpUi;
//                     for (int k = 0; k < 4; ++k)
//                             globalBendingForces.segment<3>(3 * elemVInd[k]) += eleBendingForce.segment<3>(3 * k);

//                     // Hessian of bending energy
//                     Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
//                     // if (isLocalProj) 
//                     //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);
//                     const int offset = i * 4 * 3 * 4 * 3;
//                     for (int k = 0; k < 4; ++k)
//                         for (int d = 0; d < 3; ++d)
//                             for (int s = 0; s < 4; ++s)
//                                 for (int t = 0; t < 3; ++t){
//                                     bendingStiffK[ offset + 
//                                     + k * 3 * 4 * 3
//                                     + d * 4 * 3
//                                     + s * 3
//                                     + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, eleBendingStiffMatrix(s * 3 + t, k * 3 + d)); 
//                                 }
//                 }
//             } 
//         }; 
//         tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nedges);
//         tbb::parallel_for(rangex, computeBending);

//     }
//     else // serial version
//     {
//         for (int i = 0; i < nedges; i++) 
//         {
//             int adjFace0 = mesh.edgeFace(i, 0);
//             int adjFace1 = mesh.edgeFace(i, 1);
//             if (adjFace0!=-1 && adjFace1!=-1) 
//             {
//                 Eigen::Vector4i elemVInd;
//                 elemVInd(0) = mesh.edgeVertex(i, 0);
//                 elemVInd(1) = mesh.edgeVertex(i, 1);
//                 elemVInd(2) = mesh.edgeOppositeVertex(i, 0);
//                 elemVInd(3) = mesh.edgeOppositeVertex(i, 1);

//                 // initial configuration
//                 const Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
//                 const Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
//                 const Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
//                 const Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
//                 Eigen::Matrix<double, 12, 1> X_e = Eigen::Matrix<double, 12, 1>::Zero();
//                 X_e.block<3, 1>(0, 0) = X1; X_e.block<3, 1>(3, 0) = X2; X_e.block<3, 1>(6, 0) = X3; X_e.block<3, 1>(9, 0) = X4;
//                 const double restLenEdge12 = (X2 - X1).norm();
//                 const Eigen::Matrix<double, 3, 1> XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
//                 const Eigen::Matrix<double, 3, 1> XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
//                 const double restHeight1 = (X3 - XP).norm();
//                 const double restHeight2 = (X4 - XQ).norm();
//                 const double P2 = (X2 - XP).norm();
//                 const double P1 = restLenEdge12 - P2;
//                 const double Q2 = (X2 - XQ).norm();
//                 const double Q1 = restLenEdge12 - Q2;
//                 const double L1 = - (P2/restHeight1 + Q2/restHeight2)/restLenEdge12;
//                 const double L2 = - (P1/restHeight1 + Q1/restHeight2)/restLenEdge12;
//                 const double L3 = 1/restHeight1;
//                 const double L4 = 1/restHeight2;
//                 Eigen::Matrix<double, 3, 12> L = Eigen::Matrix<double, 3, 12>::Zero();
//                 L.block<3, 3>(0, 0) = L1 * Eigen::Matrix3d::Identity();
//                 L.block<3, 3>(0, 3) = L2 * Eigen::Matrix3d::Identity();
//                 L.block<3, 3>(0, 6) = L3 * Eigen::Matrix3d::Identity();
//                 L.block<3, 3>(0, 9) = L4 * Eigen::Matrix3d::Identity();
//                 const double restStencilArea = 0.5 * restLenEdge12 * (restHeight1 + restHeight2); 
//                 const Eigen::Matrix<double, 3, 1> p0 = X3 - XP, q0 = X4 - XQ;
//                 const double s_P = (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm(), s_Q = (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm();
//                 Eigen::Matrix<double, 3, 1> n0;
//                 Eigen::Matrix<double, 3, 1> e0 = X2 - X1, e1 = X3 - X1, e2 = X4 - X1;
//                 if (e0.cross(e1).dot(e2) ==0)
//                     n0 = (X2 - X1).cross(X3 - X1).normalized();
//                 else
//                     n0 = (p0/p0.norm()+q0/q0.norm()).normalized();
//                 const double theta0 = n0.transpose() * L * X_e;
                
//                 const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
//                 const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
//                 const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
//                 const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
//                 Eigen::Matrix<double, 12, 1> x_e = Eigen::Matrix<double, 12, 1>::Zero();
//                 x_e.block<3, 1>(0, 0) = x1; x_e.block<3, 1>(3, 0) = x2; x_e.block<3, 1>(6, 0) = x3; x_e.block<3, 1>(9, 0) = x4;
//                 const Eigen::Matrix<double, 3, 1> e = (x2 - x1).normalized();
//                 const double curLenEdge12 = (x2 - x1).norm();
//                 const Eigen::Matrix<double, 3, 1> xp = (x2 - x1) * (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
//                 const Eigen::Matrix<double, 3, 1> xq = (x2 - x1) * (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm() + x1;
//                 const Eigen::Matrix<double, 3, 1> p = x3 - xp, q = x4 - xq;
//                 const double s_p = (x3 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm(), s_q = (x4 - x1).dot(x2 - x1) / (x2 - x1).squaredNorm();
//                 Eigen::Matrix<double, 3, 1> n;
//                 if ((x2 - x1).cross(x3 - x1).dot(x4 - x1) ==0)
//                     n = (x2 - x1).cross(x3 - x1).normalized();
//                 else
//                     n = (p/p.norm()+q/q.norm()).normalized();
//                 const double theta = n.transpose() * L * x_e;
//                 // gradient of p and q
//                 const Eigen::Matrix<double, 3, 3> dpTdx1 = -(1-s_p)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + p*e.transpose()/curLenEdge12;
//                 const Eigen::Matrix<double, 3, 3> dpTdx2 = -s_p*(Eigen::Matrix3d::Identity()-e*e.transpose()) - p*e.transpose()/curLenEdge12;
//                 const Eigen::Matrix<double, 3, 3> dpTdx3 = Eigen::Matrix3d::Identity() - e*e.transpose();
//                 const Eigen::Matrix<double, 3, 3> dpTdx4 = Eigen::Matrix3d::Zero();
//                 const Eigen::Matrix<double, 3, 3> dqTdx1 = -(1-s_q)*(Eigen::Matrix3d::Identity()-e*e.transpose()) + q*e.transpose()/curLenEdge12;
//                 const Eigen::Matrix<double, 3, 3> dqTdx2 = -s_q*(Eigen::Matrix3d::Identity()-e*e.transpose()) - q*e.transpose()/curLenEdge12;
//                 const Eigen::Matrix<double, 3, 3> dqTdx3 = Eigen::Matrix3d::Zero();
//                 const Eigen::Matrix<double, 3, 3> dqTdx4 = Eigen::Matrix3d::Identity() - e*e.transpose();
//                 Eigen::Matrix<double, 3, 12> grad_p = Eigen::Matrix<double, 3, 12>::Zero();
//                 grad_p.block<3, 3>(0, 0) = dpTdx1.transpose(); grad_p.block<3, 3>(0, 3) = dpTdx2.transpose(); grad_p.block<3, 3>(0, 6) = dpTdx3.transpose(); grad_p.block<3, 3>(0, 9) = dpTdx4.transpose();
//                 Eigen::Matrix<double, 3, 12> grad_q = Eigen::Matrix<double, 3, 12>::Zero();
//                 grad_q.block<3, 3>(0, 0) = dqTdx1.transpose(); grad_q.block<3, 3>(0, 3) = dqTdx2.transpose(); grad_q.block<3, 3>(0, 6) = dqTdx3.transpose(); grad_q.block<3, 3>(0, 9) = dqTdx4.transpose();
//                 Eigen::Matrix<double, 3, 12> grad_normalized_p = (Eigen::Matrix3d::Identity() - p.normalized()*p.normalized().transpose()) * grad_p;
//                 Eigen::Matrix<double, 3, 12> grad_normalized_q = (Eigen::Matrix3d::Identity() - q.normalized()*q.normalized().transpose()) * grad_q;
//                 // gradient of n
//                 Eigen::Matrix<double, 3, 12> grad_n = Eigen::Matrix<double, 3, 12>::Zero();
//                 if ((x2 - x1).cross(x3 - x1).dot(x4 - x1) ==0)
//                 {
//                     Eigen::Matrix<double, 3, 1> u1 = x1 - X1, u2 = x2 - X2, u3 = x3 - X3;
//                     Eigen::Matrix<double, 3, 9> grad_T1 = grad_triangle_normal(X1, X2, X3, u1, u2, u3);
//                     grad_n.template block<3, 9>(0, 0) = grad_T1;
//                 }
//                 else
//                     grad_n = (Eigen::Matrix3d::Identity() - n*n.transpose()) * (grad_normalized_p + grad_normalized_q);
//                 Eigen::Matrix<double, 3, 1> curDelx = L * x_e;
//                 // gradient of theta
//                 Eigen::Matrix<double, 1, 12> grad_theta = grad_n.row(0)*curDelx(0) + grad_n.row(1)*curDelx(1) + grad_n.row(2)*curDelx(2) + n.transpose()*L;
                
//                 // curvature change
//                 const double curvature = 2 * (theta - theta0) / (restHeight1 + restHeight2);
//                 // bending moment
//                 const double moment = bendingRigidity * curvature;
//                 // bending energy
//                 bendingEnergy += 0.5 * restStencilArea * curvature * moment;

//                 // gradient of curvature
//                 Eigen::Matrix<double, 12, 1> pCurvaturepUi = 2 * grad_theta.transpose() / (restHeight1 + restHeight2);
//                 // gradient of bending moment
//                 Eigen::Matrix<double, 12, 1> pMomentpUi = bendingRigidity * pCurvaturepUi;
//                 // gradient of bending energy
//                 Eigen::Matrix<double, 12, 1> eleBendingForce = restStencilArea * curvature * pMomentpUi;
//                 for (int k = 0; k < 4; ++k)
//                     globalBendingForces.segment<3>(3 * elemVInd[k]) += eleBendingForce.segment<3>(3 * k);

//                 // Hessian of bending energy
//                 Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
//                 // if (isLocalProj) 
//                 //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);
//                 const int offset = i * 4 * 3 * 4 * 3;
//                 for (int k = 0; k < 4; ++k)
//                     for (int d = 0; d < 3; ++d)
//                         for (int s = 0; s < 4; ++s)
//                             for (int t = 0; t < 3; ++t){
//                                 bendingStiffK[ offset + 
//                                 + k * 3 * 4 * 3
//                                 + d * 4 * 3
//                                 + s * 3
//                                 + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, eleBendingStiffMatrix(s * 3 + t, k * 3 + d)); 
//                             }
//             } 
//         } 
//     }

//     if(derivative){
//         (*derivative) = globalBendingForces;
//     }

//     if(hessian){
//         (*hessian).resize(nedges * 12 * 12);
//         (*hessian) = bendingStiffK;
//     }

//     return bendingEnergy;
// }