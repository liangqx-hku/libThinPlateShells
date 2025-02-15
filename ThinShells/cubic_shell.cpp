#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

// Cubic Shell
double cubicShellBendingEnergy(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& iniPos,
    const Eigen::MatrixXd& curPos,
    double YoungsModulus, double PoissonsRatio, double thickness,
    const std::vector<Eigen::Matrix2d>& abars,
    const SecondFundamentalFormDiscretization& sff,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian,
    bool isLocalProj,
    bool isParallel)
{
    int nfaces = mesh.nFaces();
    int nedges = mesh.nEdges();
    int nverts = curPos.rows();
    int nedgedofs = sff.numExtraDOFs();
    int dofs_ = 3 * nverts + nedgedofs * nedges;
    Eigen::VectorXd disp(3 * nverts); // displacement
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
            for (uint32_t i = range.begin(); i < range.end(); ++i) // loop over all the edges
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
                    const Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
                    const Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
                    const Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
                    const Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
                    const Eigen::Matrix<double, 3, 1> restEdgeBA = (X2 - X1); // e0
                    const Eigen::Matrix<double, 3, 1> restEdgeBC = (X3 - X1); // e1
                    const Eigen::Matrix<double, 3, 1> restEdgeBD = (X4 - X1); // e2
                    const Eigen::Matrix<double, 3, 1> restEdgeCA = (X3 - X2); // e3
                    const Eigen::Matrix<double, 3, 1> restEdgeDA = (X4 - X2); // e4
                    const double restLenEdgeBA = restEdgeBA.norm();
                    const Eigen::Matrix<double, 3, 1> restN1 = restEdgeBA.cross(restEdgeBC);
                    const Eigen::Matrix<double, 3, 1> restN2 = restEdgeBD.cross(restEdgeBA);
                    const double restH1 = restN1.norm() / restLenEdgeBA; 
                    const double restH2 = restN2.norm() / restLenEdgeBA; 
                    const double cot01 = cotTheta(restEdgeBA, restEdgeBC);
                    const double cot02 = cotTheta(restEdgeBA, restEdgeBD);
                    const double cot03 = cotTheta(-restEdgeBA, restEdgeCA);
                    const double cot04 = cotTheta(-restEdgeBA, restEdgeDA);
                    const double L1 = cot03 + cot04;
                    const double L2 = cot01 + cot02;
                    const double L3 = -cot01 - cot03;
                    const double L4 = -cot02 - cot04;
                    const Eigen::Matrix<double, 3, 1> restT0 = -cot03 * restEdgeBC -cot01 * restEdgeCA;
                    const Eigen::Matrix<double, 3, 1> restT1 = -cot04 * restEdgeBD -cot02 * restEdgeDA;
                    const double restE012 = (restEdgeBA.cross(restEdgeBC)).dot(restEdgeBD);
                    const double cosThetaBar = - restT0.dot(restT1) / std::pow(restLenEdgeBA, 2);
                    const double beta = 1 / restLenEdgeBA * (cot01 + cot03) * (cot02 + cot04);
                    const double sinThetaBar = - beta * restE012 / std::pow(restLenEdgeBA, 2);
                    const double bending_scale = 1.0; 
                    const double restStencilArea = 0.5 * restLenEdgeBA * (restH1 + restH2) * bending_scale; 
                    const double kbar = 3 * bendingRigidity * (cot01 - cot03) * (cot04 - cot02) * restE012 / (restStencilArea * std::pow(restLenEdgeBA, 3));
                    Eigen::Matrix<double, 1, 4> L = Eigen::Matrix<double, 1, 4>::Zero();
                    L << L1, L2, L3, L4; 
                    Eigen::Matrix<double, 3, 12> L_flatten = Eigen::Matrix<double, 3, 12>::Zero();
                    L_flatten.block<3, 3>(0, 0) = L1 * Eigen::Matrix<double, 3, 3>::Identity();
                    L_flatten.block<3, 3>(0, 3) = L2 * Eigen::Matrix<double, 3, 3>::Identity();
                    L_flatten.block<3, 3>(0, 6) = L3 * Eigen::Matrix<double, 3, 3>::Identity();
                    L_flatten.block<3, 3>(0, 9) = L4 * Eigen::Matrix<double, 3, 3>::Identity();
                    Eigen::Matrix<double, 12, 12> Qbar = 3 * bendingRigidity * cosThetaBar / restStencilArea * L_flatten.transpose() * L_flatten;

                    const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
                    const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
                    const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
                    const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
                    Eigen::Matrix<double, 12, 1> x_vec = Eigen::Matrix<double, 12, 1>::Zero();
                    x_vec.block<3, 1>(0, 0) = x1; x_vec.block<3, 1>(3, 0) = x2; x_vec.block<3, 1>(6, 0) = x3; x_vec.block<3, 1>(9, 0) = x4;
                    const Eigen::Matrix<double, 3, 1> curEdgeBA = (x2 - x1); // e0
                    const Eigen::Matrix<double, 3, 1> curEdgeBC = (x3 - x1); // e1
                    const Eigen::Matrix<double, 3, 1> curEdgeBD = (x4 - x1); // e2
                    const Eigen::Matrix<double, 3, 1> curEdgeCA = (x3 - x2); // e3
                    const Eigen::Matrix<double, 3, 1> curEdgeDA = (x4 - x2); // e4
                    const double curLenEdgeBA = curEdgeBA.norm();
                    const Eigen::Matrix<double, 3, 1> n1 = curEdgeBA.cross(curEdgeBC);
                    const Eigen::Matrix<double, 3, 1> n2 = curEdgeBD.cross(curEdgeBA);
                    const double h1 = n1.norm() / curLenEdgeBA; 
                    const double h2 = n2.norm() / curLenEdgeBA; 
                    const double curCot01 = cotTheta(curEdgeBA, curEdgeBC);
                    const double curCot02 = cotTheta(curEdgeBA, curEdgeBD);
                    const double curCot03 = cotTheta(-curEdgeBA, curEdgeCA);
                    const double curCot04 = cotTheta(-curEdgeBA, curEdgeDA);
                    const Eigen::Matrix<double, 3, 1> t0 = -cot03 * curEdgeBC -cot01 * curEdgeCA;
                    const Eigen::Matrix<double, 3, 1> t1 = -cot04 * curEdgeBD -cot02 * curEdgeDA;
                    const double e012 = (curEdgeBA.cross(curEdgeBC)).dot(curEdgeBD);

                    // ---  bending energy ---
                    // thin plate component
                    const double thinPlateEnergy = bendingRigidity * 3/restStencilArea * (std::pow(restLenEdgeBA, 2) + t0.dot(t1)*cosThetaBar); 
                    // cubic term
                    const double cubicEnergy = bendingRigidity * 3 * beta / restStencilArea * e012 * sinThetaBar;
                    bendingEnergy += thinPlateEnergy + cubicEnergy;

                    // --- gradient of bending energy ---
                    // thin plate component
                    Eigen::Matrix<double, 12, 1> thinPlateForce = Qbar * x_vec;
                    // cubic term
                    Eigen::Matrix<double, 3, 1> f2 = kbar * curEdgeBC.cross(curEdgeBD);
                    Eigen::Matrix<double, 3, 1> f3 = kbar * curEdgeBD.cross(curEdgeBA);
                    Eigen::Matrix<double, 3, 1> f4 = kbar * curEdgeBA.cross(curEdgeBC);
                    Eigen::Matrix<double, 3, 1> f1 = -f2 - f3 - f4;
                    Eigen::Matrix<double, 12, 1> cubicForce = Eigen::Matrix<double, 12, 1>::Zero();
                    cubicForce.block<3, 1>(0, 0) = f1; cubicForce.block<3, 1>(3, 0) = f2; cubicForce.block<3, 1>(6, 0) = f3; cubicForce.block<3, 1>(9, 0) = f4;
                    for (int j = 0; j < 4; j++)
                        globalBendingForces.segment<3>(3 * elemVInd[j]) += thinPlateForce.segment<3>(3 * j) + cubicForce.segment<3>(3 * j);
                    
                    // --- hessian of bending energy ---
                    // thin plate component
                    Eigen::Matrix<double, 12, 12> thinPlateStiffMatrix = Qbar;

                    // cubic term is not considered
                    // Eigen::Matrix<double, 12, 12> cubicStiffMatrix = Eigen::Matrix<double, 12, 12>::Zero();
                    // cubicStiffMatrix.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Zero();
                    // cubicStiffMatrix.block<3, 3>(0, 3) = kbar * crossProductMatrix(curEdgeBC - curEdgeBD); // e1 - e2
                    // cubicStiffMatrix.block<3, 3>(0, 6) = kbar * crossProductMatrix(curEdgeBD - curEdgeBA); // e2 - e0
                    // cubicStiffMatrix.block<3, 3>(0, 9) = kbar * crossProductMatrix(curEdgeBA - curEdgeBC); // e0 - e1
                    // cubicStiffMatrix.block<3, 3>(3, 0) = - kbar * crossProductMatrix(curEdgeBD - curEdgeBC); // e2 - e1
                    // cubicStiffMatrix.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Zero();
                    // cubicStiffMatrix.block<3, 3>(3, 6) = - kbar * crossProductMatrix(curEdgeBD); // - e2
                    // cubicStiffMatrix.block<3, 3>(3, 9) = kbar * crossProductMatrix(curEdgeBC); // e1
                    // cubicStiffMatrix.block<3, 3>(6, 0) = - kbar * crossProductMatrix(curEdgeBD - curEdgeBA); // e2 - e0
                    // cubicStiffMatrix.block<3, 3>(6, 3) = kbar * crossProductMatrix(curEdgeBD); // e2
                    // cubicStiffMatrix.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Zero();
                    // cubicStiffMatrix.block<3, 3>(6, 9) = - kbar * crossProductMatrix(curEdgeBA); // - e0
                    // cubicStiffMatrix.block<3, 3>(9, 0) = - kbar * crossProductMatrix(curEdgeBA - curEdgeBC); // e0 - e1
                    // cubicStiffMatrix.block<3, 3>(9, 3) = - kbar * crossProductMatrix(curEdgeBC); // - e1
                    // cubicStiffMatrix.block<3, 3>(9, 6) = kbar * crossProductMatrix(curEdgeBA); // e0
                    // cubicStiffMatrix.block<3, 3>(9, 9) = Eigen::Matrix<double, 3, 3>::Zero();

                    // Hessian of bending energy
                    // neglect the cubic term, only consider the thin plate term
                    Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = thinPlateStiffMatrix;
                    if (isLocalProj) 
                        eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);
                    // assembly the current elemental stiffness matrix to the global bending stiffness matrix
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
        for (int i = 0; i < nedges; i++) 
        {
            int adjFace0 = mesh.edgeFace(i, 0);
            int adjFace1 = mesh.edgeFace(i, 1);
            if (adjFace0!=-1 && adjFace1!=-1) 
            {
                Eigen::Vector4i elemVInd;
                elemVInd(0) = mesh.edgeVertex(i, 0);
                elemVInd(1) = mesh.edgeVertex(i, 1);
                elemVInd(2) = mesh.edgeOppositeVertex(i, 0);
                elemVInd(3) = mesh.edgeOppositeVertex(i, 1);

                const Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
                const Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
                const Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
                const Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
                const Eigen::Matrix<double, 3, 1> restEdgeBA = (X2 - X1); // e0
                const Eigen::Matrix<double, 3, 1> restEdgeBC = (X3 - X1); // e1
                const Eigen::Matrix<double, 3, 1> restEdgeBD = (X4 - X1); // e2
                const Eigen::Matrix<double, 3, 1> restEdgeCA = (X3 - X2); // e3
                const Eigen::Matrix<double, 3, 1> restEdgeDA = (X4 - X2); // e4
                const double restLenEdgeBA = restEdgeBA.norm();
                const Eigen::Matrix<double, 3, 1> restN1 = restEdgeBA.cross(restEdgeBC);
                const Eigen::Matrix<double, 3, 1> restN2 = restEdgeBD.cross(restEdgeBA);
                const double restH1 = restN1.norm() / restLenEdgeBA; 
                const double restH2 = restN2.norm() / restLenEdgeBA; 
                const double cot01 = cotTheta(restEdgeBA, restEdgeBC);
                const double cot02 = cotTheta(restEdgeBA, restEdgeBD);
                const double cot03 = cotTheta(-restEdgeBA, restEdgeCA);
                const double cot04 = cotTheta(-restEdgeBA, restEdgeDA);
                const double L1 = cot03 + cot04;
                const double L2 = cot01 + cot02;
                const double L3 = -cot01 - cot03;
                const double L4 = -cot02 - cot04;
                const Eigen::Matrix<double, 3, 1> restT0 = -cot03 * restEdgeBC -cot01 * restEdgeCA;
                const Eigen::Matrix<double, 3, 1> restT1 = -cot04 * restEdgeBD -cot02 * restEdgeDA;
                const double restE012 = (restEdgeBA.cross(restEdgeBC)).dot(restEdgeBD);
                const double cosThetaBar = - restT0.dot(restT1) / std::pow(restLenEdgeBA, 2);
                const double beta = 1 / restLenEdgeBA * (cot01 + cot03) * (cot02 + cot04);
                const double sinThetaBar = - beta * restE012 / std::pow(restLenEdgeBA, 2);
                const double bending_scale = 1.0; 
                const double restStencilArea = 0.5 * restLenEdgeBA * (restH1 + restH2) * bending_scale; 
                const double kbar = 3 * bendingRigidity * (cot01 - cot03) * (cot04 - cot02) * restE012 / (restStencilArea * std::pow(restLenEdgeBA, 3));
                Eigen::Matrix<double, 1, 4> L = Eigen::Matrix<double, 1, 4>::Zero();
                L << L1, L2, L3, L4; 
                Eigen::Matrix<double, 3, 12> L_flatten = Eigen::Matrix<double, 3, 12>::Zero();
                L_flatten.block<3, 3>(0, 0) = L1 * Eigen::Matrix<double, 3, 3>::Identity();
                L_flatten.block<3, 3>(0, 3) = L2 * Eigen::Matrix<double, 3, 3>::Identity();
                L_flatten.block<3, 3>(0, 6) = L3 * Eigen::Matrix<double, 3, 3>::Identity();
                L_flatten.block<3, 3>(0, 9) = L4 * Eigen::Matrix<double, 3, 3>::Identity();
                Eigen::Matrix<double, 12, 12> Qbar = 3 * bendingRigidity * cosThetaBar / restStencilArea * L_flatten.transpose() * L_flatten;

                const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
                const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
                const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
                const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
                Eigen::Matrix<double, 12, 1> x_vec = Eigen::Matrix<double, 12, 1>::Zero();
                x_vec.block<3, 1>(0, 0) = x1; x_vec.block<3, 1>(3, 0) = x2; x_vec.block<3, 1>(6, 0) = x3; x_vec.block<3, 1>(9, 0) = x4;
                const Eigen::Matrix<double, 3, 1> curEdgeBA = (x2 - x1); // e0
                const Eigen::Matrix<double, 3, 1> curEdgeBC = (x3 - x1); // e1
                const Eigen::Matrix<double, 3, 1> curEdgeBD = (x4 - x1); // e2
                const Eigen::Matrix<double, 3, 1> curEdgeCA = (x3 - x2); // e3
                const Eigen::Matrix<double, 3, 1> curEdgeDA = (x4 - x2); // e4
                const double curLenEdgeBA = curEdgeBA.norm();
                const Eigen::Matrix<double, 3, 1> n1 = curEdgeBA.cross(curEdgeBC);
                const Eigen::Matrix<double, 3, 1> n2 = curEdgeBD.cross(curEdgeBA);
                const double h1 = n1.norm() / curLenEdgeBA; 
                const double h2 = n2.norm() / curLenEdgeBA; 
                const double curCot01 = cotTheta(curEdgeBA, curEdgeBC);
                const double curCot02 = cotTheta(curEdgeBA, curEdgeBD);
                const double curCot03 = cotTheta(-curEdgeBA, curEdgeCA);
                const double curCot04 = cotTheta(-curEdgeBA, curEdgeDA);
                const Eigen::Matrix<double, 3, 1> t0 = -cot03 * curEdgeBC -cot01 * curEdgeCA;
                const Eigen::Matrix<double, 3, 1> t1 = -cot04 * curEdgeBD -cot02 * curEdgeDA;
                const double e012 = (curEdgeBA.cross(curEdgeBC)).dot(curEdgeBD);

                // ---  bending energy ---
                // thin plate component
                const double thinPlateEnergy = bendingRigidity * 3/restStencilArea * (std::pow(restLenEdgeBA, 2) + t0.dot(t1)*cosThetaBar); 
                // cubic term
                const double cubicEnergy = bendingRigidity * 3 * beta / restStencilArea * e012 * sinThetaBar;
                bendingEnergy += thinPlateEnergy + cubicEnergy;

                // --- gradient of bending energy ---
                // thin plate component
                Eigen::Matrix<double, 12, 1> thinPlateForce = Qbar * x_vec;
                // cubic term
                Eigen::Matrix<double, 3, 1> f2 = kbar * curEdgeBC.cross(curEdgeBD);
                Eigen::Matrix<double, 3, 1> f3 = kbar * curEdgeBD.cross(curEdgeBA);
                Eigen::Matrix<double, 3, 1> f4 = kbar * curEdgeBA.cross(curEdgeBC);
                Eigen::Matrix<double, 3, 1> f1 = -f2 - f3 - f4;
                Eigen::Matrix<double, 12, 1> cubicForce = Eigen::Matrix<double, 12, 1>::Zero();
                cubicForce.block<3, 1>(0, 0) = f1; cubicForce.block<3, 1>(3, 0) = f2; cubicForce.block<3, 1>(6, 0) = f3; cubicForce.block<3, 1>(9, 0) = f4;
                for (int j = 0; j < 4; j++)
                    globalBendingForces.segment<3>(3 * elemVInd[j]) += thinPlateForce.segment<3>(3 * j) + cubicForce.segment<3>(3 * j);
                
                // --- hessian of bending energy ---
                // thin plate component
                Eigen::Matrix<double, 12, 12> thinPlateStiffMatrix = Qbar;

                // // cubic term is not considered
                // Eigen::Matrix<double, 12, 12> cubicStiffMatrix = Eigen::Matrix<double, 12, 12>::Zero();
                // cubicStiffMatrix.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Zero();
                // cubicStiffMatrix.block<3, 3>(0, 3) = kbar * crossProductMatrix(curEdgeBC - curEdgeBD); // e1 - e2
                // cubicStiffMatrix.block<3, 3>(0, 6) = kbar * crossProductMatrix(curEdgeBD - curEdgeBA); // e2 - e0
                // cubicStiffMatrix.block<3, 3>(0, 9) = kbar * crossProductMatrix(curEdgeBA - curEdgeBC); // e0 - e1
                // cubicStiffMatrix.block<3, 3>(3, 0) = - kbar * crossProductMatrix(curEdgeBD - curEdgeBC); // e2 - e1
                // cubicStiffMatrix.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Zero();
                // cubicStiffMatrix.block<3, 3>(3, 6) = - kbar * crossProductMatrix(curEdgeBD); // - e2
                // cubicStiffMatrix.block<3, 3>(3, 9) = kbar * crossProductMatrix(curEdgeBC); // e1
                // cubicStiffMatrix.block<3, 3>(6, 0) = - kbar * crossProductMatrix(curEdgeBD - curEdgeBA); // e2 - e0
                // cubicStiffMatrix.block<3, 3>(6, 3) = kbar * crossProductMatrix(curEdgeBD); // e2
                // cubicStiffMatrix.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Zero();
                // cubicStiffMatrix.block<3, 3>(6, 9) = - kbar * crossProductMatrix(curEdgeBA); // - e0
                // cubicStiffMatrix.block<3, 3>(9, 0) = - kbar * crossProductMatrix(curEdgeBA - curEdgeBC); // e0 - e1
                // cubicStiffMatrix.block<3, 3>(9, 3) = - kbar * crossProductMatrix(curEdgeBC); // - e1
                // cubicStiffMatrix.block<3, 3>(9, 6) = kbar * crossProductMatrix(curEdgeBA); // e0
                // cubicStiffMatrix.block<3, 3>(9, 9) = Eigen::Matrix<double, 3, 3>::Zero();

                // Hessian of bending energy
                // neglect the cubic term
                Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = thinPlateStiffMatrix;
                // if (isLocalProj) // ToDO: check if the hessian is always positive definite
                //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);
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