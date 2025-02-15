#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

// Qudratic shell
double quadraticBendingEnergy(
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

    std::vector<Eigen::Triplet<double>> bendingStiffK;
    bendingStiffK.clear();
    bendingStiffK.resize(nedges * 12 * 12); 
    const double bendingRigidity = std::pow(thickness, 3) / 12.0 * YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
    if (false) // parallel version should be improved for stability for this model
    {
        auto eleVInds = std::vector<Eigen::Vector4i>(nedges);
        auto hesses = std::vector<Eigen::Matrix<double, 12, 12>>(nedges);
        auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i) // loop over all the edges
            {
                Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = Eigen::Matrix<double, 12, 12>::Zero();
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
                    const double restStencilArea = 0.5 * restLenEdgeBA * (restH1 + restH2); 
                    const double kbar = 3 * bendingRigidity * (cot01 - cot03) * (cot04 - cot02) * restE012 / (restStencilArea * std::pow(restLenEdgeBA, 3));
                    Eigen::Matrix<double, 1, 4> L = Eigen::Matrix<double, 1, 4>::Zero();
                    L << L1, L2, L3, L4; 
                    Eigen::Matrix<double, 3, 12> L_flatten = Eigen::Matrix<double, 3, 12>::Zero();
                    L_flatten.block<3, 3>(0, 0) = L1 * Eigen::Matrix<double, 3, 3>::Identity();
                    L_flatten.block<3, 3>(0, 3) = L2 * Eigen::Matrix<double, 3, 3>::Identity();
                    L_flatten.block<3, 3>(0, 6) = L3 * Eigen::Matrix<double, 3, 3>::Identity();
                    L_flatten.block<3, 3>(0, 9) = L4 * Eigen::Matrix<double, 3, 3>::Identity();
                    eleBendingStiffMatrix = 3 * bendingRigidity * cosThetaBar / restStencilArea * L_flatten.transpose() * L_flatten;
                }
                hesses[i] = eleBendingStiffMatrix;
            } 
        }; 
        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nedges);
        tbb::parallel_for(rangex, computeBending);

        for (int i = 0; i < nedges; i++)
        {
            Eigen::Vector4i elemVInd = eleVInds[i];
            int adjFace0 = mesh.edgeFace(i, 0);
            int adjFace1 = mesh.edgeFace(i, 1);
            if (adjFace0!=-1 && adjFace1!=-1) 
            {
                const int offset = i * 4 * 3 * 4 * 3;
                for (int k = 0; k < 4; ++k)
                    for (int d = 0; d < 3; ++d)
                        for (int s = 0; s < 4; ++s)
                            for (int t = 0; t < 3; ++t){
                                bendingStiffK[ offset + 
                                + k * 3 * 4 * 3
                                + d * 4 * 3
                                + s * 3
                                + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, hesses[i](s * 3 + t, k * 3 + d)); 
                            }
            }
        }
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
                const double restStencilArea = 0.5 * restLenEdgeBA * (restH1 + restH2); 
                const double kbar = 3 * bendingRigidity * (cot01 - cot03) * (cot04 - cot02) * restE012 / (restStencilArea * std::pow(restLenEdgeBA, 3));
                Eigen::Matrix<double, 1, 4> L = Eigen::Matrix<double, 1, 4>::Zero();
                L << L1, L2, L3, L4; 
                Eigen::Matrix<double, 3, 12> L_flatten = Eigen::Matrix<double, 3, 12>::Zero();
                L_flatten.block<3, 3>(0, 0) = L1 * Eigen::Matrix<double, 3, 3>::Identity();
                L_flatten.block<3, 3>(0, 3) = L2 * Eigen::Matrix<double, 3, 3>::Identity();
                L_flatten.block<3, 3>(0, 6) = L3 * Eigen::Matrix<double, 3, 3>::Identity();
                L_flatten.block<3, 3>(0, 9) = L4 * Eigen::Matrix<double, 3, 3>::Identity();
                Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = 3 * bendingRigidity * cosThetaBar / restStencilArea * L_flatten.transpose() * L_flatten;
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

    Eigen::SparseMatrix<double> globalBendSparseStiffMat(3 * nverts, 3 * nverts);
    globalBendSparseStiffMat.setFromTriplets(bendingStiffK.begin(), bendingStiffK.end());

    Eigen::VectorXd gradBendE(3 * nverts);
    gradBendE = globalBendSparseStiffMat * disp;

    const double bendingEnergy = 0.5 * disp.dot(gradBendE);
    
    if(derivative){
        (*derivative).segment(0, 3 * nverts) = gradBendE;
    }

    if(hessian){
        (*hessian).resize(nedges * 12 * 12);
        (*hessian) = bendingStiffK;
    }

    return bendingEnergy;
}