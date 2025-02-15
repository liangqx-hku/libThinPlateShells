#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

// EP
double corotationalHingeBendingEnergy(
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

    std::vector<Eigen::Triplet<double>> bendingStiffK;
    bendingStiffK.clear();
    bendingStiffK.resize(nedges * 12 * 12); 
    const double bendingRigidity = std::pow(thickness, 3) / 12.0 * YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
    if (isParallel) // parallel version
    {
        auto eleVInds = std::vector<Eigen::Vector4i>(nedges);
        auto hesses = std::vector<Eigen::Matrix<double, 12, 12>>(nedges);
        auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t i = range.begin(); i < range.end(); ++i) 
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
                    Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
                    Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
                    Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
                    Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
                    Eigen::Matrix<double, 3, 4> elePatchRestPos;
                    elePatchRestPos << X1, X2, X3, X4;
                    Eigen::Matrix<double, 3, 1> XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                    Eigen::Matrix<double, 3, 1> XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                    const Eigen::Matrix<double, 3, 1> p0 = X3 - XP, q0 = X4 - XQ;
                    Eigen::Matrix<double, 3, 1> n0;
                    Eigen::Matrix<double, 3, 1> e0 = X2 - X1, e1 = X3 - X1, e2 = X4 - X1;
                    if (e0.cross(e1).dot(e2) ==0)
                        n0 = (X2 - X1).cross(X3 - X1).normalized();
                    else
                        n0 = (p0/p0.norm()+q0/q0.norm()).normalized();

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
                    
                    // hessian of bending energy
                    eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
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
            if (adjFace0!=-1 && adjFace1!=-1) // if not a boundary
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

                Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
                Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
                Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
                Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
                Eigen::Matrix<double, 3, 4> elePatchRestPos;
                elePatchRestPos << X1, X2, X3, X4;
                Eigen::Matrix<double, 3, 1> XP = (X2 - X1) * (X3 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                Eigen::Matrix<double, 3, 1> XQ = (X2 - X1) * (X4 - X1).dot(X2 - X1) / (X2 - X1).squaredNorm() + X1;
                const Eigen::Matrix<double, 3, 1> p0 = X3 - XP, q0 = X4 - XQ;
                Eigen::Matrix<double, 3, 1> n0;
                Eigen::Matrix<double, 3, 1> e0 = X2 - X1, e1 = X3 - X1, e2 = X4 - X1;
                if (e0.cross(e1).dot(e2) ==0)
                    n0 = (X2 - X1).cross(X3 - X1).normalized();
                else
                    n0 = (p0/p0.norm()+q0/q0.norm()).normalized();

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

                // hessian of bending energy
                Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;

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


// // QTP
// #include "ElasticEnergy.h"
// #include "../Common/CommonFunctions.h"
// #include <tbb/tbb.h>
// #include <iostream>


// double corotationalHingeBendingEnergy(
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
//         (*hessian).resize(nedges * 12 * 12);
//     }

//     double bendingEnergy = 0.0;
//     std::vector<Eigen::Triplet<double>> bendingStiffK;
//     bendingStiffK.clear();
//     bendingStiffK.resize(nedges * 12 * 12); 
//     const double bendingRigidity = std::pow(thickness, 3) / 12.0 * YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
//     if (isParallel) // parallel version
//     {
//         auto eleVInds = std::vector<Eigen::Vector4i>(nedges);
//         auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
//         {
//             for (uint32_t i = range.begin(); i < range.end(); ++i) 
//             {
//                 Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = Eigen::Matrix<double, 12, 12>::Zero();
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
                    
//                     // hessian of bending energy
//                     eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
//                     // if (isLocalProj) 
//                     //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);
                    
//                     // current configuration
//                     const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
//                     const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
//                     const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
//                     const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
//                     Eigen::Matrix<double, 12, 1> x_e = Eigen::Matrix<double, 12, 1>::Zero();
//                     x_e.block<3, 1>(0, 0) = x1; x_e.block<3, 1>(3, 0) = x2; x_e.block<3, 1>(6, 0) = x3; x_e.block<3, 1>(9, 0) = x4;
//                     Eigen::Matrix<double, 12, 1> eleBendingForce = eleBendingStiffMatrix  * x_e;

//                     // bending energy
//                     bendingEnergy += 0.5 * x_e.transpose() * eleBendingForce;

//                     if(derivative){
//                         for (int k = 0; k < 4; ++k)
//                             (*derivative).segment<3>(3 * elemVInd[k]) += eleBendingForce.segment<3>(3 * k);
//                     }

//                     if(hessian){
//                         const int offset = i * 4 * 3 * 4 * 3;
//                         for (int k = 0; k < 4; ++k)
//                             for (int d = 0; d < 3; ++d)
//                                 for (int s = 0; s < 4; ++s)
//                                     for (int t = 0; t < 3; ++t){
//                                         (*hessian)[ offset + 
//                                         + k * 3 * 4 * 3
//                                         + d * 4 * 3
//                                         + s * 3
//                                         + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, eleBendingStiffMatrix(s * 3 + t, k * 3 + d)); 
//                                     }
//                     }
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
//             Eigen::Matrix<double, 12, 12> eleBendingStiffMatrix = Eigen::Matrix<double, 12, 12>::Zero();
//             int adjFace0 = mesh.edgeFace(i, 0);
//             int adjFace1 = mesh.edgeFace(i, 1);
//             Eigen::Vector4i elemVInd;
//             elemVInd(0) = mesh.edgeVertex(i, 0);
//             elemVInd(1) = mesh.edgeVertex(i, 1);
//             elemVInd(2) = mesh.edgeOppositeVertex(i, 0);
//             elemVInd(3) = mesh.edgeOppositeVertex(i, 1);
//             if (adjFace0!=-1 && adjFace1!=-1) 
//             {
//                 const Eigen::Matrix<double, 3, 1> X1 = iniPos.row(elemVInd(0)).transpose();
//                 const Eigen::Matrix<double, 3, 1> X2 = iniPos.row(elemVInd(1)).transpose();
//                 const Eigen::Matrix<double, 3, 1> X3 = iniPos.row(elemVInd(2)).transpose();
//                 const Eigen::Matrix<double, 3, 1> X4 = iniPos.row(elemVInd(3)).transpose();
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
//                 Eigen::Matrix<double, 1, 4> curvaGradOp = Eigen::Matrix<double, 1, 4>::Zero();
//                 curvaGradOp << L1, L2, L3, L4;
                
//                 const double restStencilArea = 0.5 * restLenEdge12 * (restHeight1 + restHeight2); 
                
//                 // hessian of bending energy
//                 eleBendingStiffMatrix = restStencilArea * bendingRigidity *4/(restHeight1 + restHeight2)/(restHeight1 + restHeight2)* L.transpose() * L;
//                 // if (isLocalProj) 
//                 //     eleBendingStiffMatrix = lowRankApprox(eleBendingStiffMatrix);
                
//                 const Eigen::Matrix<double, 3, 1> x1 = curPos.row(elemVInd(0)).transpose();
//                 const Eigen::Matrix<double, 3, 1> x2 = curPos.row(elemVInd(1)).transpose();
//                 const Eigen::Matrix<double, 3, 1> x3 = curPos.row(elemVInd(2)).transpose();
//                 const Eigen::Matrix<double, 3, 1> x4 = curPos.row(elemVInd(3)).transpose();
//                 Eigen::Matrix<double, 12, 1> x_e = Eigen::Matrix<double, 12, 1>::Zero();
//                 x_e.block<3, 1>(0, 0) = x1; x_e.block<3, 1>(3, 0) = x2; x_e.block<3, 1>(6, 0) = x3; x_e.block<3, 1>(9, 0) = x4;
//                 Eigen::Matrix<double, 3, 4> elePatchCurPos;
//                 elePatchCurPos << x1, x2, x3, x4;
//                 Eigen::Matrix<double, 12, 1> eleBendingForce = eleBendingStiffMatrix  * x_e;

//                 // bending energy
//                 bendingEnergy += 0.5 * x_e.transpose() * eleBendingForce;

//                 if(derivative){
//                     for (int k = 0; k < 4; ++k)
//                         (*derivative).segment<3>(3 * elemVInd[k]) += eleBendingForce.segment<3>(3 * k);
//                 }

//                 if(hessian){
//                     const int offset = i * 4 * 3 * 4 * 3;
//                     for (int k = 0; k < 4; ++k)
//                         for (int d = 0; d < 3; ++d)
//                             for (int s = 0; s < 4; ++s)
//                                 for (int t = 0; t < 3; ++t){
//                                     (*hessian)[ offset + 
//                                     + k * 3 * 4 * 3
//                                     + d * 4 * 3
//                                     + s * 3
//                                     + t] = Eigen::Triplet<double>(3 * elemVInd[k] + d, 3 * elemVInd[s] + t, eleBendingStiffMatrix(s * 3 + t, k * 3 + d)); 
//                                 }
//                 }
//             }
//         } 
//     }

//     return bendingEnergy;
// }