#include "ElasticEnergy.h"
#include "../Common/CommonFunctions.h"
#include <tbb/tbb.h>
#include <iostream>

const Eigen::Matrix<double, 2, 6> frameTransformation(const Eigen::Matrix<double, 3, 6>& eleCurrentPos) 
{
    Eigen::Matrix<double, 2, 3> local_exyT;
    local_exyT << 1, 0, 0,
                  0, 1, 0; 

    Eigen::Matrix<double, 3, 6> matX1;
    for(int i = 0; i < 6; i++)
        matX1.col(i) = eleCurrentPos.col(0); 

    return local_exyT * (eleCurrentPos - matX1);
}

const Eigen::Matrix<double, 3, 6> curviGradOpQuadraFit(const Eigen::Matrix<double, 2, 6>& localXY) 
{
    Eigen::Matrix<double, 6, 6> coeffMat;
    //          1,     x_i            y_i            x_i^2 / 2,                      y_i^2 / 2,                     x_i * y_i/2
    coeffMat << 1, localXY(0, 0), localXY(1, 0), localXY(0, 0)*localXY(0, 0)/2, localXY(1, 0)*localXY(1, 0)/2, localXY(0, 0)*localXY(1, 0)/2,
                1, localXY(0, 1), localXY(1, 1), localXY(0, 1)*localXY(0, 1)/2, localXY(1, 1)*localXY(1, 1)/2, localXY(0, 1)*localXY(1, 1)/2,
                1, localXY(0, 2), localXY(1, 2), localXY(0, 2)*localXY(0, 2)/2, localXY(1, 2)*localXY(1, 2)/2, localXY(0, 2)*localXY(1, 2)/2,
                1, localXY(0, 3), localXY(1, 3), localXY(0, 3)*localXY(0, 3)/2, localXY(1, 3)*localXY(1, 3)/2, localXY(0, 3)*localXY(1, 3)/2,
                1, localXY(0, 4), localXY(1, 4), localXY(0, 4)*localXY(0, 4)/2, localXY(1, 4)*localXY(1, 4)/2, localXY(0, 4)*localXY(1, 4)/2,
                1, localXY(0, 5), localXY(1, 5), localXY(0, 5)*localXY(0, 5)/2, localXY(1, 5)*localXY(1, 5)/2, localXY(0, 5)*localXY(1, 5)/2;

    const Eigen::Matrix<double, 3, 6> curviGradOp = coeffMat.inverse().block<3, 6>(3, 0);

    return curviGradOp;
}

const Eigen::Matrix<double, 3, 6> curviGradOpFlatSmoothedHinge(const Eigen::Matrix<double, 2, 6>& localXY) 
{
    const Eigen::Vector2d X1 = localXY.block<2, 1>(0, 0);
    const Eigen::Vector2d X2 = localXY.block<2, 1>(0, 1);
    const Eigen::Vector2d X3 = localXY.block<2, 1>(0, 2);
    const Eigen::Vector2d X4 = localXY.block<2, 1>(0, 3);
    const Eigen::Vector2d X5 = localXY.block<2, 1>(0, 4);
    const Eigen::Vector2d X6 = localXY.block<2, 1>(0, 5);

    const double AreaT432 = computeArea(X4, X3, X2);
    const double AreaT513 = computeArea(X5, X1, X3);
    const double AreaT621 = computeArea(X6, X2, X1);
    const double AreaT123 = computeArea(X1, X2, X3);

    const double lenEdge23 = (X3 - X2).norm();
    const double lenEdge31 = (X1 - X3).norm();
    const double lenEdge12 = (X2 - X1).norm();
    
    const double h4edge23 = 2 * AreaT432 / lenEdge23;
    const double h1edge23 = 2 * AreaT123 / lenEdge23;

    const double h5edge31 = 2 * AreaT513 / lenEdge31;
    const double h2edge31 = 2 * AreaT123 / lenEdge31;

    const double h6edge12 = 2 * AreaT621 / lenEdge12;
    const double h3edge12 = 2 * AreaT123 / lenEdge12;

    const auto edge31 = X1 - X3;
    const auto edge32 = X2 - X3;
    const auto edge34 = X4 - X3;
    const double P3 = edge31.dot(edge32) / lenEdge23;
    const double P2 = lenEdge23 - P3;
    const double Q3 = edge34.dot(edge32) / lenEdge23;
    const double Q2 = lenEdge23 - Q3;

    const auto edge13 = X3 - X1;
    const auto edge15 = X5 - X1;
    const auto edge12 = X2 - X1;
    const double M1 = edge15.dot(edge13) / lenEdge31;
    const double M3 = lenEdge31 - M1;
    const double N1 = edge12.dot(edge13) / lenEdge31;
    const double N3 = lenEdge31 - N1;

    const auto edge26 = X6 - X2;
    const auto edge21 = X1 - X2;
    const auto edge23 = X3 - X2;
    const double R2 = edge26.dot(edge21) / lenEdge12;
    const double R1 = lenEdge12 - R2;
    const double S2 = edge23.dot(edge21) / lenEdge12;
    const double S1 = lenEdge12 - S2;

    Eigen::Matrix<double, 6, 3> Cxy;
    Cxy <<  localXY(0, 0)*localXY(0, 0)/2, localXY(1, 0)*localXY(1, 0)/2, localXY(0, 0)*localXY(1, 0)/2,
            localXY(0, 1)*localXY(0, 1)/2, localXY(1, 1)*localXY(1, 1)/2, localXY(0, 1)*localXY(1, 1)/2,
            localXY(0, 2)*localXY(0, 2)/2, localXY(1, 2)*localXY(1, 2)/2, localXY(0, 2)*localXY(1, 2)/2,
            localXY(0, 3)*localXY(0, 3)/2, localXY(1, 3)*localXY(1, 3)/2, localXY(0, 3)*localXY(1, 3)/2,
            localXY(0, 4)*localXY(0, 4)/2, localXY(1, 4)*localXY(1, 4)/2, localXY(0, 4)*localXY(1, 4)/2,
            localXY(0, 5)*localXY(0, 5)/2, localXY(1, 5)*localXY(1, 5)/2, localXY(0, 5)*localXY(1, 5)/2;

    Eigen::Matrix<double, 3, 6> CL;
       
    CL <<  1/h1edge23, -(Q3/lenEdge23*(1/h4edge23) + P3/lenEdge23*(1/h1edge23)), -(Q2/lenEdge23*(1/h4edge23) + P2/lenEdge23*(1/h1edge23)), 1/h4edge23,          0,          0, 
            -(M3/lenEdge31*(1/h5edge31) + N3/lenEdge31*(1/h2edge31)),                                               1/h2edge31, -(M1/lenEdge31*(1/h5edge31) + N1/lenEdge31*(1/h2edge31)),          0, 1/h5edge31,          0,
            -(R2/lenEdge12*(1/h6edge12) + S2/lenEdge12*(1/h3edge12)), -(R1/lenEdge12*(1/h6edge12) + S1/lenEdge12*(1/h3edge12)),                                               1/h3edge12,          0,          0, 1/h6edge12;

    CL.row(0) = CL.row(0) * (2/(h1edge23 + h4edge23));
    CL.row(1) = CL.row(1) * (2/(h2edge31 + h5edge31));
    CL.row(2) = CL.row(2) * (2/(h3edge12 + h6edge12));
    const auto A = CL * Cxy;

    const Eigen::Matrix<double, 3, 6> curviGradOp = A.inverse() * CL;
    return curviGradOp;
}


const Eigen::Matrix<double, 3, 6> curviGradSmoothedHinge(const Eigen::MatrixXd& iniPos, const MeshConnectivity &mesh, const Eigen::Matrix<double, 2, 6>& localXY, int face, Eigen::Matrix<double, 3, 3>& bendingRigidityMat, bool& boundaryElem)
{
    Eigen::Matrix<double, 3, 6> curviGradOp = curviGradOpFlatSmoothedHinge(localXY);
    for (int ei = 0; ei < 3; ei++) 
    {
        int oppVID = mesh.vertexOppositeFaceEdge(face, ei);
        if (oppVID == -1) // if this is a boundary edge, then
        {
            boundaryElem = true;
            int edgeIdx = mesh.faceEdge(face, ei);
            int edgeNodeIdx1 = mesh.edgeVertex(edgeIdx, 0);
            int edgeNodeIdx2 = mesh.edgeVertex(edgeIdx, 1);
            Eigen::Matrix<double, 3, 1> edgeNode1 = iniPos.row(edgeNodeIdx1).transpose();
            Eigen::Matrix<double, 3, 1> edgeNode2 = iniPos.row(edgeNodeIdx2).transpose();
            double xCoord1 = edgeNode1(0); double xCoord2 = edgeNode2(0);
            bool clampedCondition = false;
            if (clampedCondition) // if this is a clamped boundary edge
            {
                curviGradOp.col(ei) = curviGradOp.col(ei) + curviGradOp.col(ei+3);
                curviGradOp.col(ei+3) = Eigen::Matrix<double, 3, 1>::Zero();
            }
            else // if this is a free boundary edge, then
            {
                bendingRigidityMat(2, 2) = 0.0;
                // NM is boundary, B is virtual node of A
                const int idxM = (ei + 2) % 3;
                const int idxN = (ei + 1) % 3;
                const auto xM = localXY.col(idxM);
                const auto xA = localXY.col(ei);
                const auto xN = localXY.col(idxN);
                const auto edgeNA = xA - xN;
                const auto edgeNM = xM - xN;
                const double sp = 0.5;
                curviGradOp.col(idxM) = curviGradOp.col(idxM) + 2 * sp * curviGradOp.col(ei + 3);
                curviGradOp.col(ei) = curviGradOp.col(ei) - curviGradOp.col(ei + 3);
                curviGradOp.col(idxN) = curviGradOp.col(idxN) + 2 * (1-sp) * curviGradOp.col(ei + 3);
                curviGradOp.col(ei + 3) = Eigen::Matrix<double, 3, 1>::Zero();
            }
        }
    }

    return curviGradOp;
}


double flatSmoothedHingeBendingEnergy(
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
    bendingStiffK.resize(nfaces * 18 * 18); 
    const double planeStressCoef = YoungsModulus / (1 - std::pow(PoissonsRatio, 2));
    Eigen::Matrix<double, 3, 3> constitutiveMat; 
    constitutiveMat << 1, PoissonsRatio, 0,
                        PoissonsRatio, 1, 0,
                        0, 0, 0.5*(1-PoissonsRatio);
    Eigen::Matrix<double, 3, 3> bendingRigidityMat = std::pow(thickness, 3) / 12.0 * planeStressCoef * constitutiveMat;
    if (isParallel)
    {
        auto computeBending = [&](const tbb::blocked_range<uint32_t>& range)
        {
            for (uint32_t face = range.begin(); face < range.end(); ++face)
            {
                Eigen::Matrix<int, 1, 6> elemVInd = Eigen::Matrix<int, 1, 6>::Constant(-1);
                for (int i = 0; i < 3; i++)
                {
                    elemVInd(i) = mesh.faceVertex(face, i);
                    elemVInd(i + 3) = mesh.vertexOppositeFaceEdge(face, i);
                }
                double triArea = 0.5 * sqrt(abars[face].determinant());
                Eigen::Vector3d X1 = iniPos.row(elemVInd(0)).transpose();
                Eigen::Vector3d X2 = iniPos.row(elemVInd(1)).transpose();
                Eigen::Vector3d X3 = iniPos.row(elemVInd(2)).transpose();
                Eigen::Vector3d X4 = elemVInd(3) != -1 ? iniPos.row(elemVInd(3)).transpose().eval() : (X2 + X3 - X1);
                Eigen::Vector3d X5 = elemVInd(4) != -1 ? iniPos.row(elemVInd(4)).transpose().eval() : (X3 + X1 - X2);
                Eigen::Vector3d X6 = elemVInd(5) != -1 ? iniPos.row(elemVInd(5)).transpose().eval() : (X1 + X2 - X3);
                Eigen::Matrix<double, 3, 6> elePatchRestPos = Eigen::Matrix<double, 3, 6>::Zero();
                elePatchRestPos << X1, X2, X3, X4, X5, X6;
                const Eigen::Matrix<double, 3, 6> localXYZ = corotationalFrame(X1, X2, X3, elePatchRestPos);
                const Eigen::Matrix<double, 2, 6> localXY = localXYZ.block<2, 6>(0, 0);
                bool boundaryElem = false;
                Eigen::Matrix<double, 3, 3> newBendingRigidityMat = bendingRigidityMat;
                Eigen::Matrix<double, 3, 6> curviGradOp = curviGradSmoothedHinge(iniPos, mesh, localXY, face, newBendingRigidityMat, boundaryElem);

                Eigen::MatrixXd bendingRigidityMat_flatten;
                bendingRigidityMat_flatten.resize(9, 9); bendingRigidityMat_flatten.setZero();
                KeroneckerProduct(newBendingRigidityMat, Eigen::Matrix<double, 3, 3>::Identity(3, 3), bendingRigidityMat_flatten); 
                Eigen::MatrixXd pAssmCurpUT;
                pAssmCurpUT.resize(9, 18); pAssmCurpUT.setZero();
                KeroneckerProduct(curviGradOp, Eigen::Matrix<double, 3, 3>::Identity(), pAssmCurpUT);

                Eigen::Matrix<double, 18, 18> elementalBendingStiffMatrix = triArea * pAssmCurpUT.transpose() * bendingRigidityMat_flatten * pAssmCurpUT;
                // if (isLocalProj) 
                //     elementalBendingStiffMatrix = lowRankApprox(elementalBendingStiffMatrix);
                const int offset = face * 6 * 3 * 6 * 3;
                for (int k = 0; k < 6; ++k)
                    for (int d = 0; d < 3; ++d)
                        for (int s = 0; s < 6; ++s)
                            for (int t = 0; t < 3; ++t){
                                if (elemVInd(k) != -1 && elemVInd(s) != -1) 
                                {
                                    bendingStiffK[offset
                                    + k * 3 * 6 * 3
                                    + d * 6 * 3
                                    + s * 3
                                    + t] = Eigen::Triplet<double>(3 * elemVInd(k) + d, 3 * elemVInd(s) + t, elementalBendingStiffMatrix(s * 3 + t, k * 3 + d));
                                }
                            }

            } 
        };

        tbb::blocked_range<uint32_t> rangex(0u, (uint32_t)nfaces);
        tbb::parallel_for(rangex, computeBending);
    }
    else // serial version
    {
        for (int face = 0; face < nfaces; ++face) 
        {
            Eigen::Matrix<int, 1, 6> elemVInd = Eigen::Matrix<int, 1, 6>::Constant(-1);
            for (int i = 0; i < 3; i++)
            {
                elemVInd(i) = mesh.faceVertex(face, i);
                elemVInd(i + 3) = mesh.vertexOppositeFaceEdge(face, i);
            }
            double triArea = 0.5 * sqrt(abars[face].determinant());
            Eigen::Vector3d X1 = iniPos.row(elemVInd(0)).transpose();
            Eigen::Vector3d X2 = iniPos.row(elemVInd(1)).transpose();
            Eigen::Vector3d X3 = iniPos.row(elemVInd(2)).transpose();
            Eigen::Vector3d X4 = elemVInd(3) != -1 ? iniPos.row(elemVInd(3)).transpose().eval() : (X2 + X3 - X1);
            Eigen::Vector3d X5 = elemVInd(4) != -1 ? iniPos.row(elemVInd(4)).transpose().eval() : (X3 + X1 - X2);
            Eigen::Vector3d X6 = elemVInd(5) != -1 ? iniPos.row(elemVInd(5)).transpose().eval() : (X1 + X2 - X3);
            Eigen::Matrix<double, 3, 6> elePatchRestPos = Eigen::Matrix<double, 3, 6>::Zero();
            elePatchRestPos << X1, X2, X3, X4, X5, X6;
            const Eigen::Matrix<double, 3, 6> localXYZ = corotationalFrame(X1, X2, X3, elePatchRestPos);
            const Eigen::Matrix<double, 2, 6> localXY = localXYZ.block<2, 6>(0, 0);
            bool boundaryElem = false;
            Eigen::Matrix<double, 3, 3> newBendingRigidityMat = bendingRigidityMat;
            Eigen::Matrix<double, 3, 6> curviGradOp = curviGradSmoothedHinge(iniPos, mesh, localXY, face, newBendingRigidityMat, boundaryElem);
            
            Eigen::MatrixXd bendingRigidityMat_flatten;
            bendingRigidityMat_flatten.resize(9, 9); bendingRigidityMat_flatten.setZero();
            KeroneckerProduct(newBendingRigidityMat, Eigen::Matrix<double, 3, 3>::Identity(3, 3), bendingRigidityMat_flatten); 
            Eigen::MatrixXd pAssmCurpUT;
            pAssmCurpUT.resize(9, 18); pAssmCurpUT.setZero();
            KeroneckerProduct(curviGradOp, Eigen::Matrix<double, 3, 3>::Identity(), pAssmCurpUT);

            Eigen::Matrix<double, 18, 18> elementalBendingStiffMatrix = triArea * pAssmCurpUT.transpose() * bendingRigidityMat_flatten * pAssmCurpUT;
            // if (isLocalProj) 
            //     elementalBendingStiffMatrix = lowRankApprox(elementalBendingStiffMatrix);
            const int offset = face * 6 * 3 * 6 * 3;
            for (int k = 0; k < 6; ++k)
                for (int d = 0; d < 3; ++d)
                    for (int s = 0; s < 6; ++s)
                        for (int t = 0; t < 3; ++t){
                            if (elemVInd(k) != -1 && elemVInd(s) != -1) 
                            {
                                bendingStiffK[offset
                                + k * 3 * 6 * 3
                                + d * 6 * 3
                                + s * 3
                                + t] = Eigen::Triplet<double>(3 * elemVInd(k) + d, 3 * elemVInd(s) + t, elementalBendingStiffMatrix(s * 3 + t, k * 3 + d));
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
        (*hessian).resize(nfaces * 18 * 18);
        (*hessian) = bendingStiffK;
    }

    return bendingEnergy;
}


