#include <Eigen/SPQRSupport>
#include <igl/per_vertex_normals.h>
#include <iostream>
#include <filesystem>
#include "CommonFunctions.h"
#include "../MeshLib/MeshGeometry.h"
#include "../MeshLib/IntrinsicGeometry.h"
#include "../MeshLib/MeshConnectivity.h"

#ifdef __APPLE__
namespace fs = std::__fs::filesystem;
#else
namespace fs = std::filesystem;
#endif

Eigen::MatrixXd lowRankApprox(Eigen::MatrixXd A)
{
    Eigen::MatrixXd posHess = A;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(posHess);
    Eigen::VectorXd evals = es.eigenvalues();

    for (int i = 0; i < evals.size(); i++)
    {
        if (evals(i) < 0)
			evals(i) = 0;
    }
    Eigen::MatrixXd D = evals.asDiagonal();
    Eigen::MatrixXd V = es.eigenvectors();
    posHess = V * D * V.inverse();

    return posHess;
}

Eigen::MatrixXd lowRankApproxBend(Eigen::MatrixXd A, bool& flag)
{
	
    Eigen::MatrixXd posHess = A;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(posHess);
    Eigen::VectorXd evals = es.eigenvalues();

    for (int i = 0; i < evals.size(); i++)
    {
        if (evals(i) < 0)
		{
			evals(i) = 0;
			flag = true;
		}
    }
	if (flag)
		std::cout << "negative eigenvalue detected, set eigenvalue to 0" << std::endl;

    Eigen::MatrixXd D = evals.asDiagonal();
    Eigen::MatrixXd V = es.eigenvectors();
    posHess = V * D * V.inverse();

    return posHess;
}

Eigen::MatrixXd nullspaceExtraction(const Eigen::SparseMatrix<double> A)
{
	Eigen::SPQR<Eigen::SparseMatrix<double>> solver_SPQR(A.transpose());
	int r = solver_SPQR.rank();

	Eigen::MatrixXd N(A.cols(), A.cols() - r);
	for (int i = 0; i < A.cols() - r; i++)
	{
		Eigen::VectorXd ei(A.cols());
		ei.setZero();
		ei(i + r) = 1;
		N.col(i) = solver_SPQR.matrixQ() * ei;
	}
	std::cout << "rank of A = " << r << std::endl;
	return N;
}

double cotan(const Eigen::Vector3d v0, const Eigen::Vector3d v1, const Eigen::Vector3d v2)
{
    double e0 = (v2 - v1).norm();
    double e1 = (v2 - v0).norm();
    double e2 = (v0 - v1).norm();
    double angle0 = acos((e1 * e1 + e2 * e2 - e0 * e0) / (2 * e1 * e2));
    double cot = 1.0 / tan(angle0);

    return cot;
}


void locatePotentialPureTensionFaces(const std::vector<Eigen::Matrix2d>& abars, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, std::set<int>& potentialPureTensionFaces)
{
	MeshConnectivity mesh(F);
	MeshGeometry curGeo(V, mesh);
	IntrinsicGeometry restGeo(mesh, abars);
	assert(abars.size() == F.rows());
	int nfaces = F.rows();
	int nverts = V.rows();

	Eigen::VectorXd smallestEvals(nfaces);

	std::vector<bool> isPureFaces(nfaces, false);
	std::vector<bool> isPureVerts(nverts, false);

	for (int i = 0; i < nfaces; i++)
	{
		Eigen::Matrix2d abar = abars[i];
		Eigen::Matrix2d a = curGeo.Bs[i].transpose() * curGeo.Bs[i];
		Eigen::Matrix2d diff = a - abar;
		Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix2d> solver(diff, abar);

		smallestEvals(i) = std::min(solver.eigenvalues()[0], solver.eigenvalues()[1]);
	}


	for (int i = 0; i < nfaces; i++)
	{
		if (smallestEvals(i) >= 0)
		{
			isPureFaces[i] = true;
		}
	}

	for (int i = 0; i < nfaces; i++)
	{
		if (isPureFaces[i])
		{
			for (int j = 0; j < 3; j++)
			{
				isPureVerts[mesh.faceVertex(i, j)] = true;
			}
		}
	}

	//// if a face is not pure tension, but all its vertices are on the other pure faces, we think it is pure tension
	//// this is used to fill the mixed state "holes" in the pure tension region
	for (int i = 0; i < nfaces; i++)
	{
		if (!isPureFaces[i])
		{
			bool isAllPureVerts = true;
			for (int j = 0; j < 3; j++)
			{
				int vid = mesh.faceVertex(i, j);
				if (isPureVerts[vid] == false)
					isAllPureVerts = false;
			}
			isPureFaces[i] = isAllPureVerts;
		}
	}

	for (int i = 0; i < nfaces; i++)
	{
		if (isPureFaces[i])
			potentialPureTensionFaces.insert(i);
	}
	// set this to empty means we never relax the integrability constriant.
	// potentialPureTensionFaces.clear();
}

void getPureTensionVertsEdges(const std::set<int>& potentialPureTensionFaces, const Eigen::MatrixXi& F, std::set<int>* pureTensionEdges, std::set<int>* pureTensionVerts)
{
	MeshConnectivity mesh(F);

	std::set<int> tmpPureTensionEdges, tmpPureTensionVerts;
	
	std::set<int> mixedStateEdges;
	// locate the mixed edges: 
	for (int i = 0; i < mesh.nEdges(); i++)
	{
		Eigen::Vector2i faceFlags;
		faceFlags << 0, 0;
		for (int j = 0; j < 2; j++)
		{
			int fid = mesh.edgeFace(i, j);
			if (potentialPureTensionFaces.find(fid) != potentialPureTensionFaces.end())
			{
				faceFlags(j) = 1;
			}
		}
		if (faceFlags(0) + faceFlags(1) == 1)
		{
			if (mixedStateEdges.find(i) == mixedStateEdges.end())
			{
				mixedStateEdges.insert(i);
			}
		}

		else if (faceFlags(0) + faceFlags(1) == 2)
		{
			if (tmpPureTensionEdges.find(i) == tmpPureTensionEdges.end())
			{
				tmpPureTensionEdges.insert(i);
			}
		}
	}
	// mixedStateEdges.clear();
	// tmpPureTensionEdges.clear();
	std::cout << mixedStateEdges.size() << " edges between compressed and pure tension faces. " << tmpPureTensionEdges.size() << " edges are inside the pure tension region" << std::endl;

	for (auto& fid : potentialPureTensionFaces)
	{
		for (int j = 0; j < 3; j++)
		{
			int vid = mesh.faceVertex(fid, j);
			if (tmpPureTensionVerts.find(vid) == tmpPureTensionVerts.end())
			{
				tmpPureTensionVerts.insert(vid);
			}
		}
	}
	// tmpPureTensionVerts.clear();
	std::cout << tmpPureTensionVerts.size() << " vertices are inside the pure tension region (including the boundary)" << std::endl;

	if (pureTensionEdges)
	{
		*pureTensionEdges = tmpPureTensionEdges;
	}
	if (pureTensionVerts)
	{
		*pureTensionVerts = tmpPureTensionVerts;
	}
}


void trivialOffset(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd &offsettedV, double d)
{
	Eigen::MatrixXd VN;
	igl::per_vertex_normals(V, F, VN);
	offsettedV = V  + d * VN;
}

void rigidBodyAlignment(const Eigen::MatrixXd& tarPos, const MeshConnectivity& mesh, const Eigen::MatrixXd &pos, Eigen::Matrix3d &R, Eigen::Vector3d &t)
{
	int nverts = tarPos.rows();
    int nfaces = mesh.nFaces();
    Eigen::VectorXd massVec;
    massVec.resize(nverts);
    massVec.setZero();
    
    Eigen::VectorXd areaList;
    igl::doublearea(tarPos, mesh.faces(), areaList);
    areaList = areaList / 2;
    
    for(int i=0; i < nfaces; i++)
    {
        double faceArea = areaList(i);
        for(int j=0; j<3; j++)
        {
            int vertIdx = mesh.faceVertex(i, j);
            massVec(vertIdx) += faceArea / 3;
        }
    }
    
    massVec = massVec / 3;
    massVec = massVec / massVec.maxCoeff();
    
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nverts, nverts);
    W.diagonal() = massVec;
    
    Eigen::Vector3d avePos, aveTarPos;
    avePos.setZero();
    aveTarPos.setZero();
    
    for(int i = 0; i < nverts; i++)
    {
        avePos += massVec(i) * pos.row(i);
        aveTarPos += massVec(i) * tarPos.row(i);
    }
    
    avePos = avePos / massVec.sum();
    aveTarPos = aveTarPos / massVec.sum();
    
    Eigen::MatrixXd onesMat(nverts,1);
    onesMat.setOnes();
    
    Eigen::MatrixXd shiftedPos = pos - onesMat * avePos.transpose();
    Eigen::MatrixXd shiftedTarPos = tarPos - onesMat * aveTarPos.transpose();
    
    Eigen::MatrixXd S = shiftedPos.transpose() * W * shiftedTarPos;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3);
    
    
    Eigen::BDCSVD<Eigen::MatrixXd> solver(S);
    solver.compute(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = solver.matrixU();
    Eigen::MatrixXd V = solver.matrixV();
    
    Eigen::MatrixXd middleMat(S.rows(), S.cols());
    middleMat.setIdentity();
    middleMat(S.rows()-1,S.cols()-1) = (V*U.transpose()).determinant();
    
    R = V * middleMat * U.transpose();
    t = aveTarPos - R*avePos;
}

void matToVec(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec)
{
	int nverts = mat.rows();
	vec.resize(3 * nverts);
	for (int i = 0; i < nverts; i++)
	{
		for (int j = 0; j < 3; j++)
			vec(3 * i + j) = mat(i, j);
	}
}

void vecToMat(const Eigen::VectorXd& vec, Eigen::MatrixXd& mat)
{
	int nverts = vec.size() / 3;
	mat.resize(nverts, 3);
	for (int i = 0; i < nverts; i++)
	{
		for (int j = 0; j < 3; j++)
			mat(i, j) = vec(3 * i + j);
	}
}

bool mkdir(const std::string& foldername)
{
	if (!fs::exists(foldername))
	{
		std::cout << "create directory: " << foldername << std::endl;
		if (!fs::create_directory(foldername))
		{
			std::cerr << "create folder failed." << foldername << std::endl;
			return false;
		}
	}
	return true;
}

const Eigen::Matrix<double, 3, 3> crossProductMatrix(const Eigen::Matrix<double, 3, 1>& v)
{
    Eigen::Matrix<double, 3, 3> m;
    m << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return m;
}

const double cotTheta(const Eigen::Matrix<double, 3, 1>& n1, const Eigen::Matrix<double, 3, 1>& n2)
{
  const double cosTheta = n1.dot(n2);
  const double sinTheta = (n1.cross(n2)).norm();
  return (cosTheta / sinTheta);
}

void KeroneckerProduct(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C)
{
    const int m = A.rows();
    const int n = A.cols();
    const int p = B.rows();
    const int q = B.cols();
    C.resize(m * p, n * q);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C.block(i * p, j * q, p, q) = A(i, j) * B;
        }
    }
}

void compute_normal_gradient(const Eigen::Matrix<double, 3, 1>& node4, const Eigen::Matrix<double, 3, 1>& node5, const Eigen::Matrix<double, 3, 1>& node6,
                             const Eigen::Matrix<double, 3, 1>& nodalU4, const Eigen::Matrix<double, 3, 1>& nodalU5, const Eigen::Matrix<double, 3, 1>& nodalU6,
                             Eigen::Matrix<double, 3, 18>& grad_n) 
{
    const double x4 = node4(0); const double y4 = node4(1); const double z4 = node4(2);
    const double x5 = node5(0); const double y5 = node5(1); const double z5 = node5(2);
    const double x6 = node6(0); const double y6 = node6(1); const double z6 = node6(2);
    const double u4 = nodalU4(0); const double v4 = nodalU4(1); const double w4 = nodalU4(2);
    const double u5 = nodalU5(0); const double v5 = nodalU5(1); const double w5 = nodalU5(2);
    const double u6 = nodalU6(0); const double v6 = nodalU6(1); const double w6 = nodalU6(2);

    const double nx = (y5+v5-y4-v4)*(z6+w6-z4-w4)-(z5+w5-z4-w4)*(y6+v6-y4-v4);
    const double ny = -((x5+u5-x4-u4)*(z6+w6-z4-w4)-(z5+w5-z4-w4)*(x6+u6-x4-u4));
    const double nz = (x5+u5-x4-u4)*(y6+v6-y4-v4)-(y5+v5-y4-v4)*(x6+u6-x4-u4);
    const double len_n = sqrt(std::pow(nx, 2) + std::pow(ny, 2) + std::pow(nz, 2));
    
    double len_n_u4 = -(ny*((z6+w6-z4-w4)-(z5+w5-z4-w4)) +nz*(-(y6+v6-y4-v4)+(y5+v5-y4-v4)))/ std::pow(len_n, 3);
    double len_n_v4 = -(nx*(-(z6+w6-z4-w4)+(z5+w5-z4-w4))+nz*(-(x5+u5-x4-u4)+(x6+u6-x4-u4)))/ std::pow(len_n, 3);
    double len_n_w4 = -(nx*(-(y5+v5-y4-v4)+(y6+v6-y4-v4))+ny*((x5+u5-x4-u4)-(x6+u6-x4-u4))) / std::pow(len_n, 3);
    double len_n_u5 = -(-ny*(z6+w6-z4-w4)+nz*(y6+v6-y4-v4))/std::pow(len_n, 3);
    double len_n_v5 = -( nx*(z6+w6-z4-w4)-nz*(x6+u6-x4-u4))/std::pow(len_n, 3);
    double len_n_w5 = -(-nx*(y6+v6-y4-v4)+ny*(x6+u6-x4-u4))/std::pow(len_n, 3);
    double len_n_u6 = -( ny*(z5+w5-z4-w4)-nz*(y5+v5-y4-v4))/std::pow(len_n, 3);
    double len_n_v6 = -(-nx*(z5+w5-z4-w4)+nz*(x5+u5-x4-u4))/std::pow(len_n, 3);
    double len_n_w6 = -( nx*(y5+v5-y4-v4)-ny*(x5+u5-x4-u4))/std::pow(len_n, 3);

    grad_n = Eigen::Matrix<double, 3, 18>::Zero();
    const double nx_u4 = nx*len_n_u4;
    const double nx_v4 = (-(z6+w6-z4-w4)+(z5+w5-z4-w4))/len_n + nx*len_n_v4;
    const double nx_w4 = (-(y5+v5-y4-v4)+(y6+v6-y4-v4))/len_n + nx*len_n_w4;
    const double nx_u5 = nx*len_n_u5;
    const double nx_v5 = (z6+w6-z4-w4) /len_n+nx*len_n_v5;
    const double nx_w5 = -(y6+v6-y4-v4)/len_n+nx*len_n_w5;
    const double nx_u6 = nx*len_n_u6;
    const double nx_v6 = -(z5+w5-z4-w4)/len_n+nx*len_n_v6;
    const double nx_w6 = (y5+v5-y4-v4) /len_n+nx*len_n_w6;
    const double ny_u4 = ((z6+w6-z4-w4)-(z5+w5-z4-w4))/len_n+ny*len_n_u4;
    const double ny_v4 = ny*len_n_v4;
    const double ny_w4 = ((x5+u5-x4-u4)-(x6+u6-x4-u4))/len_n+ny*len_n_w4;
    const double ny_u5 = -(z6+w6-z4-w4)/len_n+ny*len_n_u5;
    const double ny_v5 = ny*len_n_v5;
    const double ny_w5 = (x6+u6-x4-u4)/len_n+ny*len_n_w5;
    const double ny_u6 = (z5+w5-z4-w4)/len_n+ny*len_n_u6;
    const double ny_v6 = ny*len_n_v6;
    const double ny_w6 = -(x5+u5-x4-u4)/len_n+ny*len_n_w6;
    const double nz_u4 = (-(y6+v6-y4-v4)+(y5+v5-y4-v4))/len_n+nz*len_n_u4;
    const double nz_v4 = (-(x5+u5-x4-u4)+(x6+u6-x4-u4))/len_n+nz*len_n_v4;
    const double nz_w4 = nz*len_n_w4;
    const double nz_u5 = (y6+v6-y4-v4) /len_n+nz*len_n_u5;
    const double nz_v5 = -(x6+u6-x4-u4)/len_n+nz*len_n_v5;
    const double nz_w5 = nz*len_n_w5;
    const double nz_u6 = -(y5+v5-y4-v4)/len_n+nz*len_n_u6;
    const double nz_v6 = (x5+u5-x4-u4) /len_n+nz*len_n_v6;
    const double nz_w6 = nz*len_n_w6;

    grad_n.block<3, 9>(0, 0) << nx_u4, nx_v4, nx_w4, nx_u5, nx_v5, nx_w5, nx_u6, nx_v6, nx_w6,
                                ny_u4, ny_v4, ny_w4, ny_u5, ny_v5, ny_w5, ny_u6, ny_v6, ny_w6,
                                nz_u4, nz_v4, nz_w4, nz_u5, nz_v5, nz_w5, nz_u6, nz_v6, nz_w6;

}

void curviGradOpCurveFvmHinge(const MeshConnectivity &mesh, const Eigen::Matrix<double, 3, 6>& localXYZ, 
                              Eigen::Matrix<double, 3, 3>& bendingRigidityMat, bool& boundaryElem, int face,
                              Eigen::Matrix<double, 3, 3>& TransMat, Eigen::Matrix<double, 3, 6>& curviGradOp) 
{
    Eigen::Matrix<int, 3, 4> hingeVInds;
    hingeVInds << 2, 1, 3, 0,
                  0, 2, 4, 1,
                  1, 0, 5, 2;
    for (int edgeID = 0; edgeID < 3; ++edgeID)
    {
       
        Eigen::RowVector4i hingeVInd = hingeVInds.row(edgeID);
        Eigen::Matrix<double, 3, 4> hingeRestPos;
        hingeRestPos << localXYZ.col(hingeVInd(0)), localXYZ.col(hingeVInd(1)), localXYZ.col(hingeVInd(2)), localXYZ.col(hingeVInd(3));
        Eigen::Matrix<double, 3, 1> XA = hingeRestPos.col(0), XB = hingeRestPos.col(1), XC = hingeRestPos.col(2), XD = hingeRestPos.col(3);
        Eigen::Matrix<double, 3, 1> iniEdgeAB = XB - XA, iniEdgeAC = XC - XA, iniEdgeAD = XD - XA;
        const double restLenAB = iniEdgeAB.norm();
        const Eigen::Matrix<double, 3, 1> dirRestAB = iniEdgeAB / restLenAB; 
        const double PA = iniEdgeAC.dot(iniEdgeAB) / restLenAB;
        const double PB = restLenAB - PA;
        const double PC = (PA*dirRestAB - iniEdgeAC).norm(); 
        const double QA = iniEdgeAD.dot(iniEdgeAB) / restLenAB;
        const double QB = restLenAB - QA;
        const double QD = (QA*dirRestAB - iniEdgeAD).norm(); 
        Eigen::Matrix<double, 3, 1> m = (QA*dirRestAB - iniEdgeAD) / QD;
        const double cosPhi = m.x();
        const double sinPhi = m.y();
        TransMat.col(edgeID) << pow(cosPhi, 2), 
                                pow(sinPhi, 2), 
                                2 * cosPhi*sinPhi;
        Eigen::Matrix<double, 1, 6> L = Eigen::Matrix<double, 1, 6>::Zero();
        const double factor = 2 / (PC + QD);
        const double LA = -(PB/PC + QB/QD)/restLenAB;
        const double LB = -(PA/PC + QA/QD)/restLenAB;
        const double LC =  1.0/PC;
        const double LD =  1.0/QD;
        L(hingeVInd(0)) = factor*LA; L(hingeVInd(1)) = factor*LB; L(hingeVInd(2)) = factor*LC; L(hingeVInd(3)) = factor*LD;
        curviGradOp.row(edgeID) = L;
        // boundary conditions
        int oppVID = mesh.vertexOppositeFaceEdge(face, edgeID);
        if (oppVID == -1) // if this is a boundary edge, then
        {
            boundaryElem = true;
            curviGradOp.row(edgeID).setZero();
            bendingRigidityMat(2, 2) = 0.0;
        }
    }
}

double computeArea(const Eigen::Vector2d& X0, const Eigen::Vector2d& X1, const Eigen::Vector2d& X2)
{
    double A = std::fabs(0.5 * (X0[0]*(X1[1] - X2[1]) + X1[0]*(X2[1] - X0[1]) + X2[0]*(X0[1] - X1[1])));
    return A;
}

const Eigen::Matrix<double, 3, 6> corotationalFrame(const Eigen::Matrix<double, 3, 1>& X1, const Eigen::Matrix<double, 3, 1>& X2, const Eigen::Matrix<double, 3, 1>& X3,
                                                    const Eigen::Matrix<double, 3, 6>& elePatchRestPos) 
{
    Eigen::Matrix<double, 3, 1> normal0 = ((X2 - X1).cross(X3 - X1)).normalized();
    Eigen::Matrix<double, 3, 1> local_x_axis = (X3 - X1).normalized();
    Eigen::Matrix<double, 3, 1> local_y_axis = normal0.cross(local_x_axis);
    Eigen::Matrix<double, 3, 3> local_exyzT;
    local_exyzT << local_x_axis.transpose(),
                    local_y_axis.transpose(),
                    normal0.transpose();
    Eigen::Matrix<double, 3, 6> matX1;
    matX1 << X1, X1, X1, X1, X1, X1;
    const Eigen::Matrix<double, 3, 6> localXYZ = local_exyzT * (elePatchRestPos - matX1);

    return localXYZ;
}

void curviGradTriStencilOperator(const MeshConnectivity &mesh, const Eigen::Matrix<double, 3, 6>& localXYZ,
                                 Eigen::Matrix<double, 3, 3>& bendingRigidityMat, bool& boundaryElem, int face,
                                 Eigen::Matrix<double, 3, 3>& TransMat, Eigen::Matrix<double, 3, 6>& curviGradOp)
{
    // check if this is a boundary element
    for (int edgeID = 0; edgeID < 3; ++edgeID)
    {
        int oppVID = mesh.vertexOppositeFaceEdge(face, edgeID);
        if (oppVID == -1) // if this is a boundary edge, then
        {
            boundaryElem = true;
        }
    }
    
    const Eigen::Vector2d X1 = localXYZ.block<2, 1>(0, 0);
    const Eigen::Vector2d X2 = localXYZ.block<2, 1>(0, 1);
    const Eigen::Vector2d X3 = localXYZ.block<2, 1>(0, 2);
    const Eigen::Vector2d X4 = localXYZ.block<2, 1>(0, 3);
    const Eigen::Vector2d X5 = localXYZ.block<2, 1>(0, 4);
    const Eigen::Vector2d X6 = localXYZ.block<2, 1>(0, 5);

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
    Cxy <<  localXYZ(0, 0)*localXYZ(0, 0)/2, localXYZ(1, 0)*localXYZ(1, 0)/2, localXYZ(0, 0)*localXYZ(1, 0)/2,
            localXYZ(0, 1)*localXYZ(0, 1)/2, localXYZ(1, 1)*localXYZ(1, 1)/2, localXYZ(0, 1)*localXYZ(1, 1)/2,
            localXYZ(0, 2)*localXYZ(0, 2)/2, localXYZ(1, 2)*localXYZ(1, 2)/2, localXYZ(0, 2)*localXYZ(1, 2)/2,
            localXYZ(0, 3)*localXYZ(0, 3)/2, localXYZ(1, 3)*localXYZ(1, 3)/2, localXYZ(0, 3)*localXYZ(1, 3)/2,
            localXYZ(0, 4)*localXYZ(0, 4)/2, localXYZ(1, 4)*localXYZ(1, 4)/2, localXYZ(0, 4)*localXYZ(1, 4)/2,
            localXYZ(0, 5)*localXYZ(0, 5)/2, localXYZ(1, 5)*localXYZ(1, 5)/2, localXYZ(0, 5)*localXYZ(1, 5)/2;

    Eigen::Matrix<double, 3, 6> CL;
    CL <<  1/h1edge23, -(Q3/lenEdge23*(1/h4edge23) + P3/lenEdge23*(1/h1edge23)), -(Q2/lenEdge23*(1/h4edge23) + P2/lenEdge23*(1/h1edge23)), 1/h4edge23,          0,          0, 
            -(M3/lenEdge31*(1/h5edge31) + N3/lenEdge31*(1/h2edge31)),                                               1/h2edge31, -(M1/lenEdge31*(1/h5edge31) + N1/lenEdge31*(1/h2edge31)),          0, 1/h5edge31,          0,
            -(R2/lenEdge12*(1/h6edge12) + S2/lenEdge12*(1/h3edge12)), -(R1/lenEdge12*(1/h6edge12) + S1/lenEdge12*(1/h3edge12)),                                               1/h3edge12,          0,          0, 1/h6edge12;
    CL.row(0) = CL.row(0) * (2/(h1edge23 + h4edge23));
    CL.row(1) = CL.row(1) * (2/(h2edge31 + h5edge31));
    CL.row(2) = CL.row(2) * (2/(h3edge12 + h6edge12));

    const auto A = CL * Cxy;

    curviGradOp = A.inverse() * CL;

	for (int edgeID = 0; edgeID < 3; ++edgeID)
	{
		int oppVID = mesh.vertexOppositeFaceEdge(face, edgeID);
		if (oppVID == -1) // if this is a boundary edge, then
		{
			const int idxM = (edgeID + 2) % 3;
			const int idxN = (edgeID + 1) % 3;
			bendingRigidityMat(2, 2) = 0.0;
			double sp = 0.5;
			curviGradOp.col(idxM) = curviGradOp.col(idxM) + 2 * sp * curviGradOp.col(edgeID + 3);
			curviGradOp.col(edgeID) = curviGradOp.col(edgeID) - curviGradOp.col(edgeID + 3);
			curviGradOp.col(idxN) = curviGradOp.col(idxN) + 2 * (1-sp) * curviGradOp.col(edgeID + 3);
			curviGradOp.col(edgeID + 3) = Eigen::Matrix<double, 3, 1>::Zero();
		}
	}
    
}



