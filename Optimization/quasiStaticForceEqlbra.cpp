#include  <igl/boundary_loop.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include "ThinShellSolver.h"
#include "../Common/Timer.h"

const Eigen::VectorXd posMat2Vector(const ElasticSetup& setup, const ElasticState& curState) 
{
    int nverts = curState.curPos.rows();
	int nedges = curState.mesh.nEdges();

	int nedgedofs = setup.sff->numExtraDOFs();
	int extraDOFs = nedgedofs * nedges;

	Eigen::VectorXd fullX(3 * nverts + extraDOFs);

	for (int i = 0; i < nverts; i++)
	{
		fullX.segment(3 * i, 3) = curState.curPos.row(i);
	}

	fullX.segment(3 * nverts, nedgedofs * nedges) = curState.curEdgeDOFs;
    
    return fullX;
}

void ThinShellSolver::quasiStaticNewtonSolver(const ElasticSetup& setup, ElasticState& curState, std::string filePrefix, const FullSimOptimizationParams& params)
{
    const int maxNewIter = params.iterations;
    const double absTol = 0.0;
    const double relTol = 1e-3;
    const double LSstepSize = 0.1;

    const auto restState = curState;
    ElasticShellModel model;
	bool ok = model.initialization(setup, curState, filePrefix, params.isProjH, params.isParallel);
	if (!ok)
    {
        std::cout << "initialization failed." << std::endl;
        return;
    }
    Eigen::VectorXd initX; 
	model.convertCurState2Variables(curState, initX);

    int nverts = restState.curPos.rows();
	int nedges = restState.mesh.nEdges();
	int nedgedofs = setup.sff->numExtraDOFs();
	int extraDOFs = nedgedofs * nedges;
    int dofs_ = 3 * nverts + extraDOFs;
    
    Eigen::VectorXd u = Eigen::VectorXd::Zero(dofs_ - setup.clampedDOFs.size());

    Eigen::VectorXd exterForces;
    model.externalForces(initX, exterForces);

    // Precompute bending Hessian if not using "midEdgeShell" bending
    Eigen::SparseMatrix<double> bendingHess;
    if (setup.bendingType != "midEdgeShell") {
        bendingHess = model.bendingHessian(initX);
    }
    bool convergence = false;
    for (int i = 0; i < params.iterations; i++)
    {
        Eigen::VectorXd grad;
        if (setup.bendingType == "midEdgeShell") {
            model.gradient(initX + u, grad);
        } else {
            grad = model.membraneGrad(initX + u) + model.bendingGrad(initX + u) + model.externalForces(initX + u);
        }
        Eigen::VectorXd rhs_bc = - grad;

        const double rhs_norm = rhs_bc.norm();  
        const double exterF_norm = exterForces.norm();
        std::cout << "abs_err(rhs_norm): " << rhs_norm << ", rel_err: " << rhs_norm/exterF_norm << " at " << i+1 << " iteration. " 
              << "tolerance using relTol*extFnorm+absTol: " << relTol * exterF_norm + absTol << std::endl;

        if (rhs_norm <= relTol * exterF_norm + absTol)
        {
            convergence = true;
        }

        
        Eigen::SparseMatrix<double> hess;
        if (setup.bendingType == "midEdgeShell") {
            model.hessian(initX + u, hess);
        } 
        else {
            hess = model.membraneHessian(initX + u) + bendingHess + model.exterHessian(initX + u);
        }

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(hess);
        Eigen::VectorXd du = solver.solve(rhs_bc);

        const double du_infiNorm = du.cwiseAbs().maxCoeff(); 
        if (du_infiNorm >= LSstepSize)
        {
            du = LSstepSize / du_infiNorm * du;
        }

        u = u + du;

        if (convergence == true || i == params.iterations - 1)
        {
            model.convertVariables2CurState(initX+u, curState);
            igl::writeOBJ(setup.outMeshPath, curState.curPos, curState.mesh.faces());
            std::cout << "Convergenced! Total iteration number is: " << i << std::endl;
            break;
        }
    } // end of for loop
    model.convertVariables2CurState(initX+u, curState); // visualization
}