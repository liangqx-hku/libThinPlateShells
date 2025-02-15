#include  <igl/boundary_loop.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include "ThinShellSolver.h"
#include "NewtonDescent.h"

void ThinShellSolver::fullSimNewtonStaticSolver(const ElasticSetup& setup, ElasticState& curState, std::string filePrefix, const FullSimOptimizationParams& params)
{
	const auto restState = curState;
	ElasticShellModel model;
	bool ok = model.initialization(setup, curState, filePrefix, params.isProjH, params.isParallel);
	if (!ok)
    {
        std::cout << "initialization failed." << std::endl;
        return;
    }

	Eigen::VectorXd initX; 
	Eigen::MatrixXi F;
	model.convertCurState2Variables(curState, initX);
	double energy = model.value(initX);

	Eigen::VectorXd grad;
	model.gradient(initX, grad);

	if(params.printLog)
	{
		std::cout << "started energy: " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << energy << ", ||g|| = " << grad.template lpNorm<Eigen::Infinity>() << std::endl;
	}

	if(grad.norm() < params.gradNorm)
	{
		if(params.printLog)
			std::cout << "init gradient norm is small" << std::endl;
		return;
	}

	Eigen::SparseMatrix<double> bendingHess;
	std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, Eigen::SparseMatrix<double>*, bool)> elasticFunc;

	if (setup.bendingType == "EP" || setup.bendingType == "ES" 
		|| setup.bendingType == "QS" || setup.bendingType == "CS"
		|| setup.bendingType == "FP" || setup.bendingType == "SS" 
		|| setup.bendingType == "SP" || setup.bendingType == "FS")
	{
		bendingHess = model.bendingHessian(initX);
		elasticFunc = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
		{
			model._isUsePosHess = isProj;
			double energy = model.value(x);

			if(deriv)
				*deriv = model.membraneGrad(x) + model.bendingGrad(x) + model.externalForces(x);
			if (hess)
				*hess = model.membraneHessian(x) + bendingHess + model.exterHessian(x);
			return energy;
		};
	}
	else
	{
		elasticFunc = [&](const Eigen::VectorXd& x, Eigen::VectorXd* deriv, Eigen::SparseMatrix<double>* hess, bool isProj)
		{
			model._isUsePosHess = isProj;
			double energy = model.value(x);

			if(deriv)
				model.gradient(x, *deriv);
			if (hess)
				model.hessian(x, *hess);
			return energy;
		};
	}

    OptSolver::newtonSolver(elasticFunc, initX, params.iterations, params.gradNorm, params.xDelta, params.fDelta, params.printLog);
    model.convertVariables2CurState(initX, curState);
	igl::writeOBJ(setup.outMeshPath, curState.curPos, curState.mesh.faces());
}