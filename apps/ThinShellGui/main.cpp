#include <igl/writePLY.h>
#include <igl/hausdorff.h>
#include "../../MeshLib/MeshConnectivity.h"
#include <iostream>
#include <random>
#include <igl/signed_distance.h>
#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/boundary_loop.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>

#include <CLI/CLI.hpp>

#include "polyscope/polyscope.h"
#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/view.h"

#include <Eigen/CholmodSupport>

#include "../../Optimization/ThinShellSolver.h"
#include "../../Collision/Obstacle.h"

#include "../../ThinShells/ElasticEnergy.h"

#include "../../Common/CommonFunctions.h"


#include "../../ThinShells/ElasticIO.h"


static StretchingType stretchingType;
int bendingType;

ElasticSetup setup;
ElasticState curState;
int numSteps;
FullSimOptimizationParams fullSimOptParams;

double thickness;
double pressure;
double PoissonRatio;
double YoungsModulus;
double materialDensity;
double x_externalForce;
double y_externalForce;
double z_externalForce;
int frameFreq;
double noiseNorm;
double penaltyK;
double maxStepSize;
double innerEta;
int iterations;

bool isVisualizeOriginalMesh; // visualize the origin mesh
bool isVisualizeObstacle; // visualize the obstacle
bool restFlat; // plot tension line
double scale;   // plot scale
bool isSPDProj;

std::string filePathPrefix;
std::string defaultPath;
std::string elasticPath;

float surfShiftx = 0;
float surfShifty = 0;
float surfShiftz = 0;

float preShiftx = 0;
float preShifty = 0;
float preShiftz = 0;

float boundx = 0;
float boundy = 0;
float boundz = 0;

SFFType sfftype = MidedgeAverage;


void liftVerts(double mag) // lifing the verts in direction z
{
	for (int i = 0; i < curState.curPos.rows(); i++)
		curState.curPos(i, 2) += mag;
}

void jitter(double magnitude)
{
	assert(magnitude >= 0);
	if (magnitude == 0)
		return;
	std::cout << "Noise magnitude: " << magnitude << std::endl;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0, magnitude);

	Eigen::MatrixXd VN;

	igl::per_vertex_normals(curState.curPos, curState.mesh.faces(), VN);

	for (int i = 0; i < curState.curPos.rows(); i++)
	{
		if (setup.clampedDOFs.find(3 * i) == setup.clampedDOFs.end())
		{
			double offsetNoise = distribution(generator);
			curState.curPos.row(i) += offsetNoise * VN.row(i);
		}
	}
}

void setParameters()
{
    thickness = setup.thickness;
    materialDensity = setup.density;
    YoungsModulus = setup.YoungsModulus;
    PoissonRatio = setup.PoissonsRatio;
    numSteps = setup.numInterp;
    x_externalForce = setup.gravity(0);
    y_externalForce = setup.gravity(1);
    z_externalForce = setup.gravity(2);
    scale = 1;
    if(setup.strecthingType == "NeoHookean")
        stretchingType = NeoHookean;
    else if (setup.strecthingType == "tensionField")
        stretchingType = tensionField;
    else
        stretchingType = StVK;

    iterations = 2000;
    //set to false for default instead
    bendingType = mideEdgeBending;
    if (setup.bendingType == "QS")
        bendingType = quadraticBending;
    else if (setup.bendingType == "CS")
        bendingType = cubicShell;
    else if (setup.bendingType == "EP")
        bendingType = corotationalHinge;
    else if (setup.bendingType == "ES")
        bendingType = corotationalCurveHinge;
    else if (setup.bendingType == "FP")
        bendingType = corotationalFlatFvmHinge;
    else if (setup.bendingType == "FS")
        bendingType = corotationalCurveFvmHinge;
    else if (setup.bendingType == "SP")
        bendingType = flatSmoothedHinge;
    else if (setup.bendingType == "SS")
        bendingType = curveSmoothedHinge;
    else
        bendingType = noBending;
    noiseNorm = 0.0;
    penaltyK = setup.penaltyK;

    if (setup.sffType == "midedgeAve")
        sfftype = MidedgeAverage;
    else if (setup.sffType == "midedgeSin")
        sfftype = MidedgeSin;
    else
        sfftype = MidedgeTan;

    isSPDProj = setup.penaltyK > 0 ? false : true;

    pressure = setup.pressure;
    restFlat = setup.restFlat;
    innerEta = setup.innerEta;
    maxStepSize = setup.maxStepSize;

    isVisualizeOriginalMesh = true;
    isVisualizeObstacle = false;
}

bool loadProblem(std::string loadPath = "")
{
    if(loadPath == "")
        defaultPath = igl::file_dialog_open();
    else
        defaultPath = loadPath;
    
    int index = defaultPath.rfind("_");
    filePathPrefix = defaultPath.substr(0, index);
    elasticPath = defaultPath;
    std::cout << "Model Name is: " << filePathPrefix << std::endl;
    bool ok = loadElastic(defaultPath, setup, curState);
    
    if(ok)
    {
        setParameters();
        boundx = curState.initialGuess.col(0).maxCoeff() - curState.initialGuess.col(0).minCoeff();
        boundy = curState.initialGuess.col(1).maxCoeff() - curState.initialGuess.col(1).minCoeff();
        boundz = curState.initialGuess.col(2).maxCoeff() - curState.initialGuess.col(2).minCoeff();
    }
        
    return ok;
}

bool saveProblem(std::string savePath = "")
{
    if (savePath == "")
    {
        savePath = igl::file_dialog_save();
    }
    bool ok = saveElastic(savePath, setup, curState);
    return ok;
}

void updateView(bool isFirstTime = true)
{
    polyscope::registerSurfaceMesh("initial wrinkled mesh", curState.initialGuess, curState.mesh.faces());
    polyscope::getSurfaceMesh("initial wrinkled mesh")->setSurfaceColor({ 80 / 255.0, 122 / 255.0, 91 / 255.0 });

    polyscope::registerSurfaceMesh("current wrinkled mesh", curState.curPos, curState.mesh.faces());
    polyscope::getSurfaceMesh("current wrinkled mesh")->setSurfaceColor({ 122 / 255.0, 80 / 255.0, 91 / 255.0 });
}


void callback() {
    ImGui::PushItemWidth(100);
    float w = ImGui::GetContentRegionAvailWidth();
    float p = ImGui::GetStyle().FramePadding.x;
    if (ImGui::Button("Load", ImVec2((w - p) / 2.f, 0)))
    {
        loadProblem();
    }
    ImGui::SameLine(0, p);
    if (ImGui::Button("Save", ImVec2((w - p) / 2.f, 0)))
    {
        saveProblem();
    }


    if (ImGui::CollapsingHeader("Material Parameters", ImGuiTreeNodeFlags_DefaultOpen))
    {
        float w = ImGui::GetContentRegionAvailWidth();
        float p = ImGui::GetStyle().FramePadding.x;
        if (ImGui::InputDouble("Thickness", &thickness))
        {
            if (thickness > 0)
            {
                setup.thickness = thickness;
            }
        }
        if (ImGui::InputDouble("Density", &materialDensity))
        {
            if (materialDensity > 0)
            {
                setup.density = materialDensity;
            }
        }
        if (ImGui::InputDouble("Young's Modulus", &YoungsModulus))
        {
            if (YoungsModulus > 0)
            {
                setup.YoungsModulus = YoungsModulus;
            }
        }
        if (ImGui::InputDouble("Poisson's Ratio", &PoissonRatio))
        {
            if (PoissonRatio >= 0)
            {
                setup.PoissonsRatio = PoissonRatio;
            }
        }
    }

    

    if (ImGui::CollapsingHeader("Elastic Energy Type", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Combo("Stretching", (int*)&stretchingType, "StVK\0tensionField\0NoeHookean\0\0"))
        {
            switch (stretchingType)
            {
                case StVK:
                    setup.strecthingType = "StVK";
                    break;
                case tensionField:
                    setup.strecthingType = "tensionField";
                    break;
                case NeoHookean:
                    setup.strecthingType = "NeoHookean";
                    break;
                default:
                    setup.strecthingType = "StVK";
                    break;
            }
        }

    
        if (ImGui::Combo("Bending", (int*)&bendingType, "MidEdge\0QuadraticShell\0CubicShell\0CorotationalHinge\0CorotationalCurveHinge\0CorotationalFlatFvmHinge\0CorotationalCurveFvmHinge\0FlatSmoothedHinge\0CurveSmoothedHinge\0NoBending\0"))
        {
            switch (bendingType)
            {
                case mideEdgeBending:
                    setup.bendingType = "midEdgeShell";
                    break;
                case quadraticBending:
                    setup.bendingType = "QS";
                    break;
                case cubicShell:
                    setup.bendingType = "CS";
                    break;
                case corotationalHinge:
                    setup.bendingType = "EP";
                    break;
                case corotationalCurveHinge:
                    setup.bendingType = "ES";
                    break;
                case corotationalFlatFvmHinge:
                    setup.bendingType = "FP";
                    break;
                case corotationalCurveFvmHinge:
                    setup.bendingType = "FS";
                    break;
                case flatSmoothedHinge:
                    setup.bendingType = "SP";
                    break;
                case curveSmoothedHinge:
                    setup.bendingType = "SS";
                    break;
                case noBending:
                    setup.bendingType = "nobending";
                    break;
                default:
                    setup.bendingType = "midEdgeShell";
                    break;
            }
        }
        if (ImGui::Combo("SFF type", (int*)&sfftype, "MidedgeAve\0MidedgeSin\0MidedgeTan\0"))
        {
            switch (sfftype)
            {
            case MidedgeAverage:
                setup.sff = std::make_shared<MidedgeAverageFormulation>();
                break;
            case MidedgeSin:
                setup.sff = std::make_shared<MidedgeAngleSinFormulation>();
                break;
            case MidedgeTan:
                setup.sff = std::make_shared<MidedgeAngleTanFormulation>();
                break;
            default:
                setup.sff = std::make_shared<MidedgeAverageFormulation>();
                break;
            }
            setup.buildRestFundamentalForms();
            setup.sff->initializeExtraDOFs(curState.initialEdgeDOFs, curState.mesh, curState.initialGuess);
            curState.curEdgeDOFs = curState.initialEdgeDOFs;
        }

        if (ImGui::Button("Optimize ThinShell", ImVec2(-1, 0)))
        {
            jitter(noiseNorm);
            for (int k = 1; k <= numSteps; k++)
            {
                // ThinShellSolver::fullSimNewtonStaticSolver(setup, curState, filePathPrefix, fullSimOptParams);
                ThinShellSolver::quasiStaticNewtonSolver(setup, curState, filePathPrefix, fullSimOptParams);
                // ThinShellSolver::linearPlateBending(setup, curState, filePathPrefix, fullSimOptParams);
            }
            updateView();
        }

        if (ImGui::Button("Reset ThinShell", ImVec2(-1, 0)))
        {
            curState.resetState();
            updateView();
        }
    }   
    ImGui::PopItemWidth();
}

int main(int argc, char* argv[])
{
    CLI::App app("Quasi-static Simulator");
    app.add_option("input,-i,--input", defaultPath, "Input model (json file)")->check(CLI::ExistingFile);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        return app.exit(e);
    }
    
    // Options
    //	polyscope::options::autocenterStructures = true;
    polyscope::view::windowWidth = 800;
    polyscope::view::windowHeight = 1280;

    // Initialize polyscope
    polyscope::init();

    polyscope::view::upDir = polyscope::view::UpDir::ZUp;

    // Add the callback
    polyscope::state::userCallback = callback;

    polyscope::options::groundPlaneHeightFactor = 0.25; // adjust the plane height

    if(loadProblem(defaultPath))
        updateView();
    std::cout << "curEdgeDOFs norm = " << curState.curEdgeDOFs.norm() << std::endl;

    // Show the gui
    polyscope::show();

    return 0;

}
