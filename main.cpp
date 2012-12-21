#include "stdafx.h"
#include "SPHSolver.h"
#include "SPHSolverHost.cuh"
#include <iostream>
#include <ctime>
#include <GL\freeglut.h>

void renderScene(void);
void changeSize(int w, int h);

SPHSolver* sphptr;

int main(int argc,char **argv)
{
	//Initialise cuda
	cudaInit(argc,(const char**)argv);
	

	srand(time(NULL));

	    // init GLUT and create window
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(800,600);
	glutInitWindowSize(800,600);
	glutCreateWindow("MFW");
	
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);	
	glEnable (GL_DEPTH_TEST);
    // register callbacks
	glutReshapeFunc(changeSize); 
	glutIdleFunc(renderScene);
	glutDisplayFunc(renderScene);
	



    SPHSolver sph(0.020f);
	sphptr = &sph;
	for(int i = 0; i < 1000; i++)
	{
		sph.AddNewParticle(
			(float)rand()/((float)RAND_MAX/1.0f)-0.8f,
			0,//(float)rand()/((float)RAND_MAX/0.3f)-0.15f,
			(float)rand()/((float)RAND_MAX/1.0f)-0.8f,
			0.09f,3.2f,3.0f,998.0f);
	}

	//Setup the solver
	sph.SetupCUDASolver();
	//Start execution
	glutMainLoop();

}


void renderScene(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glShadeModel ( GL_SMOOTH );



	glLoadIdentity();

	sphptr->Advance(0.005f);
	//
	//TestParticleBuffer(partBuffer,sphptr->GetParticleCount()
	//
	gluLookAt(0,-10.0f,-1.0,0,0,0,0,0,1);
	sphptr->Draw(NULL);
	


	glutSwapBuffers();
}


void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;
	float ratio = 1.0* w / h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

        // Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,1,1000);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
}