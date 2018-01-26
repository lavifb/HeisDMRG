#include "meas.h"
#include "block.h"
#include "linalg.h"
#include "util.h"
#include <mkl.h>
#include <stdio.h>

meas_data_t *createMeas(int num_sites) {

	meas_data_t *meas = mkl_malloc(sizeof(meas_data_t), MEM_DATA_ALIGN);

	meas->num_sites = num_sites;
	meas->Szs = mkl_malloc(num_sites * sizeof(double), MEM_DATA_ALIGN);
	meas->SSs = mkl_malloc(num_sites * sizeof(double), MEM_DATA_ALIGN);

	return meas;
}

void freeMeas(meas_data_t *meas) {

	mkl_free(meas->Szs);
	mkl_free(meas->SSs);

	// if(meas->truncation_error) { mkl_free(meas->truncation_error); }

	mkl_free(meas);
}

/*  Write measure data to files in path
*/
int outputMeasData(const char* filepath, meas_data_t *meas) {
	
	FILE *m_f = fopen(filepath, "w");
	if (m_f == NULL) {
		errprintf("Cannot open file '%s'.\n", filepath);
		return -1;
	}

	fprintf(m_f, "%-6s%-20s%-20s\n"
				 "---------------------------------------------\n"
				 , "Site", "Sz", "SS");

	for (int i = meas->num_sites-1; i>=0; i--) {
		// #if COMPLEX
		// fprintf(m_f, "%-6d%- 20.12f%- 20.12f\n", i+1, meas->Szs[i].real, meas->SSs[i].real);
		// #else
		fprintf(m_f, "%-6d%- 20.12f%- 20.12f\n", i+1, meas->Szs[i], meas->SSs[i]);
		// #endif
	}

	fclose(m_f);

	return 0;
}

/*  Measures Sz and records the measurement in meas.

	Sz_mat_offset: offset in sys_enl->ops that points to Sz

	meas: struct where measurements are recorded
*/
void measureSzs(DMRGBlock *sys_enl, int dimEnv, MAT_TYPE *psi, int Sz_mat_offset, meas_data_t *meas) {
	
	int dimSys = sys_enl->d_block;
	int dimSup = dimSys * dimEnv;
	const model_t *model = sys_enl->model;

	// Make Measurements
	#if COMPLEX
	const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
	const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
	#endif

	// <S_i> spins
	for (int i = 0; i<meas->num_sites; i++) {
		MAT_TYPE* temp = mkl_malloc(dimEnv*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);

		#if COMPLEX
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, &one, psi, dimEnv, sys_enl->ops[i + Sz_mat_offset], dimSys, &zero, temp, dimEnv);
		MKL_Complex16 Szi;
		cblas_zdotc_sub(dimSup, psi, 1, temp, 1, &Szi);
		meas->Szs[i] = Szi.real;
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, 1.0, psi, dimEnv, sys_enl->ops[i + Sz_mat_offset], dimSys, 0.0, temp, dimEnv);
		double Szi = cblas_ddot(dimSup, psi, 1, temp, 1);
		meas->Szs[i] = Szi;
		#endif

		mkl_free(temp);
	}
}

/*  Measures <S_i S_j> correlations and records the measurement in meas.

	Sz_mat_offset: offset in sys_enl->ops that points to Sz

	meas: struct where measurements are recorded
*/
void measureSSs(DMRGBlock *sys_enl, int dimEnv, MAT_TYPE *psi, int Sz_mat_offset, meas_data_t *meas) {
	
	int dimSys = sys_enl->d_block;
	int dimSup = dimSys * dimEnv;
	const model_t *model = sys_enl->model;

	// Make Measurements
	#if COMPLEX
	const MKL_Complex16 one  = {.real=1.0, .imag=0.0};
	const MKL_Complex16 zero = {.real=0.0, .imag=0.0};
	#endif

	// <S_i S_j> correlations
	for (int i = 0; i<meas->num_sites; i++) {
		MAT_TYPE* SSop = mkl_malloc(dimSys*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);
		MAT_TYPE* temp = mkl_malloc(dimEnv*dimSys * sizeof(MAT_TYPE), MEM_DATA_ALIGN);


		#if COMPLEX
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dimSys, dimSys, dimSys, &one, sys_enl->ops[i + Sz_mat_offset], dimSys, sys_enl->ops[1], dimSys, &zero, SSop, dimSys);
		cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, &one, psi, dimEnv, SSop, dimSys, &zero, temp, dimEnv);
		MKL_Complex16 SSi;
		cblas_zdotc_sub(dimSup, psi, 1, temp, 1, &SSi);
		meas->SSs[i] = SSi.real;
		#else
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, dimSys, dimSys, dimSys, 1.0, sys_enl->ops[i + Sz_mat_offset], dimSys, sys_enl->ops[1], dimSys, 0.0, SSop, dimSys);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, dimEnv, dimSys, dimSys, 1.0, psi, dimEnv, SSop, dimSys, 0.0, temp, dimEnv);
		double SSi = cblas_ddot(dimSup, psi, 1, temp, 1);
		meas->SSs[i] = SSi;
		#endif

		mkl_free(temp);
		mkl_free(SSop);
	}
}