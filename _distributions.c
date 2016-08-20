/*
 * This file contains functions to evaluate probability distribution functions 
 * (PDFs) quickly by using the C programming language. It provides functions to 
 * evaluate the PDFs and cumulative distributions functions CDFs of the 
 * following distributions
 * - Lognormal distribution
 * - Gamma distribution (based on http://svn.mi.fu-berlin.de/seqan/releases/seqan-1.3.1/lib/samtools/bcftools/kfunc.c)
 *
 * Before the sofware can be run this file needs to be compiled by issuing the 
 * following command from the command line

python _distributionssetup.py
 
 * Other distributions can be implemented easily in two steps
 * 
 * 1. Implement the following template
 
static PyObject* mydistribution_pdf(PyObject* self, PyObject* args)
{
	double x, pdf; //The value at which to evaluate the pdf, the pdf value
	double param1, param2; //Parameters characterizing the distribution

	//Parse the argument tuple and function arguments
	if (!PyArg_ParseTuple(args, "ddd", &param1, &param2, &x) )
		return NULL; //Extracting function arguments failed
	//Calculate the pdf
	pdf = calculate_pdf_for_mydistribution;
	//Return the pdf
	return PyFloat_FromDouble(pdf);
}

static PyObject* mydistribution_cdf(PyObject* self, PyObject* args)
{
	double x, cdf; //The value at which to evaluate the cdf, the cdf value
	double param1, param2; //Parameters characterizing the distribution

	//Parse the argument tuple and function arguments
	if (!PyArg_ParseTuple(args, "ddd", &param1, &param2, &x) )
		return NULL; //Extracting function arguments failed
	//Calculate the cdf
	pdf = calculate_cdf_for_mydistribution;
	//Return the cdf
	return PyFloat_FromDouble(cdf);
}

 * 2. Add the functions to the method table so they can be accessed from python
 * by adding the following lines to the initialisation of _distributionsMethods
 * at the end of this file
 
{"mydistribution_pdf", mydistribution_pdf, METH_VARARGS, "Computes the mydistribution pdf."},
{"mydistribution_cdf", mydistribution_cdf, METH_VARARGS, "Computes the mydistribution cdf."},
 
 */

#include <Python.h>
#include <math.h>

/*
 * Log normal probability distribution.
 */
static PyObject* lognormal_pdf(PyObject* self, PyObject* args)
{
	double mu, sigma, x, pdf; //Parameters of the distribution

	//Parse the argument tuple and extract two vectors
	if (!PyArg_ParseTuple(args, "ddd", &mu, &sigma, &x) )
		return NULL;
	//Calculate the pdf
	pdf = x == 0 ? 0 : exp(-.5 * ((mu - log(x)) / sigma) * ((mu - log(x)) / sigma))
                / (sqrt(2 * M_PI) * sigma * x);
	//Return the pdf
	return PyFloat_FromDouble(pdf);
}

/*
 * Log normal probability distribution.
 */
static PyObject* lognormal_cdf(PyObject* self, PyObject* args)
{
	double mu, sigma, x, cdf; //Parameters of the distribution

	//Parse the argument tuple and extract two vectors
	if (!PyArg_ParseTuple(args, "ddd", &mu, &sigma, &x) )
		return NULL;
	//Calculate the pdf
	cdf = x == 0 ? 0 : .5 * erfc((mu - log(x)) / (M_SQRT2 * sigma));
	//Return the pdf
	return PyFloat_FromDouble(cdf);
}

/*
 * Gamma probability distribution.
 */
static PyObject* gamma_pdf(PyObject* self, PyObject* args)
{
	double alpha, beta, x, pdf; //Parameters of the distribution

	//Parse the argument tuple and extract two vectors
	if (!PyArg_ParseTuple(args, "ddd", &alpha, &beta, &x) )
		return NULL;
	//Calculate the pdf
	pdf = pow(beta, -alpha) / gamma(alpha) * pow(x, alpha - 1) * exp(-x / beta);
	//Return the pdf
	return PyFloat_FromDouble(pdf);
}

//Beginning of functionality copied from
//http://svn.mi.fu-berlin.de/seqan/releases/seqan-1.3.1/lib/samtools/bcftools/kfunc.c

#define KF_GAMMA_EPS 1e-14

double kf_lgamma(double z)
{
	double x = 0;
	x += 0.1659470187408462e-06 / (z+7);
	x += 0.9934937113930748e-05 / (z+6);
	x -= 0.1385710331296526     / (z+5);
	x += 12.50734324009056      / (z+4);
	x -= 176.6150291498386      / (z+3);
	x += 771.3234287757674      / (z+2);
	x -= 1259.139216722289      / (z+1);
	x += 676.5203681218835      / z;
	x += 0.9999999999995183;
	return log(x) - 5.58106146679532777 - z + (z-0.5) * log(z+6.5);
}

static double _kf_gammap(double s, double z)
{
	double sum, x;
	int k;
	for (k = 1, sum = x = 1.; k < 100; ++k) {
		sum += (x *= z / (s + k));
		if (x / sum < KF_GAMMA_EPS) break;
	}
	return exp(s * log(z) - z - kf_lgamma(s + 1.) + log(sum));
}

//End of copied functionality

/*
 * Gamma probability distribution.
 */
static PyObject* gamma_cdf(PyObject* self, PyObject* args)
{
	double alpha, beta, x, pdf; //Parameters of the distribution

	//Parse the argument tuple and extract two vectors
	if (!PyArg_ParseTuple(args, "ddd", &alpha, &beta, &x) )
		return NULL;
	//Calculate the pdf
	pdf = _kf_gammap(alpha, x / beta);
	//Return the pdf
	return PyFloat_FromDouble(pdf);
}

/*
 * Boiler plate code.
 */

static PyMethodDef _distributionsMethods[] =
{
	{"lognormal_pdf", lognormal_pdf, METH_VARARGS, "Computes the log normal pdf."},
	{"lognormal_cdf", lognormal_cdf, METH_VARARGS, "Computes the log normal cdf."},
	{"gamma_pdf", gamma_pdf, METH_VARARGS, "Computes the gamma pdf."},
	{"gamma_cdf", gamma_cdf, METH_VARARGS, "Computes the gamma cdf."},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC

init_distributions(void)
{
	 (void) Py_InitModule("_distributions", _distributionsMethods);
}
