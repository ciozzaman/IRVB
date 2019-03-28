
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_020546b21a6d4ca3578b83a0d6493bee : public Expression
  {
     public:
       double foilhoriz;
std::shared_ptr<dolfin::GenericFunction> generic_function_T1;
std::shared_ptr<dolfin::GenericFunction> generic_function_T2;


       dolfin_expression_020546b21a6d4ca3578b83a0d6493bee()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          double T2;
            generic_function_T2->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&T2), x);
          double T1;
            generic_function_T1->eval(Eigen::Map<Eigen::Matrix<double, 1, 1>>(&T1), x);
          values[0] = phi*(x[0]/foilhoriz-pow(x[0]/foilhoriz,2))+T2*x[0]/foilhoriz+T1*(1-x[0]/foilhoriz);

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "foilhoriz") { foilhoriz = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "foilhoriz") return foilhoriz;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {
          if (name == "T1") { generic_function_T1 = _value; return; }          if (name == "T2") { generic_function_T2 = _value; return; }
       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {
          if (name == "T1") return generic_function_T1;          if (name == "T2") return generic_function_T2;
       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_020546b21a6d4ca3578b83a0d6493bee()
{
  return new dolfin::dolfin_expression_020546b21a6d4ca3578b83a0d6493bee;
}

