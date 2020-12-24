#include <vector>

typedef double number;
typedef std::vector<number> vec;
typedef std::vector<vec> mat;
typedef std::pair<int,int> pi;

number dot(const vec&, const vec&);
vec dot(const vec&, const number);
vec dot(const number, const vec&);
vec add(const vec&, const vec&);
vec ew_prod(const vec&, const vec&);
