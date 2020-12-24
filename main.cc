#include <cstdio>
#include <cmath>
#include "types.h"

const int epochs = 100;
number lr1 = 0.01;
number lr2 = lr1;
const number nf6 = 0.5;
const number nf7 = 0.5;
const number df = 1.0;
const number vs = 1.0;
const int h = 100;

number dot(const vec& a, const vec& b) {
    assert(a.size() == b.size());
    const size_t len = a.size();
    number res = 0;
    for (int i = 0; i < len; i++) {
        res += a[i] * b[i];
    }
    return res;
}

vec add(const vec &a, const vec &b) {
    assert(a.size() == b.size());
    const size_t len = a.size();
    vec res(len);
    for (int i = 0; i < len; i++) {
        res[i] = a[i] + b[i];
    }
    return res;
}

class SVDPP {
    int nu, ni; // num of user, num of item, hidden factors
    number mu;
    mat p, q, y;
    vec bu, bi;
    std::vector< std::vector< int > > *nei; // N
    vec cache_yj;

    number b(int u, int i) { return bu[u] + bi[i]; }
    void update_yj(int u) {
        vec rt(h);
        number f = sqrt((number)((*nei)[u].size()));
        for (int j: (*nei)[u]) for (int i = 0; i < h; i++) rt[i] += y[j][i];
        for (int i = 0; i < h; i++) rt[i] *= f;
        cache_yj = rt;
    }
    number r_hat(int u, int i) {
        // update_yj(u);
        // vec rt = add(p[u], cache_yj);
        return b(u, i) + dot(q[i], p[u]);
    }
    number random() {
        return drand48() * vs;
    }
    void init_variables() {
        p = mat(nu, vec(h));
        q = mat(ni, vec(h));
        y = mat(ni, vec(h));
        bu = vec(nu);
        bi = vec(ni);
        for (int i = 0; i < ni; i++) {
            bi[i] = random();
            for (int j = 0; j < h; j++) {
                q[i][j] = random();
                y[i][j] = random();
            }
        }
        for (int i = 0; i < nu; i++) {
            bu[i] = random();
            for (int j = 0; j < h; j++) {
                p[i][j] = random();
            }
        }
    }
    std::pair<number,number> learn(int u, int i, number r) {
        number rh = r_hat(u, i);
        number e = r - rh;
        number nudrt = sqrt((number)(*nei)[u].size());
        bu[u] += lr1 * (e - nf6 * bu[u]);
        bi[i] += lr1 * (e - nf6 * bi[i]);
        for (int k = 0; k < h; k++) { q[i][k] += lr2 * (e * p[u][k] - nf7 * q[i][k]); }
        for (int k = 0; k < h; k++) p[u][k] += lr2 * (e * q[i][k] - nf7 * p[u][k]);
        // for (int j: (*nei)[u]) {
        //     for (int k = 0; k < h; k++) y[j][k] += lr2 * (e * nudrt * q[i][k] - nf7 * y[j][k]);
        // }
        return std::make_pair(e, ((rh > 0.5 ? 1 : 0) == r));
    }
public:
    SVDPP(int nu, int ni, number mu, std::vector<std::vector<int> > * nei) {
        this->nu = nu;
        this->ni = ni;
        this->mu = mu;
        this->nei = nei;
        init_variables();
    }
    void decay() {
        lr1 *= df;
        lr2 *= df;
    }
    number predict(int u, int i) { return r_hat(u, i); }
    number operator()(int u, int i) { return predict(u, i); }
    auto fit(int u, int i, number r) { return learn(u, i, r); }
};

using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("./%s <data.dat> <input.dat>\n", argv[0]);
        return -1;
    }
    FILE * data = fopen(argv[1], "r");
    FILE * test = fopen(argv[2], "r");
    int nu, ni;

    srand(135);

    puts("fetching nu, ni...");
    fscanf(data, "%d%d", &nu, &ni);
    puts("fetching ratings data...");
    vector< vector< pi > > ratings(nu);
    vector< vector< int > > nei(nu);
    vector<pi> tests;
    number mu = .0;
    int cnt = 0;
    for (int i = 0; i < nu; i++) {
        int len; fscanf(data, "%d", &len);
        for (int j = 0; j < len; j++) {
            int jid, rt;
            fscanf(data, "%d%d", &jid, &rt);
            assert(jid < ni);
            assert(rt == 0 || rt == 1);
            ratings[i].push_back(make_pair(jid, rt));
            mu += rt;
            ++cnt;
        }
    }
    puts("fetching neighbors data...");
    mu /= cnt;
    for (int i = 0; i < nu; i++) {
        int len; fscanf(data, "%d", &len);
        for (int j = 0; j < len; j++) {
            int jid;
            fscanf(data, "%d", &jid);
            assert(jid < ni);
            nei[i].push_back(jid);
        }
    }
    fclose(data);

    puts("fetching test data...");
    int nt;
    fscanf(test, "%d", &nt);
    for (int k = 0; k < nt; k++) {
        int u, i; fscanf(test, "%d%d", &u, &i);
        tests.push_back(make_pair(u, i));
    }
    fclose(test);

    SVDPP mac(nu, ni, mu, &nei);
    puts("initialized machine...");

    puts("fitting machines...");
    for (int ei = 0; ei < epochs; ei++) {
        int c = 0;
        pair<number,number> es = make_pair(0.0, 0);
        int lc = 0;
        for (int u = 0; u < nu; u++) {
            for (pi rt: ratings[u]) {
                auto r = mac.fit(u, rt.first, rt.second);
                es.first += r.first * r.first;
                es.second += r.second;
                ++c;
                ++lc;
            }
        }
        printf("  completed epoch %d... (%lf, %lf)\n", ei + 1, sqrt(es.first/lc), es.second/lc);
        mac.decay();
    }

    FILE * result = fopen("./result.csv", "w");
    fprintf(result, "applied\n");
    for (int k = 0; k < nt; k++) {
        auto t = tests[k];
        number p = mac.predict(t.first, t.second);
        fprintf(result, "%d\n", p > 0.5 ? 1 : 0);
    }
    fclose(result);

    return 0;
}
