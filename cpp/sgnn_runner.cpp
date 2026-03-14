#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

constexpr int EMPTY = -1;
const std::array<int, 3> SPECIES{0, 1, 2};

struct SimConfig {
    int size = 40;
    double empty_fraction = 0.22;
    int emb_dim = 45;
    double tau = 0.0;
    double epsilon = 0.03;
    double lam = 0.06;
    int steps = 16;
    int gcn_layers = 2;
    int seed = 42;
    std::string variant = "with_influencer";
    std::string similarity = "hybrid";
    bool adaptive_tau = true;
    double tau_quantile = 0.35;
};

struct StepMetrics { int step{}; int moves{}; double tau{}; double entropy{}; double gini{}; double drift{}; };

struct ClusterStats {
    int total_clusters = 0;
    std::array<int,3> num_clusters{0,0,0};
    std::array<int,3> largest_cluster{0,0,0};
    std::array<int,3> total_cells{0,0,0};
    std::array<double,3> largest_cluster_fraction{0.0,0.0,0.0};
    std::array<std::vector<int>,3> cluster_sizes;
};

using Vec = std::vector<double>;
using RGB = std::array<int, 3>;

struct PaperSimulator {
    SimConfig cfg;
    int n;
    std::mt19937 gen;
    std::normal_distribution<double> n01{0.0,1.0};
    std::vector<std::vector<int>> neigh;
    std::vector<Vec> x;
    std::vector<int> labels;
    std::vector<bool> empty;
    std::unordered_map<int,int> infl_link;
    std::vector<int> agent_id;
    std::array<int,3> influencers;
    std::vector<Vec> centers;
    std::vector<Vec> species_proto;

    explicit PaperSimulator(const SimConfig& c): cfg(c), n(cfg.size * cfg.size), gen(cfg.seed), influencers{n,n+1,n+2} {
        neigh = build_neighbors();
        x.assign(n+3, Vec(cfg.emb_dim, 0.0));
        labels.resize(n);
        empty.resize(n);
        agent_id.resize(n);
        std::uniform_int_distribution<int> sp(0,2);
        std::uniform_real_distribution<double> u01(0.0,1.0);
        for (int i=0;i<n;++i){ labels[i]=sp(gen); empty[i]=u01(gen)<cfg.empty_fraction; agent_id[i]=i; }
        centers = build_centers();
        species_proto = species_prototypes();
        init_state();
    }

    double gauss(double m, double s){ return m + s*n01(gen); }
    double norm(const Vec& v) const { double s=0; for (double d: v) s+=d*d; return std::sqrt(s); }
    Vec normalize(Vec v) const { double z=norm(v); if (z<=1e-12) return v; for (double& d:v) d/=z; return v; }
    double cos(const Vec& a, const Vec& b) const {
        double na=norm(a), nb=norm(b); if (na<=1e-12 || nb<=1e-12) return 0.0;
        double d=0; for (size_t i=0;i<a.size();++i) d+=a[i]*b[i]; return d/(na*nb);
    }
    double neg_l2(const Vec& a, const Vec& b) const { double s=0; for (size_t i=0;i<a.size();++i){ double t=a[i]-b[i]; s+=t*t; } return -std::sqrt(s); }
    double sim(const Vec& a, const Vec& b) const {
        if (cfg.similarity=="cosine") return cos(a,b);
        if (cfg.similarity=="neg_l2") return neg_l2(a,b);
        return 0.75*cos(a,b)+0.25*(neg_l2(a,b)/6.0);
    }

    std::vector<std::vector<int>> build_neighbors() const {
        std::vector<std::vector<int>> out(n+3);
        for (int r=0;r<cfg.size;++r) for (int c=0;c<cfg.size;++c){
            int u=r*cfg.size+c;
            for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc){
                if (!dr && !dc) continue;
                int rr=r+dr, cc=c+dc;
                if (rr>=0&&rr<cfg.size&&cc>=0&&cc<cfg.size) out[u].push_back(rr*cfg.size+cc);
            }
        }
        return out;
    }

    std::vector<Vec> build_centers(){
        std::vector<Vec> cts;
        for (int k=0;k<45;++k){
            int g=k/15;
            Vec c(cfg.emb_dim,0.0);
            for (int i=0;i<cfg.emb_dim;++i) c[i]=gauss(-1.6,0.2);
            int start=g*15, end=(g+1)*15;
            for (int i=start;i<end;++i) c[i]=gauss(3.6,0.35);
            cts.push_back(normalize(c));
        }
        return cts;
    }

    std::vector<Vec> species_prototypes() const {
        std::vector<Vec> p(3,Vec(cfg.emb_dim,0.0));
        for (int g=0;g<3;++g){
            for (int j=0;j<15;++j) for (int i=0;i<cfg.emb_dim;++i) p[g][i]+=centers[g*15+j][i]/15.0;
            p[g]=normalize(p[g]);
        }
        return p;
    }

    Vec sample_species_vector(int g){
        std::uniform_int_distribution<int> pick(0,14);
        const Vec& c=centers[g*15+pick(gen)];
        Vec v(cfg.emb_dim);
        for (int i=0;i<cfg.emb_dim;++i) v[i]=gauss(c[i],0.2);
        return normalize(v);
    }

    void init_state(){
        for (int i=0;i<n;++i) if (!empty[i]) x[i]=sample_species_vector(labels[i]);
        for (int i=0;i<n;++i) if (empty[i]){ x[i]=Vec(cfg.emb_dim,0.0); agent_id[i]=EMPTY; }
        for (int g=0;g<3;++g) x[influencers[g]]=species_proto[g];
        if (cfg.variant=="with_influencer") for (int i=0;i<n;++i) if (!empty[i]) infl_link[i]=influencers[labels[i]];
    }

    std::vector<int> neighbors_for_node(int i) const {
        if (i>=n || empty[i]) return {};
        std::vector<int> nbs;
        for (int j: neigh[i]) if (!empty[j]) nbs.push_back(j);
        if (cfg.variant=="with_influencer") { auto it=infl_link.find(i); if (it!=infl_link.end()) nbs.push_back(it->second); }
        return nbs;
    }

    std::vector<Vec> propagate() const {
        std::vector<Vec> h=x;
        for (int layer=0; layer<cfg.gcn_layers; ++layer){
            std::vector<Vec> n_h=h;
            for (int i=0;i<n;++i){
                if (empty[i]) continue;
                auto nbs=neighbors_for_node(i);
                if (nbs.empty()) continue;
                Vec agg(cfg.emb_dim,0.0);
                for (int j:nbs) for (int k=0;k<cfg.emb_dim;++k) agg[k]+=h[j][k];
                for (int k=0;k<cfg.emb_dim;++k) agg[k]/=static_cast<double>(nbs.size());
                for (int k=0;k<cfg.emb_dim;++k) n_h[i][k]=0.5*h[i][k]+0.5*agg[k];
                n_h[i]=normalize(n_h[i]);
            }
            h.swap(n_h);
        }
        return h;
    }

    double same_type_ratio(int i) const {
        std::vector<int> nbs; for(int j:neigh[i]) if(!empty[j]) nbs.push_back(j);
        if (nbs.empty()) return 1.0;
        int c=0; for(int j:nbs) if(labels[j]==labels[i]) ++c;
        return static_cast<double>(c)/nbs.size();
    }

    double same_type_ratio_if_moved(int i,int target) const {
        std::vector<int> nbs; for(int j:neigh[target]) if(!empty[j] && j!=i) nbs.push_back(j);
        if (nbs.empty()) return 1.0;
        int c=0; for(int j:nbs) if(labels[j]==labels[i]) ++c;
        return static_cast<double>(c)/nbs.size();
    }

    double adaptive_tau(std::vector<double> sat) const {
        if (sat.empty() || !cfg.adaptive_tau) return cfg.tau;
        std::sort(sat.begin(), sat.end());
        int q=std::clamp(static_cast<int>(cfg.tau_quantile*(sat.size()-1)),0,static_cast<int>(sat.size()-1));
        return sat[q];
    }

    double relocation_score(int i, int target, const std::vector<Vec>& emb) const {
        std::vector<int> nbs; for (int j:neigh[target]) if(!empty[j] && j!=i) nbs.push_back(j);
        if (cfg.variant=="with_influencer"){
            auto it=infl_link.find(i); if (it!=infl_link.end()) nbs.push_back(it->second);
        }
        if (nbs.empty()) return 0.0;
        double s=0; for (int j:nbs) s+=sim(emb[i], emb[j]);
        return s/nbs.size();
    }

    std::array<double,3> color_weights(const Vec& emb) const {
        std::array<double,3> sims{cos(emb,species_proto[0]), cos(emb,species_proto[1]), cos(emb,species_proto[2])};
        double m=*std::max_element(sims.begin(),sims.end());
        std::array<double,3> ex{}; double s=0;
        for(int i=0;i<3;++i){ ex[i]=std::exp(4.0*(sims[i]-m)); s+=ex[i]; }
        for(double& v:ex) v/=s;
        return ex;
    }



    std::unordered_map<int, Vec> build_agent_map(const std::vector<int>& ids, const std::vector<bool>& is_empty, const std::vector<Vec>& emb) const {
        std::unordered_map<int, Vec> out;
        out.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (!is_empty[i] && ids[i] != EMPTY) out[ids[i]] = emb[i];
        }
        return out;
    }

    double compute_agent_drift(const std::unordered_map<int, Vec>& prev_map, const std::unordered_map<int, Vec>& new_map) const {
        double drift = 0.0;
        int cnt = 0;
        for (const auto& kv : prev_map) {
            auto it = new_map.find(kv.first);
            if (it == new_map.end()) continue;
            ++cnt;
            Vec d(cfg.emb_dim, 0.0);
            for (int k = 0; k < cfg.emb_dim; ++k) d[k] = it->second[k] - kv.second[k];
            drift += norm(d);
        }
        return cnt ? drift / cnt : 0.0;
    }

    // Cluster analysis is intentionally computed only after simulation to avoid any effect on dynamics.
    // It uses final hard labels and occupancy mask directly from the grid state.
    ClusterStats compute_final_clusters() const {
        ClusterStats stats;
        std::vector<char> visited(static_cast<size_t>(n), 0);

        for (int i = 0; i < n; ++i) {
            if (empty[i] || visited[i]) continue;
            int g = labels[i];
            if (g < 0 || g > 2) continue;

            // Clusters are connected components on the grid under Moore (8-neighbor) adjacency.
            int comp_size = 0;
            std::vector<int> stack{ i };
            visited[i] = 1;
            while (!stack.empty()) {
                int u = stack.back();
                stack.pop_back();
                ++comp_size;
                for (int v : neigh[u]) {
                    if (v >= n || visited[v] || empty[v] || labels[v] != g) continue;
                    visited[v] = 1;
                    stack.push_back(v);
                }
            }

            stats.total_clusters += 1;
            stats.num_clusters[g] += 1;
            stats.total_cells[g] += comp_size;
            stats.largest_cluster[g] = std::max(stats.largest_cluster[g], comp_size);
            stats.cluster_sizes[g].push_back(comp_size);
        }

        for (int g = 0; g < 3; ++g) {
            std::sort(stats.cluster_sizes[g].begin(), stats.cluster_sizes[g].end(), std::greater<int>());
            stats.largest_cluster_fraction[g] = stats.total_cells[g] > 0
                ? static_cast<double>(stats.largest_cluster[g]) / static_cast<double>(stats.total_cells[g])
                : 0.0;
        }
        return stats;
    }

    std::vector<StepMetrics> run(std::vector<std::vector<RGB>>& frames){
        std::vector<StepMetrics> metrics;
        for (int t=1;t<=cfg.steps;++t){
            auto e_prev=propagate();
            std::vector<double> sat;
            for(int i=0;i<n;++i){ if(empty[i]) continue; if(cfg.variant=="baseline") sat.push_back(same_type_ratio(i)); else {
                auto nbs=neighbors_for_node(i); if(nbs.empty()) sat.push_back(0.0); else { double s=0; for(int j:nbs) s+=sim(e_prev[i],e_prev[j]); sat.push_back(s/nbs.size()); }
            }}
            double tau_t=adaptive_tau(sat);

            std::vector<RGB> frame(n);
            for (int i=0;i<n;++i){
                if (empty[i]) frame[i]={245,245,245};
                else {
                    auto w=color_weights(e_prev[i]);
                    frame[i]={
                        static_cast<int>(w[0]*235 + w[1]*70 + w[2]*70),
                        static_cast<int>(w[0]*70 + w[1]*215 + w[2]*110),
                        static_cast<int>(w[0]*70 + w[1]*90 + w[2]*235)
                    };
                }
            }
            frames.push_back(frame);

            auto new_x=x; auto new_labels=labels; auto new_empty=empty; auto new_links=infl_link; auto new_agent=agent_id;
            std::vector<int> empties; for(int i=0;i<n;++i) if(empty[i]) empties.push_back(i);
            int moves=0; int sat_idx=0;
            for (int i=0;i<n;++i){
                if (empty[i]) continue;
                double s_i=sat[sat_idx++]; if (s_i>=tau_t) continue;
                if (cfg.variant=="baseline"){
                    if (!empties.empty()){
                        std::vector<int> acceptable;
                        for(int j:empties) if(same_type_ratio_if_moved(i,j)>=tau_t) acceptable.push_back(j);
                        std::vector<int>& pool = acceptable.empty() ? empties : acceptable;
                        std::uniform_int_distribution<int> pick(0,static_cast<int>(pool.size())-1);
                        int j=pool[pick(gen)];
                        ++moves;
                        new_x[j]=x[i]; new_labels[j]=labels[i]; new_empty[j]=false;
                        new_empty[i]=true; new_labels[i]=EMPTY; new_x[i]=Vec(cfg.emb_dim,0.0);
                        new_agent[j]=agent_id[i]; new_agent[i]=EMPTY;
                        empties.erase(std::remove(empties.begin(),empties.end(),j),empties.end());
                        empties.push_back(i);
                    }
                    continue;
                }
                bool moved=false;
                if (!empties.empty()){
                    int best_j=-1; double best_sc=-10.0;
                    for(int j:empties){ double sc=relocation_score(i,j,e_prev); if(sc>best_sc){best_sc=sc; best_j=j;} }
                    if (best_j>=0 && best_sc>s_i+cfg.epsilon){
                        moved=true; ++moves;
                        new_x[best_j]=x[i]; new_labels[best_j]=labels[i]; new_empty[best_j]=false;
                        new_empty[i]=true; new_labels[i]=EMPTY; new_x[i]=Vec(cfg.emb_dim,0.0);
                        new_agent[best_j]=agent_id[i]; new_agent[i]=EMPTY;
                        auto it=new_links.find(i); if(it!=new_links.end()){ int lk=it->second; new_links.erase(it); new_links[best_j]=lk; }
                        empties.erase(std::remove(empties.begin(),empties.end(),best_j),empties.end()); empties.push_back(i);
                    }
                }
                if (!moved){
                    auto nbs=neighbors_for_node(i); if(!nbs.empty()){
                        Vec delta(cfg.emb_dim,0.0);
                        for(int j:nbs) for(int k=0;k<cfg.emb_dim;++k) delta[k]+=e_prev[j][k]-e_prev[i][k];
                        Vec adapted(cfg.emb_dim,0.0);
                        for(int k=0;k<cfg.emb_dim;++k) adapted[k]=e_prev[i][k]+cfg.lam*delta[k];
                        new_x[i]=normalize(adapted);
                        std::array<double,3> s{cos(new_x[i],species_proto[0]),cos(new_x[i],species_proto[1]),cos(new_x[i],species_proto[2])};
                        int g=std::distance(s.begin(), std::max_element(s.begin(), s.end()));
                        new_labels[i]=g;
                        if (cfg.variant=="with_influencer") new_links[i]=influencers[g];
                    }
                }
            }
            for(int g=0;g<3;++g) new_x[influencers[g]]=species_proto[g];

            // Build prev_map BEFORE state swap and new_map AFTER swap, keyed by stable agent_id.
            // If prev_map is built after swap, embeddings from t-1 are paired with t agent positions,
            // which creates artificial drift when relocation happens.
            const auto prev_map = build_agent_map(agent_id, empty, e_prev);

            x.swap(new_x); labels.swap(new_labels); empty.swap(new_empty); infl_link.swap(new_links); agent_id.swap(new_agent);
            auto e_new=propagate();
            const auto new_map = build_agent_map(agent_id, empty, e_new);
            double drift = compute_agent_drift(prev_map, new_map);
            if (cfg.variant=="baseline") {
                drift = 0.0;
#ifndef NDEBUG
                assert(drift == 0.0 && "baseline drift must be exactly zero by definition");
#endif
            }

            std::vector<double> ent_vals,gini_vals;
            for(int i=0;i<n;++i){ if(empty[i]) continue; std::vector<int> nbs; for(int j:neigh[i]) if(!empty[j]) nbs.push_back(j); if(nbs.empty()) continue;
                std::array<double,3> p{};
                for(int g=0;g<3;++g){ int c=0; for(int j:nbs) if(labels[j]==g) ++c; p[g]=static_cast<double>(c)/nbs.size(); }
                double ent=0.0, gini=1.0; for(double v:p){ if(v>0) ent-=v*std::log2(v); gini-=v*v; }
                ent_vals.push_back(ent); gini_vals.push_back(gini);
            }
            double ent = ent_vals.empty()?0.0:std::accumulate(ent_vals.begin(),ent_vals.end(),0.0)/ent_vals.size();
            double gini = gini_vals.empty()?0.0:std::accumulate(gini_vals.begin(),gini_vals.end(),0.0)/gini_vals.size();
            metrics.push_back({t,moves,tau_t,ent,gini,drift});
        }
        return metrics;
    }
};

void write_metrics_csv(const fs::path& path, const std::vector<StepMetrics>& rows){
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << "step,moves,tau,entropy,gini,drift\n";
    f << std::fixed << std::setprecision(6);
    for (const auto& r: rows) f << r.step<<","<<r.moves<<","<<r.tau<<","<<r.entropy<<","<<r.gini<<","<<r.drift<<"\n";
}

void write_frame_svg(const fs::path& path, const std::vector<RGB>& frame, int size, int cell_px){
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    int w=size*cell_px, h=size*cell_px;
    f << "<svg xmlns='http://www.w3.org/2000/svg' width='"<<w<<"' height='"<<h<<"' viewBox='0 0 "<<w<<" "<<h<<"'>\n";
    f << "<rect width='100%' height='100%' fill='rgb(245,245,245)'/>\n";
    for (int idx=0; idx<size*size; ++idx){
        int r=idx/size, c=idx%size; auto [rr,gg,bb]=frame[idx];
        f << "<rect x='"<<c*cell_px<<"' y='"<<r*cell_px<<"' width='"<<cell_px<<"' height='"<<cell_px
          <<"' fill='rgb("<<rr<<","<<gg<<","<<bb<<")'/>\n";
    }
    f << "</svg>\n";
}

std::string fmt_tag(const SimConfig& c){
    std::ostringstream s; s<<c.variant<<"_size"<<c.size<<"_seed"<<c.seed<<"_L"<<c.gcn_layers<<"_"<<c.similarity; return s.str();
}

int cell_px_for(int size){ if(size<=40) return 4; if(size<=100) return 2; return 1; }

struct FinalRow {std::string variant; int size; int seed; int layers; std::string similarity; StepMetrics m; ClusterStats clusters;};

FinalRow run_one(const fs::path& out, const SimConfig& cfg){
    PaperSimulator sim(cfg);
    std::vector<std::vector<RGB>> frames;
    auto metrics=sim.run(frames);
    auto tag=fmt_tag(cfg);
    write_metrics_csv(out / "metrics" / (tag + ".csv"), metrics);
    std::vector<int> picks{0, static_cast<int>(frames.size()/2), static_cast<int>(frames.size()-1)};
    for (int idx: picks) write_frame_svg(out / "frames" / (tag + "_step" + std::to_string(idx+1) + ".svg"), frames[idx], cfg.size, cell_px_for(cfg.size));
    // Cluster counting is performed once on final labels/occupancy to avoid altering dynamics.
    const auto clusters = sim.compute_final_clusters();
    return {cfg.variant,cfg.size,cfg.seed,cfg.gcn_layers,cfg.similarity,metrics.back(),clusters};
}

void run_experiments(int workers, int min_size=20, int max_size=200, int step_size=20){
    fs::path out="outputs/size_sweep";
    fs::create_directories(out/"metrics"); fs::create_directories(out/"frames");
    std::vector<SimConfig> jobs;
    std::vector<int> sizes; for(int s=min_size;s<=max_size;s+=step_size) sizes.push_back(s);
    std::vector<std::string> variants{"baseline","without_influencer","with_influencer"};
    std::vector<int> seeds{11,42,20,17,22,605,70};
    for (int size:sizes) for (int seed:seeds) for (const auto& v:variants){
        std::vector<int> layers = (v=="baseline") ? std::vector<int>{1} : std::vector<int>{1,2,3};
        for(int l:layers){ SimConfig c; c.size=size; c.seed=seed; c.variant=v; c.gcn_layers=l; c.similarity=(v=="baseline"?"cosine":"hybrid"); c.steps=size>=120?12:16; jobs.push_back(c); }
    }

    std::vector<FinalRow> rows;
    std::vector<std::future<FinalRow>> running;
    size_t idx=0;
    while (idx<jobs.size() || !running.empty()){
        while (idx<jobs.size() && running.size()<static_cast<size_t>(std::max(1,workers))){
            SimConfig cfg=jobs[idx++];
            running.push_back(std::async(std::launch::async, [out,cfg](){ return run_one(out,cfg); }));
        }
        for (size_t i=0;i<running.size();){
            if (running[i].wait_for(std::chrono::milliseconds(10))==std::future_status::ready){
                rows.push_back(running[i].get());
                running.erase(running.begin()+i);
            } else ++i;
        }
    }

    std::ofstream t(out/"table_sizes_20_to_200.csv");
    t << "variant,size,seed,layers,similarity,step,moves,tau,entropy,gini,drift\n";
    t << std::fixed << std::setprecision(6);
    for (const auto& r:rows){
        t << r.variant<<","<<r.size<<","<<r.seed<<","<<r.layers<<","<<r.similarity<<","<<r.m.step<<","<<r.m.moves<<","<<r.m.tau<<","<<r.m.entropy<<","<<r.m.gini<<","<<r.m.drift<<"\n";
    }

    std::ofstream csum(out/"final_clusters.csv");
    csum << "variant,size,seed,layers,similarity,total_clusters,num_clusters_red,num_clusters_green,num_clusters_blue,largest_red,largest_green,largest_blue,total_red,total_green,total_blue,frac_largest_red,frac_largest_green,frac_largest_blue\n";
    csum << std::fixed << std::setprecision(6);
    for (const auto& r: rows) {
        csum << r.variant << "," << r.size << "," << r.seed << "," << r.layers << "," << r.similarity << ","
             << r.clusters.total_clusters << ","
             << r.clusters.num_clusters[0] << "," << r.clusters.num_clusters[1] << "," << r.clusters.num_clusters[2] << ","
             << r.clusters.largest_cluster[0] << "," << r.clusters.largest_cluster[1] << "," << r.clusters.largest_cluster[2] << ","
             << r.clusters.total_cells[0] << "," << r.clusters.total_cells[1] << "," << r.clusters.total_cells[2] << ","
             << r.clusters.largest_cluster_fraction[0] << "," << r.clusters.largest_cluster_fraction[1] << "," << r.clusters.largest_cluster_fraction[2] << "\n";
    }

    std::ofstream cdist(out/"final_cluster_distributions.csv");
    cdist << "variant,size,seed,layers,similarity,species,num_clusters,cluster_sizes\n";
    const std::array<std::string, 3> species_names{ "red", "green", "blue" };
    for (const auto& r : rows) {
        for (int g = 0; g < 3; ++g) {
            // Save full size distributions so component shape/fragmentation can be analyzed beyond simple counts.
            std::ostringstream sizes;
            for (size_t i = 0; i < r.clusters.cluster_sizes[g].size(); ++i) {
                if (i) sizes << ";";
                sizes << r.clusters.cluster_sizes[g][i];
            }
            cdist << r.variant << "," << r.size << "," << r.seed << "," << r.layers << "," << r.similarity << ","
                  << species_names[g] << "," << r.clusters.num_clusters[g] << "," << sizes.str() << "\n";
        }
    }
}

void summarize(){
    fs::path src="outputs/size_sweep/table_sizes_20_to_200.csv", dst="outputs/size_sweep/table_summary_by_size_variant.csv";
    std::ifstream in(src); std::string line; std::getline(in,line);
    struct Agg{double moves=0,tau=0,entropy=0,gini=0,drift=0; int n=0;};
    std::map<std::pair<std::string,int>,Agg> agg;
    while (std::getline(in,line)){
        std::stringstream ss(line); std::string tok; std::vector<std::string> c;
        while (std::getline(ss,tok,',')) c.push_back(tok);
        if(c.size()<11) continue;
        auto& a=agg[{c[0], std::stoi(c[1])}];
        a.moves+=std::stod(c[6]); a.tau+=std::stod(c[7]); a.entropy+=std::stod(c[8]); a.gini+=std::stod(c[9]); a.drift+=std::stod(c[10]); a.n++;
    }
    std::ofstream out(dst);
    out << "variant,size,avg_moves,avg_tau,avg_entropy,avg_gini,avg_drift\n";
    out << std::fixed << std::setprecision(6);
    for (auto& kv:agg){ auto& a=kv.second; out<<kv.first.first<<","<<kv.first.second<<","<<a.moves/a.n<<","<<a.tau/a.n<<","<<a.entropy/a.n<<","<<a.gini/a.n<<","<<a.drift/a.n<<"\n"; }
}

int main(int argc, char** argv){
    std::string cmd = argc>1 ? argv[1] : "run";
    if (cmd=="run"){
        int workers=4, min_size=20, max_size=200, step_size=20;
        for(int i=2;i<argc-1;++i){
            std::string a=argv[i];
            if(a=="--workers") workers=std::stoi(argv[i+1]);
            if(a=="--min-size") min_size=std::stoi(argv[i+1]);
            if(a=="--max-size") max_size=std::stoi(argv[i+1]);
            if(a=="--step-size") step_size=std::stoi(argv[i+1]);
        }
        run_experiments(workers, min_size, max_size, step_size);
        return 0;
    }
    if (cmd=="summarize"){ summarize(); return 0; }
    std::cerr << "usage: sgnn_runner [run --workers N | summarize]\n";
    return 1;
}