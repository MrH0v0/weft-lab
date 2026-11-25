// SPDX-License-Identifier: MIT
// Finite-volume update for the local wet knit lattice. Lengths are metres,
// reservoir inventories are millilitres, and all arithmetic is double precision.
extern "C" __declspec(dllexport) void weft_transport_film(
    int cells, int steps, double dt, double pitch, double viscosity, double tension,
    double uptake, double drying, double uptake_rate, const int* neighbour,
    const unsigned char* connected, const double* edge, const double* length,
    const double* width, const double* area, const double* capacity,
    double* outer, double* knit, double* momentum, double* result) {
    std::vector<double> depth(cells), velocity(3*cells);
    std::vector<double> water_delta(cells), momentum_delta(3*cells);
    const double uptake_dt = uptake_rate * uptake * dt / std::sqrt(viscosity);
    for (int step=0;step<steps;++step) {
        for (int i=0;i<cells;++i) {
            const double prior = outer[i];
            const double h = prior * 1e-6 / area[i];
            const double empty = std::max(0.,capacity[i]-knit[i]);
            const double absorbed = prior == 0. ? 0. : std::min(prior,
                empty * -std::expm1(-uptake_dt*(1.+2.*h/(h+.00035))));
            outer[i] -= absorbed;
            knit[i] += absorbed;
            result[5] += absorbed;
            depth[i] = outer[i]*1e-6/area[i];
            const double mm = depth[i]*1000.;
            const double damping = 1.+dt*viscosity*(8.+4./std::max(.015,mm*mm));
            const double retained = outer[i]/std::max(prior,1e-12);
            for (int k=0;k<3;++k) {
                const int v=3*i+k;
                velocity[v]=momentum[v]*retained/std::max(outer[i],1e-12)/damping;
                momentum[v]=velocity[v]*outer[i];
            }
        }
        std::fill(water_delta.begin(),water_delta.end(),0.);
        std::fill(momentum_delta.begin(),momentum_delta.end(),0.);
        for (int i=0;i<cells;++i) {
            if (outer[i]==0.) continue;
            double flux[4],sum=0.;
            for (int k=0;k<4;++k) {
                const int face=4*i+k;
                const int j=neighbour[face]<0 ? 0 : neighbour[face];
                const bool inside=connected[face]!=0;
                const double* d=edge+3*face;
                const double other=inside ? depth[j] : depth[i];
                const double slope=(-d[2]+(depth[i]-other)*(1.-.70*tension))/length[face];
                const double h=slope>=0. ? depth[i] : other;
                const double drive=9.81*h*h*slope/(3e-6*viscosity);
                const double wave=std::sqrt(19.62*std::max(0.,h));
                const double hydraulic=drive/(1.+std::abs(drive)/std::max(wave,1e-12));
                double impact=0.;
                for (int c=0;c<3;++c) {
                    const double speed=inside ? (momentum[3*i+c]+momentum[3*j+c])/std::max(outer[i]+outer[j],1e-12) : velocity[3*i+c];
                    impact+=speed*(d[c]/length[face]);
                }
                flux[k]=std::max(0.,impact+hydraulic)*depth[i]*width[face]*1e6;
                sum+=flux[k];
            }
            const double fraction=std::min(dt,.85*outer[i]/std::max(sum,1e-30));
            for (int k=0;k<4;++k) {
                const double q=flux[k]*fraction;
                if (!q) continue;
                const int face=4*i+k, j=neighbour[face];
                water_delta[i]-=q;
                for(int c=0;c<3;++c) momentum_delta[3*i+c]-=q*velocity[3*i+c];
                if (connected[face]) {
                    water_delta[j]+=q;
                    for(int c=0;c<3;++c) {
                        momentum_delta[3*j+c]+=q*velocity[3*i+c];
                        result[c]+=q*edge[3*face+c];
                    }
                } else {
                    result[3]+=q;
                    if(edge[3*face+2]<-pitch*.1) result[4]+=q;
                }
            }
        }
        for (int i=0;i<cells;++i) {
            const double prior=outer[i]+water_delta[i];
            const double dry_outer=std::min(prior,area[i]*drying*.9*dt);
            const double dry_knit=std::min(knit[i],area[i]*drying*.55*dt);
            outer[i]=prior-dry_outer;knit[i]-=dry_knit;
            result[6]+=dry_outer+dry_knit;
            const double retain=outer[i]/std::max(prior,1e-12);
            for(int c=0;c<3;++c) momentum[3*i+c]=(momentum[3*i+c]+momentum_delta[3*i+c])*retain;
        }
    }
}
