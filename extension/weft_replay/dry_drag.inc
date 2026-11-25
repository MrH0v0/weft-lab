// SPDX-License-Identifier: MIT
// Local cloth operations on owned arrays, with the same supports as the NumPy path.
template<class T> void gauge_samples(const T* points,const int* corners,const double* weights,int count,
        Vec* centers,Vec* vectors,double* lengths,double* angles) {
    for(int i=0;i<count;++i) {
        Vec samples[4]{};
        for(int k=0;k<4;++k)for(int q=0;q<3;++q) {
            const int j=12*i+3*k+q;
            samples[k]=samples[k]+read3(points,corners[j])*weights[j];
        }
        const Vec across=samples[1]-samples[0],up=samples[3]-samples[2];
        centers[i]=(samples[0]+samples[1]+samples[2]+samples[3])*.25;
        vectors[2*i]=across;vectors[2*i+1]=up;
        lengths[2*i]=std::max(across.length(),1e-9);lengths[2*i+1]=std::max(up.length(),1e-9);
        angles[i]=across.dot(up)/(lengths[2*i]*lengths[2*i+1]);
    }
}
extern "C" __declspec(dllexport) void weft_gauges(const void* points,int bytes,const int* corners,
        const double* weights,int count,Vec* centers,Vec* vectors,double* lengths,double* angles) {
    if(bytes==8)gauge_samples(static_cast<const double*>(points),corners,weights,count,centers,vectors,lengths,angles);
    else gauge_samples(static_cast<const float*>(points),corners,weights,count,centers,vectors,lengths,angles);
}

// Advance the same exponential response and impose the barycentric hold in
// one traversal. The correction uses the three already advanced grip corners.
extern "C" __declspec(dllexport) void weft_advance_grip(Vec* pos,const Vec* goal,int nv,
        const double* weights,const int* grip,const double* bary,const Vec* target,
        double factor,const double* flank,const Vec* normal) {
    Vec anchor{};
    for(int k=0;k<3;++k){int i=grip[k];anchor=anchor+(pos[i]+(goal[i]-pos[i])*factor)*bary[k];}
    const Vec correction=*target-anchor, lift=*normal*correction.dot(*normal);
    for(int i=0;i<nv;++i) {
        Vec next=pos[i]+(goal[i]-pos[i])*factor+correction*weights[i];
        pos[i]=flank?next+lift*flank[i]:next;
    }
}

extern "C" __declspec(dllexport) void weft_pinched_profile(Vec* goal,const Vec* offset,const double* distance2,
        const double* taper,const int* ids,int count,const int* grip,const Vec* travel,const Vec* direction,
        const Vec* normal,double spread,double length,double lift,double tension,const double* flank,
        double* weights,double* shifts) {
    const double radial2=spread*spread,reach=spread+.75*length,blend=std::min(.64,lift/.036);
    for(int k=0;k<count;++k) {
        int i=ids[k];double forward=std::max(0.,offset[i].dot(*direction));
        double profile=std::max(0.,distance2[i]-forward*forward)/radial2+forward*forward/(reach*reach);
        double influence=std::exp(-.5*profile),radius=std::sqrt(profile);
        if(lift>0.)influence+=blend*(std::exp(-.83*radius)-influence);
        bool held=i==grip[0]||i==grip[1]||i==grip[2];
        double w=held?1.:taper[i]*influence,shift=0.;
        if(!held && lift>0.) {
            double pad=std::max(0.,radius-.0038/spread)/.89;
            shift=taper[i]*std::max(0.,std::exp(-.5*pad*pad)-influence);
        }
        if(flank){shifts[i]=held?0.:tension*(flank[i]-1.)*w;shift+=shifts[i];}
        goal[i]=goal[i]+*travel*w+*normal*(shift*lift);
        weights[i]=w;
    }
}

static double midpoint_median(std::vector<double>& values) {
    auto mid=values.begin()+values.size()/2;
    std::nth_element(values.begin(),mid,values.end());
    if(values.size()%2)return *mid;
    return (.5*(*mid+*std::max_element(values.begin(),mid)));
}
extern "C" __declspec(dllexport) void weft_repair_fans(Vec* points,const Vec* ref,int count,
        const int* seeds,int nseed,int attempt,unsigned char* active) {
    std::vector<int> remaining(seeds,seeds+nseed),group,other;
    for(int i:remaining)active[i]=1;
    while(!remaining.empty()) {
        group.assign(1,remaining.front());remaining.erase(remaining.begin());
        for(int j=0;j<12 && !remaining.empty();++j) {
            Vec center{};for(int i:group)center=center+points[i];center=center*(1./group.size());
            double radius=0.;for(int i:group)radius=std::max(radius,(points[i]-center).length());
            radius+=.004;other.clear();size_t previous=group.size();
            for(int i:remaining) {
                Vec delta=points[i]-center;
                if(delta.dot(delta)<radius*radius)group.push_back(i);else other.push_back(i);
            }
            if(group.size()==previous)break;
            remaining.swap(other);
        }
        Vec x{},y{};std::vector<double> dx,dy,dz;
        for(int i:group) {
            x=x+points[i];y=y+ref[i];Vec d=points[i]-ref[i];
            dx.push_back(d.x);dy.push_back(d.y);dz.push_back(d.z);
        }
        const Vec shift{midpoint_median(dx),midpoint_median(dy),midpoint_median(dz)};
        x=x*(1./group.size());y=y*(1./group.size());
        double rx=0.,ry=0.;
        for(int i:group){rx=std::max(rx,(points[i]-x).length());ry=std::max(ry,(ref[i]-y).length());}
        const double band=.006+.001*attempt,reach=rx+.0008+band;
        for(int i=0;i<count;++i) {
            const Vec a=points[i]-x;
            if(a.dot(a)>reach*reach)continue;
            double distance=std::max(a.length()-rx,(ref[i]-y).length()-ry);
            double t=std::clamp((distance-.0008)/band,0.,1.);
            const double weight=1.-t*t*(3.-2.*t);
            if(weight>0.){active[i]=1;points[i]=points[i]+(shift-(points[i]-ref[i]))*weight;}
        }
    }
}

extern "C" __declspec(dllexport) int weft_tangent_fans(Vec* points,const Vec* ref,const Vec* normals,
        int nv,const Indices* tri,int nf) {
    std::vector<Vec> unit(nf),adjustment(nv);
    std::vector<double> area(nf);std::vector<int> hits(nv),touched;
    for(int j=0;j<nf;++j) {
        auto f=tri[j];Vec n=(ref[f.b]-ref[f.a]).cross(ref[f.c]-ref[f.a]);
        area[j]=n.length();unit[j]=n*(1./std::max(area[j],1e-14));
    }
    bool changed=false;
    for(int it=0;it<24;++it) {
        for(int j=0;j<nf;++j) {
            auto f=tri[j];if(area[j]<=1e-12)continue;
            const int id[]={f.a,f.b,f.c};
            Vec a=points[f.a],b=points[f.b],c=points[f.c],n=unit[j];
            double signed_area=(b-a).cross(c-a).dot(n);
            if(signed_area>=.009*area[j])continue;
            Vec g[]={(b-c).cross(n),(c-a).cross(n),(a-b).cross(n)};
            double den=0.;for(int k=0;k<3;++k){Vec axis=normals[id[k]];g[k]=g[k]-axis*g[k].dot(axis);den+=g[k].dot(g[k]);}
            double amount=(.055*area[j]-signed_area)/std::max(den,1e-16);
            for(int k=0;k<3;++k){int i=id[k];if(!hits[i]++)touched.push_back(i);adjustment[i]=adjustment[i]+g[k]*amount;}
        }
        if(touched.empty())break;
        changed=true;
        for(int i:touched){points[i]=points[i]+adjustment[i]*(1./hits[i]);adjustment[i]={};hits[i]=0;}
        touched.clear();
    }
    return int(changed);
}
