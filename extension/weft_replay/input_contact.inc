// SPDX-License-Identifier: MIT

// Build the exact edge and corner indices once, before tagging a new draw mesh.
extern "C" __declspec(dllexport) int weft_mesh_edges(const int* tri,int nv,int nf,int* edges,int* corners) {
    std::vector<int> heads(nv,-1),next;next.reserve(2*nf);int count=0;
    for(int f=0;f<nf;++f)for(int k=0;k<3;++k) {
        int a=tri[3*f+k],b=tri[3*f+(k+1)%3];
        if(a>b)std::swap(a,b);
        int slot=heads[a];
        while(slot>=0 && edges[2*slot+1]!=b)slot=next[slot];
        if(slot<0) {
            slot=count++;edges[2*slot]=a;edges[2*slot+1]=b;
            next.push_back(heads[a]);heads[a]=slot;
        }
        corners[3*f+k]=slot;
    }
    return count;
}



extern "C" __declspec(dllexport) void weft_buckle_patch(Vec* goal,const int* ids,int nv,const Indices* tri,int nf,
        const Vec* grad_a,const Vec* grad_b,const Vec* rest_a,const Vec* rest_b,const double* area,
        const double* corner,const Vec* offsets,const Vec* normal,const double* distance,const double* weights,
        const Vec* along,const Vec* across,const Vec* up,const Vec* center,double travel,double skew) {
    std::vector<double> stress(nv);
    for(int j=0;j<nf;++j){
        auto f=tri[j];double u=grad_a[j].dot(*along),v=grad_b[j].dot(*along);
        Vec a=goal[ids[f.a]],b=goal[ids[f.b]],c=goal[ids[f.c]];
        double lost=(rest_a[j]*u+rest_b[j]*v).length()-((b-a)*u+(c-a)*v).length()-.035;
        double w=std::clamp(lost,0.,.48)*area[j];
        stress[f.a]+=w;stress[f.b]+=w;stress[f.c]+=w;
    }
    constexpr double pi=3.141592653589793;
    double scale=.008+.13*std::min(travel,.036),width=.011+.15*std::min(travel,.045);
    for(int i=0;i<nv;++i) {
        if(stress[i]<=0.)continue;
        double a=offsets[i].dot(*along),b=offsets[i].dot(*across),strip=b+skew*a;
        double phase=(a+skew*b+.0022*std::sin(strip/.011))/scale;
        double envelope=std::exp(-.5*strip*strip/(width*width));
        double support=std::clamp((distance[i]-.0035)/.007,0.,1.)*std::clamp((.049-distance[i])/.016,0.,1.);
        double lift=.46*scale/pi*std::sqrt(stress[i]/corner[i])*(1.-std::cos(2*pi*phase))
             *support*envelope*std::sqrt(std::clamp(weights[ids[i]],0.,1.))*std::clamp(normal[i].dot(*up),0.,1.);
        if(center){Vec d=offsets[i]-*center;lift*=1.-.76*std::exp(-.5*d.dot(d)/(.029*.029));}
        goal[ids[i]]=goal[ids[i]]+normal[i]*lift;
    }
}

// Fixed-pose contact queries read an immutable skin index.
extern "C" __declspec(dllexport) void weft_patch_planes(const Index* skin,const Vec* rest,const Vec* axes,
        int count,Vec* anchors,Vec* normals,signed char* status) {
    constexpr int block=64;
    parallel_ranges((count+block-1)/block,count>=384?4:1,[&](int,int job) {
        for(int i=job*block;i<std::min(count,(job+1)*block);++i) {
            const Point p=narrow(rest[i]);Hit hit;skin->query(0,p,0.f,hit);
            if(hit.index<0 || hit.next<=hit.best+4.f*std::numeric_limits<float>::epsilon()*hit.best) {
                status[i]=-1;continue; // The Blender tree decides the first tied face.
            }
            float dist=std::sqrt(hit.best);Point face=skin->triangle(hit.index).normal();
            const Vec n=widen(face);
            if(dist>.004f || n.dot(axes[i])<.65)continue;
            const float range=dist+.000002f;
            if(skin->opposing(0,p,face,range*range)){status[i]=-1;continue;}
            const double gap=std::max(0.,(rest[i]-widen(hit.point)).dot(n)-.000075);
            anchors[i]=rest[i]-n*gap;normals[i]=n;status[i]=1;
        }
    });
}

extern "C" __declspec(dllexport) void weft_project_grab(const Index* skin,Vec* pos,const Vec* rest,
        const Vec* seating,const Vec* normals,const int* grip,int count,Vec* tested,unsigned char* valid) {
    std::vector<int> moved;moved.reserve(count/6);
    for(int i=0;i<count;++i) {
        const Vec delta=pos[i]-rest[i]-seating[i];
        if(delta.dot(delta)<=1e-8 || (grip && (i==grip[0]||i==grip[1]||i==grip[2])))continue;
        if(valid[i] && pos[i].x==tested[i].x && pos[i].y==tested[i].y && pos[i].z==tested[i].z)continue;
        moved.push_back(i);
    }
    const int n=int(moved.size()),block=64;
    parallel_ranges((n+block-1)/block,n>=384?4:1,[&](int,int job) {
        for(int k=job*block;k<std::min(n,(job+1)*block);++k) {
            int i=moved[k];Vec axis=normals[i];
            Point p=narrow(pos[i]+axis*.012),dir=narrow(axis*(-1));
            float length=std::sqrt(squared(dir));bool found=false;double best=.01265;
            if(length>1e-10f)first_hit(*skin,0,p,dir*(1.f/length),best,found);
            if(found)pos[i]=pos[i]+axis*double(.01265f-float(best));
            tested[i]=pos[i];valid[i]=1;
        }
    });
}

extern "C" __declspec(dllexport) void weft_material_normals(const Vec* points,const Indices* tri,
        int nv,int nf,Vec* output) {
    for(int j=0;j<nf;++j){
        auto f=tri[j];Vec cross=(points[f.b]-points[f.a]).cross(points[f.c]-points[f.a]);
        for(int v:{f.a,f.b,f.c})output[v]=output[v]+cross;
    }
    for(int i=0;i<nv;++i)output[i]=output[i]*(1./std::max(output[i].length(),1e-12));
}

extern "C" __declspec(dllexport) void weft_face_probe_groups(const Vec* points,const Indices* tri,int nf,int* groups) {
    for(int j=0;j<nf;++j) {
        auto f=tri[j];Vec a=points[f.a],b=points[f.b],c=points[f.c];
        Vec x=b-a,y=c-b,z=a-c;double size=std::max({x.dot(x),y.dot(y),z.dot(z)});
        groups[j]=size>.018*.018?7:size>.009*.009?4:size>.004*.004?2:0;
    }
}
