// SPDX-License-Identifier: MIT
// Local triangle projections and conservative contact candidates for the dry sheet.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <atomic>
#include <memory>

// Independent skin rays use the Windows worker pool. All Blender data is
// already copied into const arrays; no Blender API is called by a worker.
template<class F> void parallel_ranges(int jobs,int workers,F fun){
    if(workers<2){for(int i=0;i<jobs;i++)fun(0,i);return;}
    struct Work {
        F& f;int jobs;std::atomic<int> next{0},lane{1};
        void run(int who){for(int i;(i=next.fetch_add(1,std::memory_order_relaxed))<jobs;)f(who,i);}
        static void CALLBACK worker(PTP_CALLBACK_INSTANCE,void* ctx,PTP_WORK){
            auto* self=static_cast<Work*>(ctx);
            self->run(self->lane.fetch_add(1,std::memory_order_relaxed));
        }
    } state{fun,jobs};
    PTP_WORK work=CreateThreadpoolWork(&Work::worker,&state,nullptr);
    if(!work){state.run(0);return;}
    for(int i=1;i<workers;i++)SubmitThreadpoolWork(work);
    state.run(0);WaitForThreadpoolWorkCallbacks(work,FALSE);CloseThreadpoolWork(work);
}

struct Vec {
    double x,y,z;
    Vec operator+(Vec b) const {return {x+b.x,y+b.y,z+b.z};}
    Vec operator-(Vec b) const {return {x-b.x,y-b.y,z-b.z};}
    Vec operator*(double v) const {return {x*v,y*v,z*v};}
    double dot(Vec b) const {return x*b.x+y*b.y+z*b.z;}
    Vec cross(Vec b) const {return {y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x};}
    double length() const {return std::sqrt(dot(*this));}
};
template<class T> Vec read3(const T* p, int i){
    return {double(p[3*i]),double(p[3*i+1]),double(p[3*i+2])};
}
template<class T> void add3(T* p,int i,Vec v) {
    p[3*i]=T(p[3*i]+v.x);p[3*i+1]=T(p[3*i+1]+v.y);p[3*i+2]=T(p[3*i+2]+v.z);
}

template<class T>
void project_sheet(T* points,int nv,const int* faces,int nf,const double* lens,const double* ts,
                   const double* heights,const Vec* normals,const T* neutral,const int64_t* slots,const int64_t* adjacent,bool dry_hold=false) {
    std::vector<Vec> delta(nv);
    std::vector<int> hits(nv), touched;
    auto accumulate=[&](int i,Vec v){
        if (!hits[i]++) touched.push_back(i);
        delta[i]=delta[i]+v;
    };
    std::vector<int> scope,seen(nf,-1);int pass=0;
    for(int stage=neutral?2:0;stage<=(neutral?2:1);++stage){
        bool full=true;
        const int rounds=stage==0?10:stage==1?48:dry_hold?24:32;
        for(int it=0;it<rounds;it++){
            int size=full?nf:int(scope.size());
            for(int slot=0;slot<size;++slot){
                const int j=full?slot:scope[slot];
                const int ia=faces[3*j],ib=faces[3*j+1],ic=faces[3*j+2];
                const Vec a=read3(points,ia),b=read3(points,ib),c=read3(points,ic);
                const Vec edge=b-a,n=normals[j];const double h=heights[j],l=lens[j];
                if(stage==0){
                    const double t=ts[j],size=edge.length();Vec lateral=(c-a)-edge*t;
                    const double alt=lateral.length();Vec across=n.cross(edge);
                    across=across*(1./std::max(1e-12,across.length()));
                    if(alt<=2.65*h && lateral.dot(across)>=.13*h && size<=2.65*l) continue;
                    Vec goal=lateral*std::min(1.,2.60*h/std::max(1e-12,alt));
                    goal=goal+across*std::max(0.,.16*h-goal.dot(across));
                    Vec corr=(goal-lateral)*(1./(1.+(1.-t)*(1.-t)+t*t));
                    Vec stretch=edge*(.5*std::max(0.,1.-2.60*l/std::max(1e-12,size)));
                    accumulate(ia,stretch-corr*(1.-t));accumulate(ib,stretch*(-1.)-corr*t);accumulate(ic,corr);
                } else {
                    const double ar=l*h,signed_area=edge.cross(c-a).dot(n);
                    if(dry_hold){
                        const Vec ra=read3(neutral,ia),rb=read3(neutral,ib),rc=read3(neutral,ic);
                        const double unloaded=(rb-ra).cross(rc-ra).dot(n);
                        const double floor=std::min(.035*ar,.35*unloaded);
                        if(unloaded<=1e-14 || signed_area>=floor)continue;
                        Vec shift=(a-ra)*(1.-ts[j])+(b-rb)*ts[j]-(c-rc);
                        const Vec grad=n.cross(edge);
                        const double residual=std::max(0.,2.*floor-signed_area-shift.dot(grad));
                        accumulate(ic,shift+grad*(residual/std::max(1e-20,grad.dot(grad))));
                        continue;
                    }
                    if(signed_area>=(stage==1?.065:.018)*ar) continue;
                    if(stage==1){
                        Vec ga=(b-c).cross(n),gb=(c-a).cross(n),gc=(a-b).cross(n);
                        const double scale=(.16*ar-signed_area)/std::max(1e-15,ga.dot(ga)+gb.dot(gb)+gc.dot(gc));
                        accumulate(ia,ga*scale);accumulate(ib,gb*scale);accumulate(ic,gc*scale);
                    } else {
                        Vec da=a-read3(neutral,ia),db=b-read3(neutral,ib),dc=c-read3(neutral,ic);
                        Vec mean=(da+db+dc)*(1./3.);
                        accumulate(ia,(mean-da)*.85);accumulate(ib,(mean-db)*.85);accumulate(ic,(mean-dc)*.85);
                    }
                }
            }
            if(touched.empty())break;
            // Only incident constraints can change sign in the next sweep.
            // Keep original face order so shared vertices receive the same sums.
            full=int(touched.size())*6>nf;
            scope.clear();++pass;
            for(const int i:touched){
                add3(points,i,delta[i]*(1./hits[i]));delta[i]={};hits[i]=0;
                if(!full)for(int64_t k=slots[i];k<slots[i+1];++k){
                    int face=int(adjacent[k]);
                    if(seen[face]!=pass){seen[face]=pass;scope.push_back(face);}
                }
            }
            if(!full)std::sort(scope.begin(),scope.end());
            touched.clear();
        }
    }
}
extern "C" __declspec(dllexport) void weft_project_sheet(void* points,int bytes,int nv,
        const int* faces,int nf,const double* l,const double* t,const double* h,const Vec* n,const void* neutral,const int64_t* slots,const int64_t* adjacent){
    if(bytes==8)project_sheet(static_cast<double*>(points),nv,faces,nf,l,t,h,n,static_cast<const double*>(neutral),slots,adjacent);
    else project_sheet(static_cast<float*>(points),nv,faces,nf,l,t,h,n,static_cast<const float*>(neutral),slots,adjacent);
}

extern "C" __declspec(dllexport) void weft_project_dry_hold(void* points,int bytes,int nv,
        const int* faces,int nf,const double* l,const double* t,const double* h,const Vec* n,const void* neutral,const int64_t* slots,const int64_t* adjacent){
    if(bytes==8)project_sheet(static_cast<double*>(points),nv,faces,nf,l,t,h,n,static_cast<const double*>(neutral),slots,adjacent,true);
    else project_sheet(static_cast<float*>(points),nv,faces,nf,l,t,h,n,static_cast<const float*>(neutral),slots,adjacent,true);
}

Vec widen(Point p){return {p.x,p.y,p.z};}
bool ray_box(const Bounds& b,Point origin,Point direction,double stop) {
    double start=0.;
    for(int k=0;k<3;k++){
        const double s=origin.axis(k),v=direction.axis(k);
        const double lo=b.lo.axis(k)-2e-7, hi=b.hi.axis(k)+2e-7;
        if(std::abs(v)<1e-14){if(s<lo||s>hi)return false;continue;}
        const double a=(lo-s)/v,c=(hi-s)/v;
        start=std::max(start,std::min(a,c));stop=std::min(stop,std::max(a,c));
        if(start>stop)return false;
    }
    return true;
}
void first_hit(const Index& idx,int id,Point origin,Point direction,double& best,bool& hit){
    const auto& node=idx.nodes[id];if(!ray_box(node.bounds,origin,direction,best))return;
    if(node.left>=0){
        first_hit(idx,node.left,origin,direction,best,hit);
        first_hit(idx,node.right,origin,direction,best,hit);return;
    }
    const Vec d=widen(direction);
    for(int k=node.start;k<node.end;k++){
        const auto f=idx.triangle(idx.order[k]);const Vec a=widen(f.a),ab=widen(f.b)-a,ac=widen(f.c)-a;
        const Vec h=d.cross(ac);const double den=ab.dot(h);
        if(std::abs(den)<1e-17)continue;
        const Vec s=widen(origin)-a;const double u=s.dot(h)/den;
        if(u< -2e-8 || u>1.00000002)continue;
        const Vec q=s.cross(ab);const double v=d.dot(q)/den;
        if(v< -2e-8 || u+v>1.00000002)continue;
        double dist=ac.dot(q)/den;
        if(dist>=0. && dist<=best){best=dist;hit=true;}
    }
}
extern "C" __declspec(dllexport) void weft_first_hits(const Index* idx,const Point* origins,
        const Point* directions,int count,double limit,float* result) {
    constexpr int block=192;
    parallel_ranges((count+block-1)/block,count>=1200?4:1,[&](int,int job){
        for(int j=job*block;j<std::min(count,(job+1)*block);j++){
            Point dir=directions[j];float len=std::sqrt(squared(dir));
            double best=limit;bool hit=false;
            if(len>1e-10)first_hit(*idx,0,origins[j],dir*(1.f/len),best,hit);
            result[j]=hit?float(best):std::numeric_limits<float>::quiet_NaN();
        }
    });
}

// Cull pairs before asking Blender for the exact self-intersection. Keep
// near-coplanar and marginal pairs for Blender's own floating point rules.
struct FaceBox{Bounds bounds; Vec pt[3],edge[3],normal;};
bool separated(const FaceBox& a,const FaceBox& b,Vec axis,double margin){
    double l2=axis.dot(axis);if(l2<1e-34)return false;
    double amin=0.,amax=0.,bmin=1e100,bmax=-1e100;
    for(int i=0;i<3;i++){
        double x=(a.pt[i]-a.pt[0]).dot(axis),y=(b.pt[i]-a.pt[0]).dot(axis);
        amin=std::min(amin,x);amax=std::max(amax,x);bmin=std::min(bmin,y);bmax=std::max(bmax,y);
    }
    const double tol=margin*std::sqrt(l2);
    return amin>bmax+tol || bmin>amax+tol;
}
bool boxes_touch(const Bounds& a,const Bounds& b,float eps=3e-7f){
    return a.lo.x<=b.hi.x+eps && b.lo.x<=a.hi.x+eps && a.lo.y<=b.hi.y+eps &&
        b.lo.y<=a.hi.y+eps && a.lo.z<=b.hi.z+eps && b.lo.z<=a.hi.z+eps;
}
unsigned spread_bits(unsigned v) {
    v=(v | v<<16)&0x030000ffu;v=(v | v<<8)&0x0300f00fu;
    v=(v | v<<4)&0x030c30c3u;return (v | v<<2)&0x09249249u;
}
struct OverlapTree {
    const int *ids;
    const Point *points;
    unsigned char *flags;
    std::vector<FaceBox> faces;
    std::vector<unsigned char> edited_nodes, edited_faces;
    bool restricted=false;
    float margin=3e-7f;
    std::vector<std::pair<unsigned,int>> order;
    std::vector<Node> nodes;
    OverlapTree(const Point* xyz,const int* tiles,int count,unsigned char* out):
        ids(tiles),points(xyz),flags(out),faces(count),order(count) {
        Bounds box;
        for(int j=0;j<count;j++){
            auto& f=faces[j];
            for(int k=0;k<3;k++){Point p=pts(j,k);f.bounds.include(p);f.pt[k]=widen(p);}
            for(int k=0;k<3;k++)f.edge[k]=f.pt[(k+1)%3]-f.pt[k];
            f.normal=f.edge[0].cross(f.edge[1]);box.include(f.bounds);
        }
        const Point span=box.hi-box.lo;
        for(int j=0;j<count;j++){
            const auto& b=faces[j].bounds;Point c=(b.lo+b.hi)*.5f-box.lo;
            unsigned key=0;
            for(int k=0;k<3;k++){
                float t=c.axis(k)/std::max(span.axis(k),1e-10f);
                key|=spread_bits(static_cast<unsigned>(std::clamp(t,0.f,1.f)*1023.f))<<k;
            }
            order[j]={key,j};
        }
        std::sort(order.begin(),order.end());
        nodes.reserve(count);
        build(0,count);
    }
    void refit(const Point* xyz,const int* tiles,unsigned char* out){
        points=xyz;ids=tiles;flags=out;
        for(size_t j=0;j<faces.size();j++){
            auto& f=faces[j];f.bounds=Bounds{};
            for(int k=0;k<3;k++){
                Point p=points[ids[3*j+k]];f.bounds.include(p);f.pt[k]=widen(p);
            }
            for(int k=0;k<3;k++)f.edge[k]=f.pt[(k+1)%3]-f.pt[k];
            f.normal=f.edge[0].cross(f.edge[1]);
        }
        for(auto it=nodes.rbegin();it!=nodes.rend();++it){
            Node& n=*it;
            if(n.left>=0){
                n.bounds=nodes[n.left].bounds;n.bounds.include(nodes[n.right].bounds);
            }else{
                n.bounds=Bounds{};
                for(int j=n.start;j<n.end;j++)n.bounds.include(faces[order[j].second].bounds);
            }
        }
    }
    void restrict_to(const unsigned char* active){
        restricted=active!=nullptr;
        if(!restricted)return;
        edited_faces.resize(faces.size());edited_nodes.resize(nodes.size());
        for(size_t i=0;i<faces.size();++i)
            edited_faces[i]=active[ids[3*i]] || active[ids[3*i+1]] || active[ids[3*i+2]];
        for(size_t i=nodes.size();i-->0;){
            const Node& n=nodes[i];
            if(n.left>=0)edited_nodes[i]=edited_nodes[n.left] || edited_nodes[n.right];
            else{
                bool edited=false;
                for(int j=n.start;j<n.end&&!edited;++j)edited=edited_faces[order[j].second]!=0;
                edited_nodes[i]=edited;
            }
        }
    }
    Point pts(int j,int k) const {return points[ids[3*j+k]];}
    int build(int first,int end){
        const int id=int(nodes.size());nodes.push_back({{},first,end});
        if(end-first<=4){
            for(int i=first;i<end;i++)nodes[id].bounds.include(faces[order[i].second].bounds);
        } else {
            const int mid=(first+end)/2;const int left=build(first,mid),right=build(mid,end);
            nodes[id].left=left;nodes[id].right=right;
            nodes[id].bounds=nodes[left].bounds;nodes[id].bounds.include(nodes[right].bounds);
        }
        return id;
    }
    void check(int ai,int bi,unsigned char* flags){
        if(flags[ai]&&flags[bi])return;
        if(restricted&&!edited_faces[ai]&&!edited_faces[bi])return;
        const auto& a=faces[ai];const auto& b=faces[bi];
        if(!boxes_touch(a.bounds,b.bounds,margin))return;
        for(int u=0;u<3;u++)for(int v=0;v<3;v++)if(ids[3*ai+u]==ids[3*bi+v])return;
        if(separated(a,b,a.normal,margin)||separated(a,b,b.normal,margin))return;
        for(int u=0;u<3;u++)for(int v=0;v<3;v++)if(separated(a,b,a.edge[u].cross(b.edge[v]),margin))return;
        // Almost coplanar nylon triangles also have separating axes in
        // their own plane. Discard only projections with a positive margin.
        for(int u=0;u<3;u++){
            if(separated(a,b,a.normal.cross(a.edge[u]),margin) ||
               separated(a,b,b.normal.cross(b.edge[u]),margin))return;
        }
        flags[ai]=flags[bi]=1;
    }
    void pairs(int ia,int ib,unsigned char* out){
        if(restricted&&!edited_nodes[ia]&&!edited_nodes[ib])return;
        const Node& a=nodes[ia];const Node& b=nodes[ib];
        if(ia!=ib&&!boxes_touch(a.bounds,b.bounds,margin))return;
        if(a.left<0 && b.left<0){
            for(int i=a.start;i<a.end;i++)for(int j=(ia==ib?i+1:b.start);j<b.end;j++)
                check(order[i].second,order[j].second,out);
        } else if(ia==ib){
            pairs(a.left,a.left,out);pairs(a.left,a.right,out);pairs(a.right,a.right,out);
        } else if(a.left>=0 && (b.left<0 || a.end-a.start>=b.end-b.start)){
            pairs(a.left,ib,out);pairs(a.right,ib,out);
        } else {pairs(ia,b.left,out);pairs(ia,b.right,out);}
    }
};
static void find_overlap_candidates(const Point* pts,const int* ids,int nf,const unsigned char* active,unsigned char* flags,float margin=3e-7f){
    if(!nf)return;
    if(nf<1200){OverlapTree small(pts,ids,nf,flags);small.margin=margin;small.restrict_to(active);small.pairs(0,0,flags);return;}
    // Each caller owns a refittable broadphase. Check connectivity by value:
    // Blender and NumPy can recycle an address after a cut or a new grip.
    thread_local std::vector<int> topology;
    thread_local std::unique_ptr<OverlapTree> cached;
    if(topology.size()!=size_t(3*nf)||!std::equal(topology.begin(),topology.end(),ids)){
        topology.assign(ids,ids+3*nf);
        cached=std::make_unique<OverlapTree>(pts,ids,nf,flags);
    }else{
        cached->refit(pts,ids,flags);
    }
    OverlapTree& tree=*cached;tree.margin=margin;
    tree.restrict_to(active);
    // Split along the same tree. Each worker owns its candidate bitmap.
    std::vector<std::pair<int,int>> jobs;
    auto partition=[&](auto&& self,int ia,int ib)->void{
        if(tree.restricted&&!tree.edited_nodes[ia]&&!tree.edited_nodes[ib])return;
        const Node& a=tree.nodes[ia];const Node& b=tree.nodes[ib];
        if(ia!=ib&&!boxes_touch(a.bounds,b.bounds,tree.margin))return;
        if(std::max(a.end-a.start,b.end-b.start)<200){jobs.emplace_back(ia,ib);return;}
        if(ia==ib){self(self,a.left,a.left);self(self,a.left,a.right);self(self,a.right,a.right);}
        else if(a.end-a.start>=b.end-b.start){self(self,a.left,ib);self(self,a.right,ib);}
        else{self(self,ia,b.left);self(self,ia,b.right);}
    };
    partition(partition,0,0);
    std::vector<unsigned char> bits(4*nf);
    parallel_ranges(int(jobs.size()),4,[&](int lane,int j){
        tree.pairs(jobs[j].first,jobs[j].second,bits.data()+lane*nf);
    });
    for(int j=0;j<nf;j++)flags[j]=bits[j]|bits[nf+j]|bits[2*nf+j]|bits[3*nf+j];
}
extern "C" __declspec(dllexport) void weft_overlap_candidates(const Point* pts,const int* ids,int nf,unsigned char* flags){
    find_overlap_candidates(pts,ids,nf,nullptr,flags);
}
extern "C" __declspec(dllexport) void weft_overlap_changed_candidates(const Point* pts,const int* ids,int nf,const unsigned char* active,unsigned char* flags){
    find_overlap_candidates(pts,ids,nf,active,flags);
}

extern "C" __declspec(dllexport) void weft_close_courses(const Point* pts,const int* faces,int nf,float margin,unsigned char* flags){
    find_overlap_candidates(pts,faces,nf,nullptr,flags,margin);
}
