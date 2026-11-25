// SPDX-License-Identifier: MIT
// Batch closest-triangle queries for the dry contact solver.
// The triangle region tests follow the Ericson formulation used by Blender.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {
struct Point {
    float x, y, z;
    float axis(int i) const { return i == 0 ? x : i == 1 ? y : z; }
    Point operator+(Point b) const { return {x+b.x, y+b.y, z+b.z}; }
    Point operator-(Point b) const { return {x-b.x, y-b.y, z-b.z}; }
    Point operator*(float s) const { return {x*s, y*s, z*s}; }
};

static_assert(sizeof(Point) == 3*sizeof(float));

float dot(Point a, Point b) { return (a.x*b.x + a.y*b.y) + a.z*b.z; }
float squared(Point a) { return dot(a, a); }

struct Triangle {
    Point a, b, c;
    Point closest(Point p) const {
        const Point ab = b-a, ac = c-a, ap = p-a, bp = p-b;
        const float d1 = dot(ab,ap), d2 = dot(ac,ap);
        if (d1 <= 0 && d2 <= 0) return a;
        const float d3 = dot(ab,bp), d4 = dot(ac,bp);
        if (d3 >= 0 && d4 <= d3) return b;
        const float vc = d1*d4 - d3*d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
            return d1 == d3 ? a : a + ab*(d1/(d1-d3));
        }
        const Point cp = p-c;
        const float d5 = dot(ab,cp), d6 = dot(ac,cp);
        if (d6 >= 0 && d5 <= d6) return c;
        const float vb = d5*d2 - d1*d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            return d2 == d6 ? a : a + ac*(d2/(d2-d6));
        }
        const float va = d3*d6 - d5*d4;
        const float db = d4-d3, dc = d5-d6;
        if (va <= 0 && db >= 0 && dc >= 0) {
            return db+dc == 0 ? b : b + (c-b)*(db/(db+dc));
        }
        const float recipro = 1.f/((va+vb)+vc);
        return (a + ab*(vb*recipro)) + ac*(vc*recipro);
    }

    Point normal() const {
        const Point ab = a-b, bc = b-c;
        const Point n{ab.y*bc.z-ab.z*bc.y, ab.z*bc.x-ab.x*bc.z, ab.x*bc.y-ab.y*bc.x};
        const float norm = squared(n);
        return norm > 1e-35f ? n*(1.f/std::sqrt(norm)) : Point{};
    }
};

struct Indices { std::int32_t a,b,c; };
constexpr float inf = std::numeric_limits<float>::infinity();

struct Bounds {
    Point lo{inf,inf,inf}, hi{-inf,-inf,-inf};
    void include(Point p) {
        lo = {std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
        hi = {std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
    }
    void include(const Bounds& other) { include(other.lo); include(other.hi); }
    float lower_bound(Point p) const {
        return squared({std::max({0.f,lo.x-p.x,p.x-hi.x}),
                        std::max({0.f,lo.y-p.y,p.y-hi.y}),
                        std::max({0.f,lo.z-p.z,p.z-hi.z})});
    }
};

struct Node {
    Bounds bounds;
    int start, end, left = -1, right = -1;
    int vertices_begin = 0, vertices_end = 0;
};

struct Hit {
    Point point{};
    int index = -1;
    float best = inf, next = inf, bound = inf;
};

struct Index {
    std::vector<Indices> triangles;
    std::vector<int> order;
    std::vector<Node> nodes;
    std::vector<int> leaf_vertices;
    const Point* points = nullptr;
    float guard;

    Index(const Point* vertices, int nverts, const Indices* ids, int count)
        : triangles(ids,ids+count), order(count) {
        for (int i=0;i<count;i++) {
            const auto t = ids[i];
            if (std::min({t.a,t.b,t.c}) < 0 || std::max({t.a,t.b,t.c}) >= nverts) throw count;
        }
        std::vector<Point> centers(count);
        for (int i=0;i<count;i++) {
            const auto t = ids[i];
            centers[i] = (vertices[t.a]+vertices[t.b]+vertices[t.c])*(1.f/3.f);
        }
        std::iota(order.begin(),order.end(),0);
        nodes.reserve(count/3+1);
        split(0,count,centers);
        refit(vertices);
    }

    Triangle triangle(int id) const {
        const auto t = triangles[id];
        return {points[t.a],points[t.b],points[t.c]};
    }

    int split(int lo, int hi, const std::vector<Point>& centers) {
        const int index = int(nodes.size());
        nodes.push_back({{},lo,hi});
        if (hi-lo <= 12) {
            int buffer[36], count=0;
            for (int i=lo;i<hi;i++) {
                const auto t=triangles[order[i]];
                buffer[count++]=t.a;
                buffer[count++]=t.b;
                buffer[count++]=t.c;
            }
            std::sort(buffer,buffer+count);
            const auto end=std::unique(buffer,buffer+count);
            nodes[index].vertices_begin=int(leaf_vertices.size());
            leaf_vertices.insert(leaf_vertices.end(),buffer,end);
            nodes[index].vertices_end=int(leaf_vertices.size());
            return index;
        }
        Bounds span;
        for (int i=lo;i<hi;i++) span.include(centers[order[i]]);
        const Point size = span.hi-span.lo;
        int axis = size.x >= size.y ? 0 : 1;
        if (size.z > size.axis(axis)) axis = 2;
        const int mid = (lo+hi)/2;
        std::nth_element(order.begin()+lo,order.begin()+mid,order.begin()+hi,
                         [&](int a,int b) { return centers[a].axis(axis) < centers[b].axis(axis); });
        const int left = split(lo,mid,centers), right = split(mid,hi,centers);
        nodes[index].left = left;
        nodes[index].right = right;
        return index;
    }

    void refit(const Point* vertices) {
        points = vertices;
        for (auto it=nodes.rbegin();it!=nodes.rend();++it) {
            Bounds b;
            if (it->left < 0) {
                for (int i=it->vertices_begin;i<it->vertices_end;i++) b.include(points[leaf_vertices[i]]);
            } else {
                b.include(nodes[it->left].bounds);
                b.include(nodes[it->right].bounds);
            }
            it->bounds=b;
        }
        const auto box=nodes[0].bounds;
        guard = 2e-6f*std::max({1.f, std::fabs(box.lo.x), std::fabs(box.lo.y), std::fabs(box.lo.z),
                              std::fabs(box.hi.x), std::fabs(box.hi.y), std::fabs(box.hi.z)});
    }

    float swept_limit(float best) const {
        return best + 2.f*guard*std::sqrt(best) + guard*guard;
    }

    void query(int id, Point p, float lower, Hit& hit) const {
        if (lower > std::min(hit.next,hit.bound)) return;
        const auto& node=nodes[id];
        if (node.left < 0) {
            for (int i=node.start;i<node.end;i++) {
                const int face=order[i];
                const auto q=triangle(face).closest(p);
                const float d=squared(p-q);
                if (d < hit.best) {
                    hit.next=hit.best;
                    hit.best=d;
                    hit.bound=swept_limit(d);
                    hit.index=face;
                    hit.point=q;
                } else if (d < hit.next) {
                    hit.next=d;
                }
            }
            return;
        }
        int a=node.left, b=node.right;
        float d1=nodes[a].bounds.lower_bound(p), d2=nodes[b].bounds.lower_bound(p);
        if (d2 < d1) { std::swap(a,b); std::swap(d1,d2); }
        query(a,p,d1,hit);
        query(b,p,d2,hit);
    }

    bool opposing(int id, Point p, Point normal, float radius2) const {
        const auto& node=nodes[id];
        if (node.bounds.lower_bound(p)>radius2) return false;
        if (node.left >= 0) {
            return opposing(node.left,p,normal,radius2) || opposing(node.right,p,normal,radius2);
        }
        for (int i=node.start;i<node.end;i++) {
            const auto t=triangle(order[i]);
            if (squared(p-t.closest(p))<=radius2 && dot(t.normal(),normal)<-.799f) return true;
        }
        return false;
    }

    bool clear(int id, Point p, float radius2, float threshold, float opposing2) const {
        const auto& node=nodes[id];
        if (node.bounds.lower_bound(p)>radius2) return true;
        if (node.left >= 0) {
            return clear(node.left,p,radius2,threshold,opposing2) && clear(node.right,p,radius2,threshold,opposing2);
        }
        for (int i=node.start;i<node.end;i++) {
            const auto t=triangle(order[i]);
            const Point q=t.closest(p), delta=p-q;
            if (squared(delta)>radius2) continue;
            const Point normal=t.normal();
            if (dot(delta,normal)<threshold || (opposing2>0 && opposing(0,q,normal,opposing2))) return false;
        }
        return true;
    }
};

struct Result { Point point, normal; float distance, signed_gap; };
static_assert(sizeof(Result) == 8*sizeof(float));
} // namespace

#include "sheet_kernels.inc"
#include "wet_transport.inc"
#include "wet_face_contact.inc"
#include "dry_drag.inc"
#include "input_contact.inc"

extern "C" __declspec(dllexport) Index* weft_index_new(const Point* points, int nverts,
                                                      const Indices* ids, int count) {
    try { return count > 0 && nverts > 0 ? new Index(points,nverts,ids,count) : nullptr; }
    catch (...) { return nullptr; }
}

extern "C" __declspec(dllexport) void weft_index_delete(Index* index) { delete index; }
extern "C" __declspec(dllexport) void weft_index_refit(Index* index,const Point* points) {
    index->refit(points);
}

extern "C" __declspec(dllexport) void weft_index_nearest(const Index* index, const Point* queries,
                                                       int count, int check_faces, const float* clear_at, Result* results,
                                                       std::int32_t* faces) {
    for (int i=0;i<count;i++) {
        const Point p=queries[i];
        Hit hit;
        index->query(0,p,0.f,hit);
        faces[i]=-1;
        if (hit.index<0) continue;
        const Point normal=index->triangle(hit.index).normal();
        if (squared(normal)<.5f) continue;
        // Spatial pruning has a larger guard than the float comparison of two
        // squared distances already evaluated with identical triangle math.
        const bool tied=hit.next<=hit.best+4.f*std::numeric_limits<float>::epsilon()*hit.best;
        const float gap=dot(p-hit.point,normal);
        const float range=.000010f+2.f*index->guard;
        const float opposing2=check_faces ? range*range : 0.f;
        if (clear_at) {
            const float threshold=clear_at[i]+index->guard;
            if (gap>=threshold) {
                const bool clear = tied ? index->clear(0,p,index->swept_limit(hit.best),threshold,opposing2)
                                        : !(check_faces && index->opposing(0,hit.point,normal,opposing2));
                if (clear) {
                    faces[i]=-2;
                    continue;
                }
            }
        }
        if (tied) continue;
        if (check_faces && index->opposing(0,hit.point,normal,opposing2)) continue;
        faces[i]=hit.index;
        results[i]={hit.point,normal,std::sqrt(hit.best),gap};
    }
}
