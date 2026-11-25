// SPDX-License-Identifier: MIT
// All interior probes of one garment face share a small skin BVH neighbourhood.
struct ClothProbe {
    Vec origin, direction;
    double first = .014111;
};

static void visit_probe_neighbours(const Index& skin, int node_id, const Bounds& bounds,
                                   ClothProbe* rays, int count) {
    const auto& node=skin.nodes[node_id];
    if (!boxes_touch(node.bounds,bounds)) return;
    if (node.left >= 0) {
        visit_probe_neighbours(skin,node.left,bounds,rays,count);
        visit_probe_neighbours(skin,node.right,bounds,rays,count);
        return;
    }
    for (int i=node.start; i<node.end; ++i) {
        const auto triangle=skin.triangle(skin.order[i]);
        Bounds extent;
        extent.include(triangle.a);extent.include(triangle.b);extent.include(triangle.c);
        if (!boxes_touch(extent,bounds)) continue;
        const Vec a=widen(triangle.a), ab=widen(triangle.b)-a, ac=widen(triangle.c)-a;
        for (int j=0; j<count; ++j) {
            ClothProbe& ray=rays[j];
            const Vec h=ray.direction.cross(ac);
            const double det=ab.dot(h);
            if (std::abs(det)<1e-17) continue;
            const Vec delta=ray.origin-a;
            const double u=delta.dot(h)/det;
            if (u< -2e-8 || u>1.00000002) continue;
            const Vec cross=delta.cross(ab);
            const double v=ray.direction.dot(cross)/det;
            if (v< -2e-8 || u+v>1.00000002) continue;
            const double t=ac.dot(cross)/det;
            if (t>=0. && t<ray.first) ray.first=t;
        }
    }
}

static Point narrow(Vec v) {return {float(v.x),float(v.y),float(v.z)};}

extern "C" __declspec(dllexport) void weft_wet_face_lifts(
        const Index* skin, const Vec* pos, const Vec* normal, const Indices* faces, int count,
        const Vec* bary, int samples, double* lifts) {
    constexpr int block=24;
    parallel_ranges((count+block-1)/block,count>=96?4:1,[&](int,int job) {
        std::vector<ClothProbe> rays(samples);
        for (int i=job*block;i<std::min(count,(job+1)*block);++i) {
            const auto f=faces[i];
            Bounds bounds;
            for (int j=0;j<samples;++j) {
                const Vec b=bary[j];
                const Vec q=pos[f.a]*b.x+pos[f.b]*b.y+pos[f.c]*b.z;
                const Vec n=normal[f.a]*b.x+normal[f.b]*b.y+normal[f.c]*b.z;
                const Vec axis=n*(1./std::max(n.length(),1e-12));
                const Point p=narrow(q+axis*.014),d=narrow(axis*(-1.));
                const float length=std::sqrt(squared(d));
                const Point dir=length>1e-10f?d*(1.f/length):Point{};
                rays[j]={widen(p),widen(dir),.014111};
                bounds.include(p);bounds.include(p+dir*.014111f);
            }
            visit_probe_neighbours(*skin,0,bounds,rays.data(),samples);
            double lift=0.;
            for(const auto& ray:rays) {
                // Match the float32 depth returned by first_hits before the gap test.
                const float gap=float(ray.first)-.014f;
                if(gap<.00011f) lift=std::max(lift,double(std::min(.006f,.00034f-gap)));
            }
            lifts[i]=lift;
        }
    });
}
