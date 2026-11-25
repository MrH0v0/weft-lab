"""Local finite-volume liquid model for Weft's evaluated sock and skin.

Metres, seconds, mL. The pulse and film closures follow splash.mjs and
cloth.cpp. Each lattice owns its inventory in that surface's material frame.
"""
from dataclasses import dataclass

import numpy as np

if __package__:
    from .contact_index import transport_film
else:
    transport_film=None


HZ = 60
DT = 1 / HZ
TRANSPORT_SUBSTEPS = 12
PULSE = .078
ROUGHNESS = .00008
AREAL_MASS = .035
FRONT_MM = .05
GRAVITY = np.array((0., 0., -9.81))
WATER_KINEMATIC_VISCOSITY = 1e-6


@dataclass(frozen=True)
class Liquid:
    volume: float = 2.
    radius: float = .012
    viscosity: float = 1.
    tension: float = .65
    uptake: float = 1.
    drying: float = 1.
    wet_gain: float = 1.
    color: tuple = (.19, .59, .82)
    opacity: float = .8


@dataclass(frozen=True)
class KnitWetting:
    """Uncalibrated thin-stockinette parameters in SI, except capacity in mL/m²."""
    thickness_m: float = .00015
    solid_density_kg_m3: float = 1120.
    retained_pore_fraction: float = .6
    uptake_rate_s: float = 5.
    diffusivity_m2_s: tuple = (4e-6, 9e-6)  # across / along the local leg chart

    @property
    def capacity_ml_m2(self):
        pore_depth = max(0., self.thickness_m - AREAL_MASS / self.solid_density_kg_m3)
        return 1e6 * self.retained_pore_fraction * pore_depth


KNIT_WETTING = KnitWetting()


def pulse(age):
    t = np.clip(age / PULSE, 0., 1.)
    return t - np.sin(2 * np.pi * t) / (2 * np.pi)


@dataclass
class Pour:
    origin: np.ndarray
    velocity: np.ndarray
    flight: np.ndarray
    delay: np.ndarray
    shares: np.ndarray
    sites: np.ndarray
    weights: np.ndarray
    tangent: np.ndarray
    volume: float
    age: float = 0.

    def points(self, age):
        t = np.asarray(age)[..., None]
        return self.origin + t * self.velocity + .5 * t * t * GRAVITY


class Film:
    def __init__(self, size, pitch, valid, area, skin_area, liquid):
        rows, columns = (size, size) if isinstance(size, int) else size
        self.shape, self.pitch, self.liquid = (rows, columns), pitch, liquid
        self.sock_valid, self.skin_valid = (mask.copy() for mask in valid)
        n = rows * columns
        index = np.arange(n).reshape(rows, columns)
        self.i = index.ravel()
        padded = np.pad(index, 1, constant_values=-1)
        self.nb = np.column_stack([padded[y:y+rows, x:x+columns].ravel()
                                   for y, x in ((1, 2), (1, 0), (2, 1), (0, 1))])
        self.transport_neighbours=np.ascontiguousarray(self.nb,dtype=np.int32)
        self.src = np.repeat(self.i, 4)
        self.dst = self.nb.ravel().clip(min=0)
        self.open = self.nb >= 0
        self.capacity = area * KNIT_WETTING.capacity_ml_m2
        self.area, self.skin_area = area, skin_area
        self.outer, self.knit, self.skin = np.zeros((3, n))
        self.momentum = np.zeros((n, 3))
        self.pours = []
        self.dosed = self.out = self.evap = 0.
        self.outer_runoff = self.downslope_runoff = self.outer_to_knit = 0.
        self.first_contact = None
        self.step = 0
        self.flow = np.zeros(3)
        self.routed_ml = 0.

    def add_pour(self, pour):
        self.pours.append(pour)
        self.dosed += pour.volume

    def dose(self, side, distance):
        if side == 'dry':
            return
        valid, target = ((self.skin_valid, self.skin) if side == 'skin' else
                         (self.sock_valid, self.outer))
        w = np.maximum(0., 1. - (distance / self.liquid.radius)**2)**2 * valid
        if not w.sum():
            raise ValueError('加水范围内没有有效表面')
        target += w * (self.liquid.volume / w.sum())
        self.dosed += self.liquid.volume

    def remove_sock_sites(self, valid):
        lost = self.sock_valid & ~valid
        self.out += (self.outer[lost] + self.knit[lost]).sum()
        self.outer[lost] = self.knit[lost] = 0.
        self.momentum[lost] = 0.
        self.sock_valid = valid.copy()

    def depth(self, reservoir, area):
        return reservoir * 1e-6 / area

    def inlet(self):
        pending = []
        for p in self.pours:
            a = p.age - p.delay - p.flight
            p.age += DT
            q = (pulse(a + DT) - pulse(a)) * p.shares * p.volume
            indices = p.sites.clip(min=0)
            hit = (p.sites >= 0) & self.sock_valid[indices]
            delivery = q[:, None] * p.weights
            self.out += delivery[~hit].sum()
            np.add.at(self.outer, indices[hit], delivery[hit])
            momentum = delivery[..., None] * p.tangent[:, None, :]
            np.add.at(self.momentum, indices[hit], momentum[hit])
            if self.first_contact is None and delivery[hit].sum() > 1e-12:
                self.first_contact = self.step
            if np.any(a + DT < PULSE):
                pending.append(p)
        self.pours = pending

    def move(self, value, valid, volume_flux, vectors=None):
        receiver = self.open & valid[self.nb.clip(min=0)]
        flux = volume_flux.ravel()
        incoming = (volume_flux * receiver).ravel()
        value += np.bincount(self.dst, incoming, minlength=value.size) - volume_flux.sum(axis=1)
        self.out += (flux - incoming).sum()
        if vectors is not None:
            for k in range(3):
                self.momentum[:, k] += (np.bincount(self.dst, incoming * vectors[self.src, k],
                                                     minlength=value.size)
                                         - volume_flux.sum(axis=1) * vectors[:, k])

    def outer_geometry(self, positions):
        nb = self.nb.clip(min=0)
        connected = self.open & self.sock_valid[nb]
        edge = positions[nb] - positions[:, None, :]
        # At an open edge, continue along the tangent from the last inside face.
        # The outlet has no imposed suction across the numerical boundary.
        opposite = [1, 0, 3, 2]
        d = np.where(connected[..., None], edge, -edge[:, opposite])
        length = np.maximum(self.pitch*.25, np.linalg.norm(d, axis=2))
        width = np.where(connected, (self.area[:, None] + self.area[nb])*.5,
                         self.area[:, None]) / length
        width *= self.sock_valid[:, None] & (connected | connected[:, opposite])
        return connected, d, length, width

    def transport(self, geometry, dt):
        p = self.liquid
        nb = self.nb.clip(min=0)
        connected, d, length, width = geometry
        direction = d / length[..., None]
        depth = self.depth(self.outer, self.area)
        before = self.outer.copy()
        # A thicker contacting film wets empty pores faster while the pore
        # volume stays bounded. Do this before routing liquid across the face.
        empty = np.maximum(0., self.capacity - self.knit)
        intake = KNIT_WETTING.uptake_rate_s * (1 + 2*depth / (depth + .00035))
        absorbed = np.minimum(self.outer, empty * -np.expm1(-intake*p.uptake*dt/np.sqrt(p.viscosity)))
        self.knit += absorbed
        self.outer -= absorbed
        self.outer_to_knit += absorbed.sum()
        self.momentum *= (self.outer / np.maximum(before, 1e-12))[:, None]

        depth = self.depth(self.outer, self.area)
        mm = 1000 * depth
        velocity = self.momentum / np.maximum(self.outer[:, None], 1e-12)
        velocity /= (1 + dt * p.viscosity * (8 + 4 / np.maximum(.015, mm**2)))[:, None]
        self.momentum[:] = velocity * self.outer[:, None]
        mass = self.outer[:, None] + self.outer[nb]
        face_velocity = (self.momentum[:, None, :] + self.momentum[nb]) / np.maximum(mass[..., None], 1e-12)
        face_velocity = np.where(connected[..., None], face_velocity, velocity[:, None, :])
        impact = np.sum(face_velocity * direction, axis=2)
        neighbour_depth = np.where(connected, depth[nb], depth[:, None])
        slope = (-d[..., 2] + (depth[:, None] - neighbour_depth)*(1-.70*p.tension)) / length
        h = np.where(slope >= 0., depth[:, None], neighbour_depth)
        # Laminar depth-averaged gravity/current-head velocity, transitioning
        # to a gravity-wave speed scale for deep and steep transient pools.
        drive = -GRAVITY[2] * h**2 * slope / (3*WATER_KINEMATIC_VISCOSITY*p.viscosity)
        wave = np.sqrt(-2*GRAVITY[2]*h)
        hydraulic = drive / (1 + np.abs(drive)/np.maximum(wave, 1e-12))
        flux = np.maximum(0., impact+hydraulic) * depth[:, None] * width * 1e6
        q = flux * np.minimum(dt, .85*self.outer / np.maximum(flux.sum(axis=1), 1e-30))[:, None]
        self.flow += np.sum((q*connected)[..., None]*d, axis=(0, 1))
        escaped = q * ~connected
        self.outer_runoff += escaped.sum()
        self.downslope_runoff += escaped[d[..., 2] < -self.pitch*.1].sum()
        self.move(self.outer, self.sock_valid, q, velocity)
        before = self.outer.copy()
        for reservoir, rate in ((self.outer, .9), (self.knit, .55)):
            evap = np.minimum(reservoir, self.area * p.drying * rate * dt)
            self.evap += evap.sum()
            reservoir -= evap
        self.momentum *= (self.outer / np.maximum(before, 1e-12))[:, None]

    def wick(self, positions):
        """Move retained liquid between intact pores driven by saturation gradient."""
        p = self.liquid
        nb = self.nb.clip(min=0)
        valid = self.open & self.sock_valid[:, None] & self.sock_valid[nb]
        uphill = self.knit / np.maximum(self.capacity, 1e-12)
        gradient = np.maximum(0., uphill[:, None] - uphill[nb])
        delta = positions[nb] - positions[:, None, :]
        length_squared = np.maximum((self.pitch*.25)**2, np.sum(delta**2, axis=2))
        across, along = KNIT_WETTING.diffusivity_m2_s
        diffusivity = np.array((across, across, along, along)) / p.viscosity
        flux = (diffusivity * DT * gradient * np.minimum(self.capacity[:, None], self.capacity[nb])
                / length_squared * valid)
        flux *= np.minimum(1., self.knit / np.maximum(flux.sum(axis=1), 1e-30))[:, None]
        demand = np.bincount(self.dst, flux.ravel(), minlength=len(self.knit))
        empty = np.maximum(0., self.capacity - self.knit)
        flux *= np.minimum(1., empty / np.maximum(demand, 1e-30))[nb]
        self.move(self.knit, self.sock_valid, flux)

    def exchange(self, sites, gaps):
        p = self.liquid
        mapped = sites >= 0
        j = sites.clip(min=0)
        depth = self.depth(self.skin, self.skin_area)
        touching = mapped & self.sock_valid & self.skin_valid[j] & (gaps <= depth[j] + .00001)
        available = np.full(self.skin.size, np.inf)
        np.minimum.at(available, j[mapped & self.sock_valid],
                      np.maximum(0., gaps[mapped & self.sock_valid]))
        transfer = (self.skin[j] / self.skin_area[j] - .22 * self.knit / self.area)
        transfer *= self.area * .26 * p.uptake * DT * touching
        forward = np.minimum(np.maximum(0., transfer), np.maximum(0., self.capacity - self.knit))
        reverse = np.minimum(np.maximum(0., -transfer), self.knit)
        required = np.bincount(j, forward, minlength=self.skin.size)
        forward *= np.minimum(1., self.skin[j] / np.maximum(required[j], 1e-30))
        q = forward - reverse
        self.knit += q
        self.skin -= np.bincount(j, q, minlength=self.skin.size)
        depth = self.depth(self.skin, self.skin_area)
        excess = np.maximum(0., depth - available)
        mobility = depth / (depth + ROUGHNESS)
        rate = mobility**3 / (.035 * p.viscosity)
        head = np.maximum(0., depth[:, None] - depth[self.nb.clip(min=0)]) * self.open
        total = head.sum(axis=1)
        squeezed = excess * self.skin_area * 1e6 * DT * rate / (1 + DT * rate)
        squeeze_flux = squeezed[:, None] * head / np.maximum(total[:, None], 1e-30)
        self.move(self.skin, self.skin_valid, squeeze_flux)
        exposed = np.where(available > depth + ROUGHNESS, 1., .1)
        evaporated = np.minimum(self.skin, self.skin_area * p.drying * .25 * exposed * DT)
        self.skin -= evaporated
        self.evap += evaporated.sum()
        return float(forward.sum()), float(reverse.sum()), float(squeeze_flux.sum()), touching

    def advance(self, positions, area, skin_area, sites, gaps):
        self.step += 1
        self.area, self.skin_area = area, skin_area
        self.inlet()
        self.flow[:] = 0.
        self.routed_ml = self.outer.sum()
        if self.outer.any() or self.knit.any():
            geometry = self.outer_geometry(positions)
            dt=DT/TRANSPORT_SUBSTEPS
            if transport_film is None or not transport_film(self,geometry,TRANSPORT_SUBSTEPS,dt,KNIT_WETTING.uptake_rate_s):
                for _ in range(TRANSPORT_SUBSTEPS):
                    self.transport(geometry,dt)
        if self.knit.any():
            self.wick(positions)
        return self.exchange(sites, gaps)

    def inventory(self):
        queued = air = 0.
        for p in self.pours:
            sent = pulse(p.age - p.delay)
            arrived = pulse(p.age - p.delay - p.flight)
            queued += (p.volume * p.shares) @ (1 - sent)
            air += (p.volume * p.shares) @ (sent - arrived)
        values = [queued, air, self.outer.sum(), self.knit.sum(), self.skin.sum(), self.out, self.evap]
        return dict(zip(('queued_ml', 'airborne_ml', 'outer_ml', 'knit_ml', 'skin_ml',
                         'outflow_ml', 'evaporated_ml', 'balance_error_ml'),
                        map(float, [*values, self.dosed - sum(values)])))
