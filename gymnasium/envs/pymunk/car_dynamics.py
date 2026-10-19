"""Top-down car dynamics simulation.

Some ideas are taken from this great tutorial http://www.iforce2d.net/b2dtut/top-down-car by Chris Campbell.
This simulation is a bit more detailed, with wheels rotation.

Created by Oleg Klimov
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from gymnasium.error import DependencyNotInstalled

try:
    import pymunk
except ImportError as e:
    raise DependencyNotInstalled(
        'pymunk is not installed, run `pip install "gymnasium[pymunk]"`'
    ) from e

# constants for collision detection
WHEEL, TILE = 1, 2
SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
    1000000 * SIZE * SIZE
)  # friction ~= mass ~= size^2 (calculated implicitly using density)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]

HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]

HULL_PARTS = [
    [(x * SIZE, y * SIZE) for x, y in part]
    for part in (HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4)
]  # adjusting x,y coordinations by size and grouping into one list

WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)


@dataclass(slots=True)
class Particle:
    """Class for representing tyre marks after drift or lockup."""

    color: tuple[int, int, int]
    poly: list[tuple[int, int]]
    grass: Any
    ttl: int = 1


def _create_car_body(
    space: pymunk.Space,
    init_x: float,
    init_y: float,
    init_angle: float,
    body_parts: list[list[tuple[float, float]]],
) -> pymunk.Body:
    """Creates body of a car and adds it to the pymunk space. Needs list of parts (polygons), that form a body of a car."""
    car_body = pymunk.Body()
    car_body.position = (init_x, init_y)
    car_body.angle = init_angle

    shapes = []
    for part in body_parts:
        shape = pymunk.Poly(car_body, part)
        shape.density = 1.0
        shapes.append(shape)

    space.add(car_body, *shapes)
    return car_body


class Car:
    """Class representing car object with custom physics calculation of tyre rolling speed, braking, grip or its absence and subsequent lockup."""

    def __init__(self, space, init_angle, init_x, init_y):
        """Creates Car object in provided space at the init coordinates and angle."""
        self.space: pymunk.Space = space
        self.hull: pymunk.Body = _create_car_body(
            self.space, init_x, init_y, init_angle, HULL_PARTS
        )

        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0

            w = pymunk.Body()
            w.position = self.hull.local_to_world((wx * SIZE, wy * SIZE))
            w.angle = init_angle
            w_shape = pymunk.Poly(
                w, [(x * front_k * SIZE, y * front_k * SIZE) for x, y in WHEEL_POLY]
            )
            w_shape.density = 0.1
            w_shape.filter = pymunk.ShapeFilter(group=0, categories=0x20, mask=0x01)
            w_shape.elasticity = 0.0
            w_shape.collision_type = WHEEL

            joint = pymunk.PivotJoint(self.hull, w, (wx * SIZE, wy * SIZE), (0, 0))
            joint.collide_bodies = False

            limit = pymunk.RotaryLimitJoint(self.hull, w, -0.4, 0.4)
            motor = pymunk.SimpleMotor(self.hull, w, 0.0)
            motor.max_force = 180 * 900 * SIZE * SIZE

            w.joint = joint
            w.limit = limit
            w.motor = motor
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0  # wheel angle
            w.omega = 0.0  # angular velocity
            w.skid_start = None
            w.skid_particle = None

            self.space.add(w, w_shape, joint, limit, motor)
            w.tiles = set()
            self.wheels.append(w)

        self.drawlist = self.wheels + [self.hull]
        self.particles = deque(maxlen=30)

    def gas(self, gas):
        """control: rear wheel drive.

        Args:
            gas (float): How much gas gets applied. Gets clipped between 0 and 1.
        """
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1  # gradually increase, but stop immediately
            w.gas += diff

    def brake(self, b):
        """control: brake.

        Args:
            b (0..1): Degree to which the brakes are applied. More than 0.9 blocks the wheels to zero rotation
        """
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        """control: steer.

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side
        """
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        """Advance car physics by one time step."""
        for w in self.wheels:
            # Steer each wheel
            rel_w = w.angular_velocity - self.hull.angular_velocity
            joint_angle = (w.angle - self.hull.angle) + rel_w * dt  # predicted angle
            dir = np.sign(w.steer - joint_angle)
            val = abs(w.steer - joint_angle)
            w.motor.rate = -dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT * 0.6  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False

            # Force
            forw = pymunk.Vec2d(0, 1).rotated(w.angle)
            side = pymunk.Vec2d(1, 0).rotated(w.angle)
            v = w.velocity
            vf = forw.dot(v)  # forward speed
            vs = side.dot(v)  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * ENGINE_POWER
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.apply_force_at_world_point(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                w.position,
            )

    def draw(self, surface, zoom, translation, angle, draw_particles=True):
        """Draw the car and its shapes into canvas."""
        import pygame.draw

        if draw_particles:
            for p in self.particles:
                poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in p.poly]
                poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in poly
                ]
                pygame.draw.lines(
                    surface, color=p.color, points=poly, width=2, closed=False
                )

        for obj in self.drawlist:
            for s in obj.shapes:
                path = [obj.local_to_world(v) for v in s.get_vertices()]
                path = [(coords[0], coords[1]) for coords in path]
                path = [pygame.math.Vector2(c).rotate_rad(angle) for c in path]
                path = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in path
                ]
                color = [int(c * 255) for c in obj.color]

                pygame.draw.polygon(surface, color=color, points=path)

                if not hasattr(obj, "phase"):
                    continue
                a1 = obj.phase
                a2 = obj.phase + 1.2  # radians
                s1 = math.sin(a1)
                s2 = math.sin(a2)
                c1 = math.cos(a1)
                c2 = math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                white_poly = [
                    (-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                    (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                    (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE),
                ]
                white_poly = [obj.local_to_world(v) for v in white_poly]

                white_poly = [(coords[0], coords[1]) for coords in white_poly]
                white_poly = [
                    pygame.math.Vector2(c).rotate_rad(angle) for c in white_poly
                ]
                white_poly = [
                    (
                        coords[0] * zoom + translation[0],
                        coords[1] * zoom + translation[1],
                    )
                    for coords in white_poly
                ]
                pygame.draw.polygon(surface, color=WHEEL_WHITE, points=white_poly)

    def _create_particle(self, point1, point2, grass):
        p = Particle(
            color=WHEEL_COLOR if not grass else MUD_COLOR,
            poly=[(point1[0], point1[1]), (point2[0], point2[1])],
            grass=grass,
        )
        self.particles.append(p)
        return p

    def destroy(self):
        """Delete the hull and wheels from pymunk.Space object, sets parameters to None."""
        self.space.remove(self.hull, *self.hull.shapes)
        self.hull = None
        for w in self.wheels:
            self.space.remove(w, *w.shapes, w.joint, w.limit, w.motor)
        self.wheels = []
