# %%
########################################################################
# This file is an script that runs rrt algorithm for a unicycle.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import timeit
import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2 as cv
import fcl

np.set_printoptions(precision=1, suppress=True)
np.random.seed(1000)


########## Classes and Methods #########################################
class WorkSpace:
    def __init__(self, contours, lbx, ubx, lby, uby, eps=0.1):
        self.bounds = np.array((lbx, ubx)), np.array((lby, uby))
        self._contours = self._approximate_with_convex(contours, eps)

    def get_obstacle_mesh(self):
        return self._get_mesh_obstacles(self._contours)

    def get_cartesian_obstacle_contours(self):
        return self._contours

    def _approximate_with_convex(self, contours, eps=0.1):
        """
        Approximates a given contour with its convex hull if the
        convexity defects are not more than eps*100 percentage.
        """
        approx_contours = []
        for cnt in contours:
            if not cv.isContourConvex(cnt):
                hull = cv.convexHull(cnt)
                area_cnt = cv.contourArea(cnt)
                area_hull = cv.contourArea(hull)
                if area_hull < (1 + eps) * area_cnt:
                    cnt = hull
                else:
                    cnt = cv.approxPolyDP(cnt, 1, True)
            approx_contours.append(cnt.reshape(-1, 2))
        return approx_contours

    def _get_mesh_obstacles(self, contours):
        """
        Triangulates obstacles and returns the mesh and mech_contour.
        """
        mesh = []
        mesh_contours = []
        for cnt in contours:
            segments = np.vstack(
                (range(len(cnt)), np.roll(range(len(cnt)), -1))
            ).T
            tris = tr.triangulate(
                {"vertices": cnt.squeeze(), "segments": segments}, "p"
            )
            verts = tris["vertices"].astype(cnt.dtype)
            tris = tris["triangles"]
            mesh_contours.extend(list(verts[tris]))
            mesh.append((verts, tris, cv.isContourConvex(cnt.astype(int))))
        # Get space mesh.
        mesh_space, mesh_contours_space = self._get_space_mesh()
        mesh += mesh_space
        mesh_contours += mesh_contours_space
        return mesh, mesh_contours

    def _get_space_mesh(self):
        (lbx, ubx), (lby, uby) = self.bounds
        verts_lbx = np.array(
            [[1.5 * lbx, 0], [lbx, 1.5 * lby], [lbx, 1.5 * uby]]
        )
        verts_ubx = np.array(
            [[1.5 * ubx, 0], [ubx, 1.5 * uby], [ubx, 1.5 * lby]]
        )
        verts_lby = np.array(
            [[0, 1.5 * lby], [1.5 * ubx, lby], [1.5 * lbx, lby]]
        )
        verts_uby = np.array(
            [[0, 1.5 * uby], [1.5 * lbx, uby], [1.5 * ubx, uby]]
        )
        tris = np.array([[0, 1, 2]])
        mesh_contours = [verts_lbx, verts_ubx, verts_lby, verts_uby]
        #
        mesh = [(vert, tris, True) for vert in mesh_contours]
        return mesh, mesh_contours


class WorkSpaceImg(WorkSpace):
    def __init__(self, img_path, eps=0.1, scale=1):
        self.img = cv.imread(img_path, cv.IMREAD_COLOR)
        self.scale = scale
        self._set_bounds()
        self.bounds = self._cartesian_bounds()
        contours = self._find_obstacles()
        self._contours = self._approximate_with_convex(contours, eps)

    def cart2pix(self, xy):
        xy = np.array(xy, dtype=float)
        if xy.ndim < 2:
            xy = xy[None, :]
        pix = np.zeros_like(xy, dtype=int)  # [[x_pix, y_pix], ...].
        pix[:, 0] = np.clip(xy[:, 0] / self.scale + self.center[0], 0, self.w)
        pix[:, 1] = np.clip(self.center[1] - xy[:, 1] / self.scale, 0, self.h)
        return pix

    def pix2cart(self, pix):
        pix = np.array(pix, dtype=int)
        if pix.ndim < 2:
            pix = pix[None, :]
        xy = np.zeros_like(pix, dtype=float)
        xy[:, -1] = -pix[:, -1]  # Angle, if present.
        xy[:, 0] = (pix[:, 0] - self.center[0]) * self.scale  # x coord.
        xy[:, 1] = (self.center[1] - pix[:, 1]) * self.scale  # y coord.
        return xy

    def _set_bounds(self):
        self.h, self.w = self.img.shape[:2]
        self.center = np.array([self.w / 2, self.h / 2], dtype=int)

    def _cartesian_bounds(self):
        half_width, half_height = self.pix2cart([self.w, self.h])[0]
        ubx = half_width
        uby = -half_height
        return np.array((-ubx, ubx)), np.array((-uby, uby))

    def get_cartesian_obstacle_contours(self):
        return [self.pix2cart(cnt) for cnt in self._contours]

    def get_obstacle_mesh(self):
        contours = self.get_cartesian_obstacle_contours()
        return self._get_mesh_obstacles(contours)

    def _find_obstacles(self):
        mask = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(mask, 127, 255, cv.THRESH_BINARY_INV)
        contours, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        return contours


class Collision:
    def __init__(self, mesh, rrob=5.0):
        self.T0 = fcl.Transform()
        self.mesh = mesh
        self.rrob = rrob
        self.ball0 = fcl.Sphere(rrob)
        self.obstacles = self._build_obstacles(mesh)
        self.cmanager, self.col_req, self.dis_req = self._build_cmanager()

    def is_collision(self, poses):
        # Setup robots collision object and manage.
        balls = [fcl.Sphere(self.rrob) for _ in range(len(poses))]
        robots = [
            fcl.CollisionObject(ball, fcl.Transform(pos))
            for ball, pos in zip(balls, poses)
        ]
        manager = fcl.DynamicAABBTreeCollisionManager()
        manager.registerObjects(robots)
        manager.setup()
        # Detect collision.
        cdata = fcl.CollisionData(request=self.col_req)
        self.cmanager.collide(manager, cdata, fcl.defaultCollisionCallback)
        # Find index of first collision.
        ind_collision = 0
        if cdata.result.is_collision:
            ind_collision = min(
                [balls.index(contact.o2) for contact in cdata.result.contacts]
            )
        return cdata.result.is_collision, ind_collision

    def get_distance(self, poses):
        # Setup robots collision object and manage.
        robots = [
            fcl.CollisionObject(self.ball0, fcl.Transform(pos))
            for pos in poses
        ]
        manager = fcl.DynamicAABBTreeCollisionManager()
        manager.registerObjects(robots)
        manager.setup()
        # Calculate minimum distance.
        ddata = fcl.DistanceData(request=self.dis_req)
        self.cmanager.distance(manager, ddata, fcl.defaultDistanceCallback)
        return ddata.result.min_distance

    def is_collision_inc(self, poses):
        is_collision = False
        # min_distance = []
        for ind_collision, pos in enumerate(poses):
            # Setup robotscollision object.
            robot = fcl.CollisionObject(self.ball0, fcl.Transform(pos))
            # Detect collision.
            cdata = fcl.CollisionData(request=self.col_req)
            self.cmanager.collide(robot, cdata, fcl.defaultCollisionCallback)
            is_collision |= cdata.result.is_collision
            # Stop if collision detected.
            if is_collision:
                break
        return is_collision, ind_collision

    def update_obstacles(self, mesh):
        self.mesh = mesh
        self.obstacles = self._build_obstacles(mesh)

    def _build_obstacles(self, mesh):
        obstacles = []
        for verts, tris, is_convex in mesh:
            if is_convex:
                faces = np.concatenate(
                    (3 * np.ones((len(tris), 1), dtype=np.int64), tris), axis=1
                ).flatten()
                obs = fcl.Convex(verts, len(tris), faces)
                obstacles.append(fcl.CollisionObject(obs))
            else:
                for tri in tris:
                    vert = verts[tri]
                    tri = np.array([[0, 1, 2]], dtype=int)
                    faces = np.concatenate(
                        (3 * np.ones((len(tri), 1), dtype=np.int64), tri),
                        axis=1,
                    ).flatten()
                    obs = fcl.Convex(vert, len(tri), faces)
                    obstacles.append(fcl.CollisionObject(obs))
        return obstacles

    def _build_cmanager(self):
        cmanager = fcl.DynamicAABBTreeCollisionManager()
        cmanager.registerObjects(self.obstacles)
        cmanager.setup()
        col_req = fcl.CollisionRequest(
            num_max_contacts=10, enable_contact=True
        )
        dis_req = fcl.DistanceRequest()
        return cmanager, col_req, dis_req


def test_collision():
    # Get space from masked image and calculate obstacle and space mesh.
    space = WorkSpaceImg("./world2.png")
    contours = space.get_cartesian_obstacle_contours()
    mesh, mesh_contours = space.get_obstacle_mesh()
    # Set up collosion detector with some examples.
    collision = Collision(mesh)
    poses_i = [
        np.array([0.0, 30.0, 0.0], dtype=float),
    ]
    poses_f = [
        np.array([0.0, 45.0, 0.0], dtype=float),
    ]
    # Determining collision and measuring distance.
    print(collision.is_collision(poses_i))
    print(collision.get_distance(poses_i))
    # Determining index of collision given a path.
    N = 100
    poses = np.linspace(poses_i[0], poses_f[0], N + 1)
    print(collision.is_collision(poses))
    # Determining index of collision of given path incrementally.
    print(collision.is_collision_inc(poses))
    # Plot space in cartesian coord.
    fig, ax = plt.subplots()
    xlim, ylim = space.bounds
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    # Plot obstacles.
    for i, cnt in enumerate(contours):
        ax.add_patch(Polygon(cnt, color="k"))
    # Plot obstacle and space mesh.
    for cnt in mesh_contours:
        ax.add_patch(Polygon(cnt, ec="lime", fill=False))
    plt.show()


########## Test ########################################################
if __name__ == "__main__":
    test_collision()
