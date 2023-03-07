# %%
########################################################################
# This file is an script that runs rrt algorithm for a unicycle.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import time
import cProfile
from typing import TypedDict, List
import re
import numpy as np
from scipy.special import gamma
import triangle as tr
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import cv2 as cv
import fcl

from dubins import Dubins

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
                ).ravel()
                obs = fcl.Convex(verts, len(tris), faces)
                obstacles.append(fcl.CollisionObject(obs))
            else:
                for tri in tris:
                    vert = verts[tri]
                    tri = np.array([[0, 1, 2]], dtype=int)
                    faces = np.concatenate(
                        (3 * np.ones((len(tri), 1), dtype=np.int64), tri),
                        axis=1,
                    ).ravel()
                    obs = fcl.Convex(vert, len(tri), faces)
                    obstacles.append(fcl.CollisionObject(obs))
        return obstacles

    def _build_cmanager(self):
        cmanager = fcl.DynamicAABBTreeCollisionManager()
        cmanager.registerObjects(self.obstacles)
        cmanager.setup()
        col_req = fcl.CollisionRequest(
            num_max_contacts=1000, enable_contact=True
        )
        dis_req = fcl.DistanceRequest()
        return cmanager, col_req, dis_req


class RRTDubins:
    class Node(TypedDict):
        pos: np.ndarray
        parent: int
        child: list
        path: np.ndarray
        cost: float
        value: float
        artists: List

    class Path(TypedDict):
        nodes: List
        value: float

    def __init__(
        self,
        start,
        goal,
        bounds,
        collision,
        stypes,
        obs_contours,
        norm_weights=[1, 1, 1],
        max_size=1000,
        max_stride=None,
        min_res=1.0,
        min_radius=10.0,
        goal_bias=0.05,
        threshold=1e-3,
    ) -> None:
        start = np.array(start, dtype=float).squeeze()
        goal = np.array(goal, dtype=float).squeeze()
        self.start = self.Node(
            pos=start,
            parent=-1,
            childs=[],
            path=None,
            cost=0.0,
            value=0.0,
            radius=float("inf"),
        )
        self.goal = self.Node(
            pos=goal,
            parent=-1,
            path=None,
            childs=[],
            cost=0.0,
            value=0.0,
            radius=float("inf"),
        )
        self.n_state = 3
        (lbx, ubx), (lby, uby) = bounds
        self.lb = np.array([lbx, lby, -np.pi])
        self.ub = np.array([ubx, uby, np.pi])
        self.collision = collision
        self.steer = Dubins(stypes)
        self.obs_contours = obs_contours
        self.weights = np.array(norm_weights, dtype=float) / np.linalg.norm(
            norm_weights
        )
        self.max_size = max_size
        self.max_stride = max_stride
        if max_stride is None:
            self.max_stride = (
                0.2 * self._weighted_norm((self.ub - self.lb)) ** 0.5
            )
        self.min_res = min_res
        self.step = int(self.max_stride / self.min_res)
        self.min_radius = min_radius
        self.goal_bias = goal_bias
        self.thr = threshold
        #
        self.nodes = [self.start]
        self.paths = []
        self.N = 1
        self._nposes = np.zeros((max_size * 10, self.n_state))
        self._nposes[0] = self.start["pos"].copy()
        self._nvalues = np.zeros(max_size*10, dtype=float)
        self._best_value = float("inf")
        self._best_ind = None
        self._goal_inds_mask = np.zeros(max_size, dtype=bool)

    def plan(self, fig_name=None, anim_name=False, anim_online=False):
        artists = []
        fig, ax = plt.subplots(constrained_layout=True)
        fig, ax, cid = self._set_up_plot(fig, ax)
        # artists.append(ax.get_children())
        for i in range(1, self.max_size):
            # Draw a collision free sample from state space.
            rnd = self._sample_collision_free()
            # Find nearest node
            nearest_ind, nearest_node = self._nearest_node(rnd)
            # Calculate path from nearest to rnd.
            path = self._steer(nearest_node["pos"], rnd)
            # Check the path for collision.
            is_collision, ind_collision = self._is_collision(path)
            # Add new nodes.
            new_node, reached_goal = self._extend(
                nearest_ind, nearest_node, path, ind_collision
            )
            """ if not i % 10:
                print(f"iteration = {i:>6d}") """
            # Draw path and check for stoppage.
            arts = []
            if new_node is not None:
                arts += self._draw(ax, new_node, color="deepskyblue", zorder=1)
            if arts and anim_name:
                artists.append(arts)
            #
            if reached_goal:
                # Update goal index mask.
                self._goal_inds_mask[self.N - 1] = True
                reached_goal = False
            #
            goal_inds = np.where(self._goal_inds_mask)[0]
            goal_values = self._nvalues[goal_inds]
            if len(goal_values) > 0:
                candidate_best_value = goal_values.min()
                if candidate_best_value < self._best_value:
                    arts = []
                    # Redraw previous best path in other color.
                    if self._best_ind is not None:
                        for node in self.paths[self._best_ind]["nodes"]:
                            arts += self._draw(
                                ax, node, color="violet", zorder=3
                            )
                    #
                    self._generate_path(goal_inds[goal_values.argmin()])
                    self._best_ind = len(self.paths) - 1
                    self._best_value = candidate_best_value
                    print(f"best value: {self._best_value}")
                    print("Solution found.")
                    # Draw new best path.
                    for node in self.paths[self._best_ind]["nodes"]:
                        arts += self._draw(ax, node, color="magenta", zorder=3)
                    if arts:
                        artists.append(arts)
            #
            if anim_online:
                plt.pause(0.001)
        # Saving final figure.
        if fig_name is not None:
            self._save_plot(fig, fig_name)
        # Generating animation and saving it if requested.
        anim = []
        if anim_name:
            fig, ax, anim = self._make_save_animation(
                fig, ax, cid, artists, anim_name
            )
        return fig, ax, anim

    def _sample(self):
        prob = np.random.rand()
        if prob > self.goal_bias:
            return np.random.uniform(self.lb, self.ub)
        else:
            return self.goal["pos"]

    def _sample_collision_free(self):
        while True:
            rnd = self._sample()
            rndc = rnd.copy()
            rndc[-1] = 0.0
            is_collision, _ = self.collision.is_collision(rndc[None, :])
            if not is_collision:
                break
        return rnd

    def _nearest_node(self, pos):
        dposes = self._nposes[: self.N] - pos
        distances = self._weighted_norm(dposes)
        distances[self._goal_inds_mask[: self.N]] = np.inf
        nearest_ind = np.argmin(distances)
        # nearest_ind = np.where(
        #    distances == np.min(distances[self._goal_mask_inds[: self.N]])
        # )[0][0]
        nearest_node = self.nodes[nearest_ind]
        return nearest_ind, nearest_node

    def _weighted_norm(self, vects):
        return (
            np.sum(self.weights * vects.reshape(-1, self.n_state) ** 2, axis=1)
            ** 0.5
        )

    def _steer(self, from_pos, to_pos):
        if np.allclose(from_pos, to_pos):
            path = np.array([[0.0, 0.0, 0.0]])
        else:
            path, _, _ = self.steer.steer(
                from_pos,
                to_pos,
                step_size=self.min_res,
                min_radius=self.min_radius,
            )
        return path

    def _is_collision(self, path):
        pathc = path.copy()
        pathc[:, 2] = 0.0
        is_collision, ind_collision = self.collision.is_collision(pathc)
        ind_collision += (not is_collision) * (len(path) + 1)
        return is_collision, ind_collision

    def _extend(self, nearest_ind, nearest_node, path, ind_collision):
        new_node = None
        reached_goal = False
        lpath = len(path)
        if lpath > 1:
            step = min(lpath - 1, self.step)
            if ind_collision > (lpath - 1):
                # Whole collision free, check if goal reached.
                if self._weighted_norm(path[-1] - self.goal["pos"]) < self.thr:
                    reached_goal = True
                    step = lpath - 1
            # If collision free, build the node.
            if ind_collision > max(1, step):
                cost = step * self.min_res
                value = nearest_node["value"] + cost
                new_node = self.Node(
                    pos=path[step],
                    parent=nearest_ind,
                    childs=[],
                    path=path[: step + 1],
                    cost=cost,
                    value=value,
                )
                nearest_node["childs"].append(self.N)
                self._nposes[self.N] = path[step]
                self._nvalues[self.N] = value
                self.nodes.append(new_node)
                self.N += 1
        return new_node, reached_goal

    def _generate_path(self, ind):
        # Generate final path and draw it.
        path_nodes = [self.nodes[ind]]
        value = path_nodes[0]["value"]
        print(f"Number of tree nodes {self.N}")
        while True:
            parent = path_nodes[0]["parent"]
            if parent < 1:
                break
            path_nodes.insert(0, self.nodes[parent])
        path = self.Path(nodes=path_nodes, value=value)
        self.paths.append(path)
        return path

    def _get_stop_req(self):
        stop_req = False
        in_str = input("Enter Y to stop: ").strip()
        if re.match("[Yy]", in_str):
            stop_req = True
        return stop_req

    def _set_up_plot(self, fig, ax, cid=-1):
        if cid != -1:
            fig.canvas.mpl_disconnect(cid)
        else:
            cid = fig.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
        ax.clear()
        ax.grid(zorder=0.0)
        # Draw obstacles.
        for i, cnt in enumerate(self.obs_contours):
            ax.add_patch(Polygon(cnt, color="k", zorder=2.0))
        # Draw start and end.
        ax.plot(
            self.start["pos"][0],
            self.start["pos"][1],
            ls="",
            marker="s",
            mfc="yellow",
            mec="k",
            zorder=10.0,
            label="start",
        )
        ax.plot(
            self.goal["pos"][0],
            self.goal["pos"][1],
            ls="",
            marker="s",
            mfc="lime",
            mec="k",
            zorder=10.0,
            label="goal",
        )
        ax.legend(loc="upper right")
        # Boundaries.
        lbx, lby = self.lb[:2]
        ubx, uby = self.ub[:2]
        ax.set_xlim(lbx, ubx)
        ax.set_ylim(lby, uby)
        #
        ax.set_title("Dubins car RRT planning")
        ax.set_aspect("equal", adjustable="box")
        return fig, ax, cid

    def _draw(self, ax, node, color="deepskyblue", zorder=1):
        arts = []
        path = node["path"]
        n = min(2, len(path))
        arts += ax.plot(
            path[:, 0], path[:, 1], ls="-", lw=1.0, c=color, zorder=zorder
        )
        dx = path[-1, 0] - path[-n, 0]
        dy = path[-1, 1] - path[-n, 1]

        arts.append(
            ax.arrow(
                path[-n, 0],
                path[-n, 1],
                dx,
                dy,
                length_includes_head=True,
                head_width=5.0,
                head_length=10.0,
                color=color,
                zorder=zorder,
            )
        )
        return arts

    def _save_plot(self, fig, name):
        """Overwrites files if they already exist."""
        fig_name = f"{name}.pdf"
        fig.savefig(fig_name, bbox_inches="tight", pad_inches=0.05)

    def _animate(self, arts, ax):
        for art in arts:
            ax.add_artist(art)

    def _make_save_animation(self, fig, ax, cid, artists, anim_name):
        print("Starting animation processing.")
        fig, ax, cid = self._set_up_plot(fig, ax, cid)
        anim = animation.FuncAnimation(
            fig,
            self._animate,
            fargs=(ax,),
            interval=10,
            frames=artists,
            repeat_delay=1000,
        )
        print("Animation is produced.")
        anim.save(anim_name + ".gif", fps=8)
        print("Animation is saved. You can close the figure.")
        return fig, ax, anim


class RRTSDubins(RRTDubins):
    def __init__(
        self,
        start,
        goal,
        bounds,
        collision,
        stypes,
        obs_contours,
        norm_weights=[1, 1, 1],
        max_size=1000,
        max_stride=1000000,
        min_res=1,
        min_radius=10,
        goal_bias=0.05,
    ) -> None:
        super().__init__(
            start,
            goal,
            bounds,
            collision,
            stypes,
            obs_contours,
            norm_weights,
            max_size,
            max_stride,
            min_res,
            min_radius,
            goal_bias,
        )
        self._gamma_s = self._calc_gamma_star()
        self._k_s = 2 * np.exp(1)

    def plans(self, fig_name=None, anim_name=False, anim_online=False):
        artists = []
        fig, ax = plt.subplots(constrained_layout=True)
        fig, ax, cid = self._set_up_plot(fig, ax)
        # artists.append(ax.get_children())
        for i in range(1, self.max_size):
            # Draw a collision free sample from state space.
            rnd = self._sample_collision_free()
            # Find nearest node
            nearest_ind, nearest_node = self._nearest_node(rnd)
            # Calculate path from nearest to rnd.
            path = self._steer(nearest_node["pos"], rnd)
            # Check the path for collision.
            is_collision, ind_collision = self._is_collision(path)
            # Add new nodes.
            new_node, reached_goal = self._extend(
                nearest_ind, nearest_node, path, ind_collision
            )
            if not i % 10:
                print(f"iteration = {i:>6d}")
            arts = []
            if new_node is not None:
                # Find nearest neighbors.
                near_inds, near_nodes = self._nearest_neighbor(new_node["pos"])
                # Rewire new_node.
                self._rewire_new_node(near_inds, near_nodes, new_node)
                # Rewire nearest neighbors.
                self._rewire_near_nodes(near_inds, near_nodes, new_node)
                # Draw path and check for stoppage.
                arts += self._draw(ax, new_node, color="deepskyblue", zorder=1)
            if arts and anim_name:
                artists.append(arts)
            #
            if reached_goal:
                # Update goal index mask.
                self._goal_inds_mask[self.N - 1] = True
                reached_goal = False
            #
            goal_inds = np.where(self._goal_inds_mask)[0]
            goal_values = self._nvalues[goal_inds]
            if len(goal_values) > 0:
                candidate_best_value = goal_values.min()
                if candidate_best_value < self._best_value:
                    arts = []
                    # Redraw previous best path in other color.
                    if self._best_ind is not None:
                        for node in self.paths[self._best_ind]["nodes"]:
                            arts += self._draw(
                                ax, node, color="violet", zorder=3
                            )
                    #
                    self._generate_path(goal_inds[goal_values.argmin()])
                    self._best_ind = len(self.paths) - 1
                    self._best_value = candidate_best_value
                    print(f"best cost: {self._best_value}")
                    print("Solution found.")
                    # Draw new best path.
                    for node in self.paths[self._best_ind]["nodes"]:
                        arts += self._draw(ax, node, color="magenta", zorder=3)
                    """ if self._get_stop_req():
                        break """
            #
            if anim_online:
                plt.pause(0.001)
        # Saving final figure.
        if fig_name is not None:
            self._save_plot(fig, fig_name)
        # Generating animation and saving it if requested.
        anim = []
        if anim_name:
            fig, ax, anim = self._make_save_animation(
                fig, ax, cid, artists, anim_name
            )
        return fig, ax, anim

    def _calc_gamma_star(self):
        d = 3  # Space dimension.
        vol = np.prod(self.ub - self.lb)
        vol_unit = np.pi ** (d / 2) / gamma(1 + d / 2)
        gamma_star = 2 * (1 + 1 / d) ** (1 / d) * (vol / vol_unit) ** (1 / d)
        gamma_star *= 1.1
        return gamma_star

    def _nearest_neighbor(self, pos):
        #r = self._gamma_s * (np.log(self.N)/self.N)**(1/self.n_state)
        #r = self.max_stride
        r = self._gamma_s
        distances = self._weighted_norm(self._nposes[: self.N - 1] - pos)
        distances[self._goal_inds_mask[: self.N - 1]] = np.inf
        near_inds = np.where(distances <= r)[0]
        near_inds = near_inds[np.argsort(distances[near_inds])]
        near_nodes = [self.nodes[near_ind] for near_ind in near_inds]
        return near_inds, near_nodes

    def _k_nearest_neighbor(self, pos):
        # Do not consider last node in calculating distances.
        distances = self._weighted_norm(self._nposes[: self.N - 1] - pos)
        distances[self._goal_inds_mask[: self.N - 1]] = np.inf
        k = int(self._k_s * np.log(self.N))
        #
        if len(distances) < k + 1:
            near_inds = np.argsort(distances)[:k]
        else:
            near_inds = np.argpartition(distances, k)[:k]
            near_inds = near_inds[np.argsort(distances[near_inds])]
        near_nodes = [self.nodes[near_ind] for near_ind in near_inds]
        return near_inds, near_nodes

    def _rewire(self, parent_ind, parent_node, child_ind, child_node):
        path = self._steer(parent_node["pos"], child_node["pos"])
        cost = self.min_res * (len(path) - 1)
        value = parent_node["value"] + cost
        if value < child_node["value"]:
            is_collision, _ = self._is_collision(path)
            if not is_collision:
                # Remove child node from its parent's child list.
                prev_parent = self.nodes[child_node["parent"]]
                prev_parent["childs"].remove(child_ind)
                # Update child node and its new parent paremeters.
                child_node["parent"] = parent_ind
                child_node["path"] = path
                child_node["cost"] = cost
                child_node["value"] = value
                self._nvalues[child_ind] = value
                parent_node["childs"].append(child_ind)
                self._propagate_cost_to_childs(child_node)
                return True
        return False

    def _rewire_new_node(self, near_inds, near_nodes, new_node):
        new_ind = self.N - 1
        new_value = self._nvalues[new_ind]
        inds = np.where(self._nvalues[near_inds] < new_value)[0]
        near_inds = near_inds[inds]
        near_nodes = [near_nodes[ind] for ind in inds]
        for near_ind, near_node in zip(near_inds[1:], near_nodes[1:]):
            self._rewire(near_ind, near_node, self.N - 1, new_node)

    def _rewire_near_nodes(self, near_inds, near_nodes, new_node):
        rewired = False
        new_ind = self.N - 1
        new_value = self._nvalues[new_ind]
        inds = np.where(self._nvalues[near_inds] > new_value)[0]
        near_inds = near_inds[inds]
        near_nodes = [near_nodes[ind] for ind in inds]
        for near_ind, near_node in zip(near_inds, near_nodes):
            rewired|=self._rewire(self.N - 1, new_node, near_ind, near_node)
        return rewired

    def _propagate_cost_to_childs(self, node):
        for child_ind in node["childs"]:
            child_node = self.nodes[child_ind]
            child_node["value"] = node["value"] + child_node["cost"]
            self._propagate_cost_to_childs(child_node)


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


def test_rrt():
    # Get space from masked image and calculate obstacle and space mesh.
    space = WorkSpaceImg("./world2.png")
    obs_contours = space.get_cartesian_obstacle_contours()
    mesh, mesh_contours = space.get_obstacle_mesh()
    # Set up collosion detector with some examples.
    collision = Collision(mesh, rrob=5.0)
    # Set up planner.
    stypes = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]
    # RRT planning.
    start = np.array([-300, -50, np.deg2rad(45)], dtype=float)
    goal = np.array([200, 100, np.deg2rad(-45)], dtype=float)
    rrt = RRTDubins(
        start,
        goal,
        space.bounds,
        collision,
        stypes,
        obs_contours,
        norm_weights=[1, 1, 1],
        max_size=2000,
        max_stride=1000,
        min_res=1.0,
        min_radius=20.0,
        goal_bias=0.05,
    )
    anim_online = False  # True#
    fig_name = None  # "final_plan"#
    anim_name = None  # "final_plan"#
    fig, ax, anim = rrt.plan(
        fig_name=fig_name, anim_name=anim_name, anim_online=anim_online
    )
    plt.show()


def test_rrts():
    # Get space from masked image and calculate obstacle and space mesh.
    space = WorkSpaceImg("./world2.png")
    obs_contours = space.get_cartesian_obstacle_contours()
    mesh, mesh_contours = space.get_obstacle_mesh()
    # Set up collosion detector with some examples.
    collision = Collision(mesh, rrob=5.0)
    # Set up planner.
    stypes = ["LSL", "RSR", "LSR", "RSL", "RLR", "LRL"]
    # RRT planning.
    start = np.array([-300, -50, np.deg2rad(45)], dtype=float)
    goal = np.array([200, 100, np.deg2rad(-45)], dtype=float)
    rrt = RRTSDubins(
        start,
        goal,
        space.bounds,
        collision,
        stypes,
        obs_contours,
        norm_weights=[1, 1, 1],
        max_size=1000,
        max_stride=150,
        min_res=1.0,
        min_radius=14.02,
        goal_bias=0.05,
    )
    anim_online = False  # True#
    fig_name = None  # "final_plan"#
    anim_name = None  # "final_plan"#

    tstart = time.time()
    fig, ax, anim = rrt.plans(
        fig_name=fig_name, anim_name=anim_name, anim_online=anim_online
    )
    tfinish = time.time()
    print(f"Elapsed time is: {tfinish - tstart:+07.2f}s.")
    plt.show()


########## Test ########################################################
if __name__ == "__main__":
    test_rrts()
    pass
