"""
Authorized by: KevinLADLee
https://github.com/KevinLADLee/carla_dataset_tools/blob/master/utils/visulize_map.py
Refer and edited by: TriDoan (doantrancaotri1108@gmail.com)
Thank you for the great work!
"""

import argparse
import json
import os
import pickle

import matplotlib
from loguru import logger

import carla

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt


class Line:
    x: list[float] = []
    y: list[float] = []


class MapVisualization:
    def __init__(self, args):
        self.args = args
        self.carla_client = carla.Client(args.host, args.port, worker_threads=1)
        # Load the specified map
        if hasattr(args, "map") and args.map:
            self.carla_client.load_world(args.map)
        self.world = self.carla_client.get_world()
        self.map = self.world.get_map()
        self.fig, self.ax = plt.subplots()
        self.line_list = []

    def destroy(self):
        self.carla_client = None
        self.world = None
        self.map = None

    @staticmethod
    def lateral_shift(transform, shift):
        """Makes a lateral shift of the forward vector of a transform"""
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def draw_line(self, points: list):
        x = []
        y = []
        for p in points:
            x.append(p.x)
            y.append(-p.y)
        line = Line()
        line.x = x
        line.y = y
        self.line_list.append(line)
        self.ax.plot(x, y, color="darkslategrey", markersize=2)
        return True

    def draw_spawn_points(self):
        spawn_points = self.map.get_spawn_points()
        for i in range(len(spawn_points)):
            p = spawn_points[i]
            x = p.location.x
            y = -p.location.y
            self.ax.text(
                x,
                y,
                str(i),
                fontsize=6,
                color="darkorange",
                va="center",
                ha="center",
                weight="bold",
            )

    def draw_roads(self):
        precision = 0.1
        topology = self.map.get_topology()
        topology = [x[0] for x in topology]
        topology = sorted(topology, key=lambda w: w.transform.location.z)
        set_waypoints = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            set_waypoints.append(waypoints)

        for waypoints in set_waypoints:
            # waypoint = waypoints[0]
            road_left_side = [
                self.lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints
            ]
            road_right_side = [
                self.lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints
            ]
            # road_points = road_left_side + [x for x in reversed(road_right_side)]
            # self.add_line_strip_marker(points=road_points)

            if len(road_left_side) > 2:
                self.draw_line(points=road_left_side)
            if len(road_right_side) > 2:
                self.draw_line(points=road_right_side)

    # region save
    def save_spawn_points_to_json(self):
        """Save spawn points to JSON file"""
        spawn_points = self.map.get_spawn_points()
        spawn_points_data = []

        for i, spawn_point in enumerate(spawn_points):
            spawn_data = {
                "index": i,
                "location": {
                    "x": spawn_point.location.x,
                    "y": spawn_point.location.y,
                    "z": spawn_point.location.z,
                },
                "rotation": {
                    "pitch": spawn_point.rotation.pitch,
                    "yaw": spawn_point.rotation.yaw,
                    "roll": spawn_point.rotation.roll,
                },
            }
            spawn_points_data.append(spawn_data)

        # Save to JSON file
        output_filename = (
            f"{self.args.output_dir}/{self.args.map}/{self.args.map}_spawn_points.json"
        )
        with open(output_filename, "w", encoding="utf-8") as json_file:
            json.dump(spawn_points_data, json_file, indent=2, ensure_ascii=False)

        logger.info(f"Spawn points saved to: {output_filename}")
        logger.info(f"Total spawn points: {len(spawn_points_data)}")
        return output_filename

    def save_map_info_to_pkl(self):
        with open(
            f"{self.args.output_dir}/{self.args.map}/{self.args.map}_map_info.pkl", "wb"
        ) as pickle_file:
            # Output map visualization info to a pkl file
            pickle.dump(self.line_list, pickle_file)

    def save_visualization_to_png(self):
        # Save the plot to a file instead of showing it (since we're using non-interactive backend)
        output_filename = (
            f"{self.args.output_dir}/{self.args.map}/{self.args.map}_visualization.png"
        )
        plt.savefig(output_filename, dpi=300, bbox_inches="tight")
        logger.info(f"Map visualization saved to: {output_filename}")

    # endregion save


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--host",
        metavar="H",
        default="localhost",
        help="IP of the host CARLA Simulator (default: localhost)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port of CARLA Simulator (default: 2000)",
    )
    argparser.add_argument(
        "-m", "--map", default="Town02", help="Load a new map to visualize"
    )
    argparser.add_argument(
        "-o",
        "--output_dir",
        default="local_rss/visualize_map",
        help="Output directory for the visualizations (default: local_rss/visualize_map)",
    )

    args = argparser.parse_args()

    viz = MapVisualization(args)
    viz.draw_roads()
    viz.draw_spawn_points()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(f"{args.output_dir}/{args.map}"):
        os.makedirs(f"{args.output_dir}/{args.map}")

    # Save spawn points to JSON file
    viz.save_spawn_points_to_json()
    viz.save_map_info_to_pkl()
    viz.save_visualization_to_png()

    viz.destroy()
    plt.axis("equal")


if __name__ == "__main__":
    # execute only if run as a script
    main()
