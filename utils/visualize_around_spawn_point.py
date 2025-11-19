#!/usr/bin/env python3
"""
Script to visualize a region around a start point in CARLA
Captures two images with only ego vehicle:
1. Third-person view from ego vehicle at coordinates (x0, y0)
2. Bird's eye view with materials and ego vehicle centered at (x0, y0)

Usage:
    python visualize_region_around_start_point.py --map Town01 --x 100.0 --y 200.0 --radius 100
"""

import argparse
import subprocess
import sys
import os
import carla
import matplotlib
import numpy as np
from loguru import logger
import time

matplotlib.use("Agg")  # Use non-interactive backend


class Line:
    """Line class to match the structure used in visualize_map.py"""

    def __init__(self):
        self.x = []
        self.y = []


class RegionVisualizer:
    def __init__(self, args):
        self.args = args
        self.carla_client = carla.Client(args.host, args.port, worker_threads=1)

        # Load the specified map
        if hasattr(args, "map") and args.map:
            self.carla_client.load_world(args.map)

        self.world = self.carla_client.get_world()
        self.map = self.world.get_map()

        # Set up output directory
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            logger.warning(f"Output directory already exists: {self.output_dir}")

    def destroy(self):
        self.carla_client = None
        self.world = None
        self.map = None

    def _spawn_ego_vehicle(self):
        """Spawn only the ego vehicle in the region"""
        logger.info("Spawning ego vehicle in the region")

        # Find closest spawn point for ego vehicle
        ego_spawn, ego_spawn_index = self.get_spawn_point_near_coordinates(
            self.args.x, self.args.y
        )

        if not ego_spawn:
            logger.error("No suitable spawn point found for ego vehicle")
            return None

        # Spawn ego vehicle
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("color", "255,0,0")  # Red color for ego
        ego_vehicle = self.world.spawn_actor(ego_bp, ego_spawn)

        logger.info(f"Ego vehicle spawned at spawn point {ego_spawn_index}")
        return ego_vehicle

    def get_spawn_point_near_coordinates(self, target_x, target_y):
        """Find the closest spawn point to the target coordinates"""
        spawn_points = self.map.get_spawn_points()
        min_distance = float("inf")
        closest_spawn = None
        closest_index = -1

        for i, spawn_point in enumerate(spawn_points):
            distance = np.sqrt(
                (spawn_point.location.x - target_x) ** 2
                + (spawn_point.location.y - target_y) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_spawn = spawn_point
                closest_index = i

        logger.info(
            f"Closest spawn point {closest_index} at distance {min_distance:.2f}m from target ({target_x}, {target_y})"
        )
        return closest_spawn, closest_index

    def capture_third_person_view(self, ego_vehicle, image_size: tuple[int, int]):
        """Capture third-person view from ego vehicle"""
        logger.info(
            f"Capturing third-person view from ego vehicle at coordinates ({int(self.args.x)}, {int(self.args.y)})"
        )

        if not ego_vehicle:
            logger.error("No ego vehicle provided")
            return False

        camera = None
        try:
            # Set up camera
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(image_size[0]))
            camera_bp.set_attribute("image_size_y", str(image_size[1]))
            camera_bp.set_attribute("fov", "90")

            # Position camera behind and above the ego vehicle for third-person view
            camera_transform = carla.Transform(
                carla.Location(x=-8.0, z=3.0),  # Behind and above the vehicle
                carla.Rotation(pitch=-10.0),  # Slight downward angle
            )

            camera = self.world.spawn_actor(
                camera_bp, camera_transform, attach_to=ego_vehicle
            )
            logger.info("Third-person camera spawned successfully")

            # Prepare image path
            image_path = f"{self.output_dir}/{self.args.map}_third_person_x{int(self.args.x)}_y{int(self.args.y)}_r{int(self.args.radius)}.png"

            # Remove existing file if it exists
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Removed existing file: {image_path}")

            # Use synchronous capture
            image_captured = False
            captured_image = None

            def save_image(image):
                nonlocal image_captured, captured_image
                captured_image = image
                image_captured = True
                logger.info("Third-person image data received from camera")

            # Start listening
            camera.listen(save_image)

            # Force world tick to ensure camera is active
            self.world.tick()
            time.sleep(1.0)  # Give camera time to initialize

            # Trigger another tick and wait for image
            self.world.tick()

            # Wait for image with shorter intervals
            timeout = 8.0
            elapsed = 0.0
            while not image_captured and elapsed < timeout:
                time.sleep(0.1)
                elapsed += 0.1
                # Tick world periodically to ensure sensor updates
                if elapsed % 1.0 < 0.1:
                    self.world.tick()

            # Stop listening immediately after capture
            camera.stop()

            if image_captured and captured_image is not None:
                # Save the image synchronously
                logger.info(f"Saving third-person image to: {image_path}")
                captured_image.save_to_disk(image_path)

                # Wait for file system to complete write
                time.sleep(1.0)

                # Verify the file was written correctly
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    logger.info(
                        f"Third-person image saved successfully: {file_size} bytes"
                    )

                    # Basic file size check (should be at least 50KB for a 1920x1080 image)
                    if file_size > 50000:
                        return True
                    else:
                        logger.error(
                            f"Third-person image file too small: {file_size} bytes"
                        )
                        return False
                else:
                    logger.error("Third-person image file was not created")
                    return False
            else:
                logger.error(
                    f"Failed to capture third-person image within {timeout} seconds"
                )
                return False

        except Exception as e:
            logger.error(f"Error capturing third-person view: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Ensure camera is always cleaned up
            if camera is not None:
                try:
                    camera.destroy()
                    logger.info("Third-person camera destroyed successfully")
                except Exception as e:
                    logger.warning(f"Error destroying third-person camera: {e}")

    def capture_birds_eye_with_materials(self, ego_vehicle):
        """Capture bird's eye view with real CARLA materials and ego vehicle"""
        logger.info(
            f"Capturing bird's eye view with materials centered at ({self.args.x}, {self.args.y}) with radius {self.args.radius}m"
        )

        # Capture real CARLA bird's eye view with materials
        carla_image_path = self._capture_carla_birds_eye_camera()

        if not carla_image_path:
            logger.error("Failed to capture CARLA bird's eye view with materials")
            return False

        logger.info(
            f"CARLA bird's eye view with materials saved to: {carla_image_path}"
        )
        return True

    def _capture_carla_birds_eye_camera(self):
        """Capture bird's eye view using CARLA camera with real materials"""
        camera = None
        try:
            # Set up bird's eye camera
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute(
                "image_size_x", "1920"
            )  # Reduced size for stability
            camera_bp.set_attribute("image_size_y", "1920")

            # Calculate camera height based on radius (higher for larger radius)
            camera_height = max(self.args.radius * 2, 200)  # At least 200m high

            # Calculate FOV to capture exactly the radius area
            # FOV = 2 * arctan(radius / camera_height) * (180 / pi)
            # Add some margin to ensure we capture the full radius
            margin_factor = 1.2  # 20% margin
            effective_radius = self.args.radius * margin_factor
            fov_radians = 2 * np.arctan(effective_radius / camera_height)
            fov_degrees = fov_radians * (180.0 / np.pi)

            # Clamp FOV to reasonable limits (CARLA supports 5-160 degrees)
            fov_degrees = max(5.0, min(160.0, fov_degrees))

            camera_bp.set_attribute("fov", str(fov_degrees))

            logger.info(
                f"Camera setup: height={camera_height}m, radius={self.args.radius}m, FOV={fov_degrees:.1f}°"
            )

            # Position camera directly above the target point
            camera_transform = carla.Transform(
                carla.Location(x=self.args.x, y=self.args.y, z=camera_height),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),  # Looking straight down
            )

            # Spawn camera
            camera = self.world.spawn_actor(camera_bp, camera_transform)
            logger.info("Camera spawned successfully")

            # Prepare image path
            carla_image_path = f"{self.output_dir}/{self.args.map}_carla_birds_eye_x{int(self.args.x)}_y{int(self.args.y)}_r{int(self.args.radius)}.png"

            # Remove existing file if it exists
            if os.path.exists(carla_image_path):
                os.remove(carla_image_path)
                logger.info(f"Removed existing file: {carla_image_path}")

            # Use synchronous capture with proper world tick synchronization
            image_captured = False
            captured_image = None

            def save_carla_image(image):
                nonlocal image_captured, captured_image
                captured_image = image
                image_captured = True
                logger.info("Image data received from camera")

            # Start listening
            camera.listen(save_carla_image)

            # Force world tick to ensure camera is active
            self.world.tick()
            time.sleep(1.0)  # Give camera time to initialize

            # Trigger another tick and wait for image
            self.world.tick()

            # Wait for image with shorter intervals
            timeout = 10.0
            elapsed = 0.0
            while not image_captured and elapsed < timeout:
                time.sleep(0.1)
                elapsed += 0.1
                # Tick world periodically to ensure sensor updates
                if elapsed % 1.0 < 0.1:
                    self.world.tick()

            # Stop listening immediately after capture
            camera.stop()

            if image_captured and captured_image is not None:
                # Save the image synchronously
                logger.info(f"Saving image to: {carla_image_path}")
                captured_image.save_to_disk(carla_image_path)

                # Wait for file system to complete write
                time.sleep(2.0)

                # Verify the file was written correctly
                if os.path.exists(carla_image_path):
                    file_size = os.path.getsize(carla_image_path)
                    logger.info(f"Image saved successfully: {file_size} bytes")

                    # Basic file size check (should be at least 100KB for a 1920x1920 image)
                    if file_size > 100000:
                        return carla_image_path
                    else:
                        logger.error(f"Image file too small: {file_size} bytes")
                        return None
                else:
                    logger.error("Image file was not created")
                    return None
            else:
                logger.error(f"Failed to capture image within {timeout} seconds")
                return None

        except Exception as e:
            logger.error(f"Error capturing CARLA bird's eye view: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Ensure camera is always cleaned up
            if camera is not None:
                try:
                    camera.destroy()
                    logger.info("Camera destroyed successfully")
                except Exception as e:
                    logger.warning(f"Error destroying camera: {e}")

    def _draw_ego_vehicle_on_plot(self, ax, ego_vehicle, x_min, x_max, y_min, y_max):
        """Draw only the ego vehicle on the plot"""
        # Draw ego vehicle (red)
        if ego_vehicle:
            ego_loc = ego_vehicle.get_location()
            ego_x, ego_y = ego_loc.x, -ego_loc.y

            if x_min <= ego_x <= x_max and y_min <= ego_y <= y_max:
                # Ego vehicle with special styling
                ax.scatter(
                    ego_x,
                    ego_y,
                    s=200,
                    c="red",
                    marker="s",
                    edgecolors="white",
                    linewidth=3,
                    alpha=0.9,
                    zorder=20,
                    label="Ego Vehicle",
                )

                # Add ego vehicle direction arrow
                ego_rotation = ego_vehicle.get_transform().rotation
                yaw_rad = np.radians(ego_rotation.yaw)
                arrow_length = 15
                dx = arrow_length * np.cos(yaw_rad)
                dy = arrow_length * np.sin(yaw_rad)

                ax.arrow(
                    ego_x,
                    ego_y,
                    dx,
                    -dy,
                    head_width=5,
                    head_length=3,
                    fc="red",
                    ec="white",
                    linewidth=2,
                    alpha=0.8,
                    zorder=21,
                )

    def _generate_map_visualization(self):
        """Generate map visualization using the existing script"""
        cmd = [
            "scripts/carla_python.sh",
            "scenario_runner/utils/visualize_map.py",
            "--map",
            self.args.map,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=os.getcwd()
            )
            if result.returncode == 0:
                logger.info("Map visualization generated successfully")
            else:
                logger.error(f"Failed to generate map visualization: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running map visualization command: {e}")

    def run(self):
        """Main execution function"""
        logger.info(
            f"Starting region visualization for {self.args.map} at ({self.args.x}, {self.args.y})"
        )

        ego_vehicle = None

        try:
            # Spawn only ego vehicle
            ego_vehicle = self._spawn_ego_vehicle()

            if not ego_vehicle:
                logger.error("Failed to spawn ego vehicle")
                return False

            # Capture third-person view
            if not self.capture_third_person_view(ego_vehicle, self.args.image_size):
                logger.error("Failed to capture third-person view")
                return False

            # Capture bird's eye view with materials
            if not self.capture_birds_eye_with_materials(ego_vehicle):
                logger.error("Failed to capture bird's eye view with materials")
                return False

            logger.info("Both visualizations completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during visualization: {e}")
            return False
        finally:
            # Clean up ego vehicle
            if ego_vehicle:
                ego_vehicle.destroy()
            self.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        metavar="H",
        default="localhost",
        help="IP of the host CARLA Simulator (default: localhost)",
    )
    parser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port of CARLA Simulator (default: 2000)",
    )
    parser.add_argument(
        "-m", "--map", required=True, help="Map name (e.g., Town01, Town02, etc.)"
    )
    parser.add_argument(
        "-x", "--x", type=float, required=True, help="X coordinate of the start point"
    )
    parser.add_argument(
        "-y", "--y", type=float, required=True, help="Y coordinate of the start point"
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=100.0,
        help="Radius for bird's eye view (default: 100.0 meters)",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="local_rss/visualize_map/around_spawn_point",
        help="Output directory for the visualizations (default: local_rss/visualize_map/around_spawn_point)",
    )
    parser.add_argument(
        "--image-size",
        type=tuple[int, int],
        default=(1920, 1080),
        help="Image size (width, height) or (x,y) coordinates (default: 1920, 1080)",
    )

    args = parser.parse_args()

    # Validate map name
    if not args.map.startswith("Town"):
        logger.warning(f"Map name '{args.map}' doesn't follow Town0x format")

    visualizer = RegionVisualizer(args)
    success = visualizer.run()

    if success:
        logger.info(
            "Both visualizations (third-person and bird's eye with materials) completed successfully!"
        )
        sys.exit(0)
    else:
        logger.error("Visualization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
