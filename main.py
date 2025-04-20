"""
Carla Gesture Control Client

This module connects to a CARLA server and enables vehicle control through facial 
gestures detected via a webcam. It leverages Mediapipe for facial landmark detection 
and a deep neural network (DNN) model for gesture classification. 
Vehicle movement and gear shifting are managed using rule-based logic.
"""


import os
import sys
import time
import random
import logging
import argparse
from typing import Tuple, List

import cv2
import torch
import pygame
import carla
import numpy as np
import pandas as pd
import mediapipe as mp
from torchvision import transforms

sys.path.append("src")

# Local imports
from src.mediapipe_facial_landmarks import MediapipeFacialLandmarks
from src.utils import load_model, load_dnn_model, data_preprocessing
from src.model.model import GestureClassifier
from src.carla_scripts.carla_control import GestureControl, World, HUD

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Gear Control Class (with Min and Max Tracking)
class GearControl:
    def __init__(self, sample_limit=500):
        self.sample_limit = sample_limit
        self.chin_samples = []
        self.mean_chin_y = None
        self.min_chin_y = None
        self.max_chin_y = None
        self.calibrated = False

    def update(self, chin_y: float) -> str:
        # Calibration phase
        if not self.calibrated:
            self.chin_samples.append(chin_y)
            if len(self.chin_samples) >= self.sample_limit:
                self.mean_chin_y = sum(self.chin_samples) / len(self.chin_samples)
                self.min_chin_y = min(self.chin_samples)
                self.max_chin_y = max(self.chin_samples)
                self.calibrated = True
            return "Calibrating..."

        # Compute thresholds
        reverse_threshold = self.min_chin_y + (self.mean_chin_y - self.min_chin_y) * 0.8
        forward_threshold = self.max_chin_y - (self.max_chin_y - self.mean_chin_y) * 0.8

        print(f"[DEBUG] ChinY: {chin_y:.2f}, \
              Reverse_Thres: {reverse_threshold:.2f}, \
                Forward_Thres: {forward_threshold:.2f}")

        if chin_y < reverse_threshold:
            return "Reverse"
        elif chin_y > forward_threshold:
            return "Forward"


def game_loop(args, model_path: str, labels: List[str]):
    pygame.init()
    pygame.font.init()

    world = None
    original_settings = None
    gear_controller = GearControl(sample_limit=180)
    model = load_dnn_model(PATH=model_path).to(device)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

        sim_world = client.get_world()
        traffic_manager = client.get_trafficmanager()

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: Asynchronous mode may cause traffic simulation issues.")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, traffic_manager, args)
        controller = GestureControl(world, args.autopilot)

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        facial_model = MediapipeFacialLandmarks(max_num_faces=1)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return

        pred_class_label = "none"
        done = False

        while not done:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            try:
                image_, landmarks_array, face_landmarks = (
                    facial_model.get_facial_patterns(image)
                )

                chin_y = face_landmarks.landmark[152].y
                nose_y = face_landmarks.landmark[1].y

                # Convert normalized y-values to pixel positions
                h, w, _ = image.shape
                chin_y_pixel = int(chin_y * h)

                # Update the gear control based on chin position
                gear_status = gear_controller.update(chin_y_pixel)

                if landmarks_array is not None:
                
                    (image_tensor, 
                     landmark_tensor) = data_preprocessing(image, landmarks_array)

                    images = image_tensor.to(device).float()
                    landmarks = landmark_tensor.to(device).float()

                    with torch.no_grad():
                        model.eval()
                        pred_class = model(images, landmarks)
                        pred_class_value = torch.argmax(pred_class, dim=1).item()
                        pred_class_label = labels[pred_class_value]

                cv2.putText(
                    image_, f'Class: {pred_class_label}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    image_, f'Gear: {gear_status}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )

            except Exception as e:
                print(f"Error processing image: {e}")
                continue

            cv2.imshow('MediaPipe Face Mesh', image_)
            if cv2.waitKey(5) & 0xFF == 27:
                break

            if args.sync:
                sim_world.tick()

            clock.tick_busy_loop(60)
            controller._parse_vehicle_keys(world, pred_class_label, gear_status)
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if original_settings:
            sim_world.apply_settings(original_settings)

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


def main():
    """
    Main entry point for the CARLA Gesture Control client.
    Parses arguments, sets up logging, and runs the main game loop.
    """
    parser = argparse.ArgumentParser(description='CARLA Manual Control Client '
                                                 'with Gesture Input')

    parser.add_argument('--host', default='127.0.0.1', help='CARLA host IP')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA port')
    parser.add_argument('-a', '--autopilot', action='store_true', help='Enable autopilot')
    parser.add_argument('--res', default='1280x720', 
                        help='Screen resolution (WIDTHxHEIGHT)')
    parser.add_argument('--filter', default='vehicle.*', help='Actor filter')
    parser.add_argument('--generation', default='All', help='Actor generation filter')
    parser.add_argument('--rolename', default='hero', help='Actor role name')
    parser.add_argument('--gamma', default=1.0, type=float, help='Gamma correction')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('-v', '--verbose', action='store_true', dest='debug', 
                        help='Enable debug logging')
    
    parser.add_argument('--model_path', 
                        default= (
                    r"models\updated_weights\model_20250420_153811_13"
                        ),
                        help='Path to the DNN model weights')
    parser.add_argument('--labels', 
                        default= [
                    'left_tilt', 'right_tilt', 'front_facing', 'frown', 'mouth_open'
                    ], 
                        help='List of gesture labels')

    args = parser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('Connecting to CARLA server at %s:%s', args.host, args.port)

    try:
        game_loop(args, model_path=args.model_path, labels=args.labels)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
