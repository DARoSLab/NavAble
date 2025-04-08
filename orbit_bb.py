import setup_path 
import json
import cv2
import airsim
import os
import numpy as np
import sys
import math
import time
import argparse
import utils

class Position:
    def __init__(self, pos):
        self.x = pos.x_val
        self.y = pos.y_val
        self.z = pos.z_val

class OrbitNavigator:
    def __init__(self, radius = 2, altitude = 0, speed = 2, iterations = 1, center = [1,0], snapshots = 100, environment_name = "Test", run_number = 1):
        self.radius = radius
        self.altitude = altitude
        self.speed = speed
        self.iterations = iterations
        self.snapshots = snapshots
        self.snapshot_delta = None
        self.next_snapshot = None
        self.z = None
        self.snapshot_index = 0
        self.takeoff = False 
        self.environment_name = environment_name
        self.run_number = run_number

        if self.snapshots is not None and self.snapshots > 0:
            self.snapshot_delta = 360 / self.snapshots

        if self.iterations <= 0:
            self.iterations = 1

        if len(center) != 2:
            raise Exception("Expecting '[x,y]' for the center direction vector")
        
        cx = float(center[0])
        cy = float(center[1])
        length = math.sqrt((cx*cx) + (cy*cy))
        if length == 0:
            cx, cy = 0, 0  # Keep center fixed at spawn point
        else:
            cx /= length
            cy /= length
        cx *= self.radius
        cy *= self.radius


        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.home = self.client.getMultirotorState().kinematics_estimated.position
        start = time.time()
        count = 0
        while count < 100:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            if abs(pos.z_val - self.home.z_val) > 1:                                 
                count = 0
                self.home = pos
                if time.time() - start > 10:
                    print("Drone position is drifting, we are waiting for it to settle down...")
                    start = time
            else:
                count += 1

        self.center = pos
        self.center.x_val += cx
        self.center.y_val += cy

        all_objects = self.client.simListSceneObjects()
        output_file = "scene_objects.txt"

        
        i = 4
        for obj_name in all_objects:
            if obj_name == "merged_blocks57":
                self.client.simSetSegmentationObjectID("merged_blocks57", 1, False)
                print("block 57 set")
            elif obj_name == "B_TrafficLight32":
                self.client.simSetSegmentationObjectID("B_TrafficLight32", 2, False)
                print("block traffic set")
            elif obj_name == "merged_blocks50":
                self.client.simSetSegmentationObjectID("merged_blocks50", 3, False)
                print("block 50 set")
            else:
                self.client.simSetSegmentationObjectID(obj_name, i, False)
            i+=1
            if (i==256):
                i = 4
    

        self.color_to_object_map = {
        (153, 108, 6): "tactile block",
        (112, 105, 191): "pedestrian sign",
        (89, 121, 72): "tactile block",
        }
    
    

    def start(self):
        print("arming the drone...")
        self.client.armDisarm(True)
        
        start = self.client.getMultirotorState().kinematics_estimated.position
        landed = self.client.getMultirotorState().landed_state
        if not self.takeoff and landed == airsim.LandedState.Landed: 
            self.takeoff = True
            print("taking off...")
            self.client.takeoffAsync().join()
            start = self.client.getMultirotorState().kinematics_estimated.position
            z = -self.altitude 
        else:
            print("already flying so we will orbit at current altitude {}".format(start.z_val))
            z = start.z_val

        print("climbing to position: {},{},{}".format(start.x_val, start.y_val, z))
        self.client.moveToPositionAsync(start.x_val, start.y_val, z, self.speed).join()
        self.z = z
        
        print("ramping up to speed...")
        count = 0
        self.start_angle = None
        self.next_snapshot = None
        
        ramptime = self.radius / 10
        self.start_time = time.time()        

        while count < self.iterations:
            if self.snapshots > 0 and not (self.snapshot_index < self.snapshots):
                break
            now = time.time()
            speed = self.speed
            diff = now - self.start_time
            if diff < ramptime:
                speed = self.speed * diff / ramptime
            elif ramptime > 0:
                print("reached full speed...")
                ramptime = 0
                
            lookahead_angle = speed / self.radius

            pos = self.client.getMultirotorState().kinematics_estimated.position
            dx = pos.x_val - self.center.x_val
            dy = pos.y_val - self.center.y_val
            actual_radius = math.sqrt((dx*dx) + (dy*dy))
            angle_to_center = math.atan2(dy, dx)

            camera_heading = (angle_to_center - math.pi) * 180 / math.pi 

            lookahead_x = self.center.x_val + self.radius * math.cos(angle_to_center + lookahead_angle)
            lookahead_y = self.center.y_val + self.radius * math.sin(angle_to_center + lookahead_angle)

            vx = lookahead_x - pos.x_val
            vy = lookahead_y - pos.y_val

            if self.track_orbits(angle_to_center * 180 / math.pi):
                count += 1
                print("completed {} orbits".format(count))
            
            self.camera_heading = camera_heading
            self.client.moveByVelocityZAsync(vx, vy, z, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, camera_heading))

        self.client.moveToPositionAsync(start.x_val, start.y_val, z, 2).join()

        if self.takeoff:            
            if z < self.home.z_val:
                print("descending")
                #self.client.moveToPositionAsync(start.x_val, start.y_val, self.home.z_val - 5, 2).join()

            print("landing...")
            #self.client.landAsync().join()

            print("disarming.")
            #self.client.armDisarm(False)


    def track_orbits(self, angle):
        if angle < 0:
            angle += 360

        if self.start_angle is None:
            self.start_angle = angle
            if self.snapshot_delta:
                self.next_snapshot = angle + self.snapshot_delta
            self.previous_angle = angle
            self.shifted = False
            self.previous_sign = None
            self.previous_diff = None            
            self.quarter = False
            return False

        if self.previous_angle is None:
            self.previous_angle = angle
            return False            

        if self.previous_angle > 350 and angle < 10:
            if self.snapshot_delta and self.next_snapshot >= 360:
                self.next_snapshot -= 360
            return False

        diff = self.previous_angle - angle
        crossing = False
        self.previous_angle = angle

        if self.snapshot_delta and angle > self.next_snapshot:            
            print("Taking snapshot {}".format(self.snapshot_index))
            self.take_snapshot()
            self.next_snapshot += self.snapshot_delta

        diff = abs(angle - self.start_angle)
        if diff > 45:
            self.quarter = True

        if self.quarter and self.previous_diff is not None and diff != self.previous_diff:
            direction = self.sign(self.previous_diff - diff)
            if self.previous_sign is None:
                self.previous_sign = direction
            elif self.previous_sign > 0 and direction < 0:
                if diff < 45:
                    self.quarter = False
                    if self.snapshots <= self.snapshot_index + 1:
                        crossing = True
            self.previous_sign = direction
        self.previous_diff = diff

        return crossing

    def take_snapshot(self):
        run_dir, environment_number = self.create_directories(environment_name, run_number)
        # First hold our current position so drone doesn't try and keep flying while we take the picture.
        pos = self.client.getMultirotorState().kinematics_estimated.position
        self.client.moveToPositionAsync(pos.x_val, pos.y_val, self.z, 0.5, 10, airsim.DrivetrainType.MaxDegreeOfFreedom, 
            airsim.YawMode(False, self.camera_heading)).join()  # Wait for the drone to stop

        camera_name = "0"  # Camera identifier
        image_type_rgb = airsim.ImageType.Scene
        image_type_seg = airsim.ImageType.Segmentation
        image_type_depth = airsim.ImageType.DepthPlanar
    
        

        # color_to_object_map = {
        #     (153, 108, 6): "tactile block",
        #     (112, 105, 191): "tactile block",
        #     (89, 121, 72): "tactile block",
        #     (190, 225, 64): "tactile block",
        #     (206, 190, 59): "tactile block",
        #     (81, 13, 36): "tactile block",
        #     (115, 176, 195): "tactile block",
        #     (161, 171, 27): "tactile block",
        #     (135, 169, 180): "tactile block",
        #     (29, 26, 199): "tactile block",
        #     (102, 16, 239): "tactile block",
        #     (242, 107, 146): "tactile block",
        #     (156, 198, 23): "tactile block",
        #     (49, 89, 160): "tactile block",
        #     (68, 218, 116): "tactile block",
        #     (11, 236, 9): "tactile block",
        #     (196, 30, 8): "tactile block",
        #     (121, 67, 28): "tactile block",
        #     (0, 53, 65): "tactile block",
        #     (146, 52, 70): "tactile block",
        #     (226, 149, 143): "tactile block",
        #     (151, 126, 171): "tactile block",
        #     (194, 39, 7): "tactile block",
        #     (205, 120, 161): "tactile block",
        #     (212, 51, 60): "tactile block",
        #     (211, 80, 208): "tactile block",
        #     (189, 135, 188): "tactile block",
        #     (54, 72, 205): "tactile block",
        #     (103, 252, 157): "tactile block",
        #     (124, 21, 123): "tactile block",
        #     (19, 132, 69): "tactile block",
        #     (195, 237, 132): "tactile block",
        #     (94, 253, 175): "tactile block",
        #     (182, 251, 87): "tactile block",
        #     (90, 162, 242): "tactile block",
        #     (199, 29, 1): "tactile block",
        #     (254, 12, 229): "tactile block",
        #     (35, 196, 244): "tactile block",
        #     (220, 163, 49): "tactile block",
        #     (86, 254, 214): "tactile block",
        #     (152, 3, 129): "tactile block",
        #     (92, 31, 106): "tactile block"
        # }


        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, image_type_rgb),
            airsim.ImageRequest(camera_name, image_type_seg, False, False),
            airsim.ImageRequest(camera_name, image_type_depth, True, False),
        ])

        bottom_responses = self.client.simGetImages([
            airsim.ImageRequest("3", image_type_rgb), 
            airsim.ImageRequest("3", image_type_seg, False, False),
            airsim.ImageRequest("3", image_type_depth, True, False),
        ])


        image_number = self.snapshot_index

        rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb_image = cv2.imdecode(rgb_image, cv2.IMREAD_COLOR)  

        seg_image = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        seg_image = seg_image.reshape(responses[1].height, responses[1].width, 3)

        depth_img = np.array(responses[2].image_data_float, dtype=np.float32)
        depth_array = depth_img.reshape(responses[2].height, responses[2].width)
        utils.write_pfm("depth_image.pfm", depth_array)
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255
        depth_img = depth_normalized.astype(np.uint8)
    

        rgb_td_image = np.frombuffer(bottom_responses[0].image_data_uint8, dtype=np.uint8)
        rgb_td_image = cv2.imdecode(rgb_td_image, cv2.IMREAD_COLOR)

        seg_td_image = np.fromstring(bottom_responses[1].image_data_uint8, dtype=np.uint8)
        seg_td_image = seg_td_image.reshape(bottom_responses[1].height, bottom_responses[1].width, 3)

        depth_td_img = np.array(bottom_responses[2].image_data_float, dtype=np.float32)
        depth_td_array = depth_td_img.reshape(bottom_responses[2].height, bottom_responses[2].width)
        utils.write_pfm("depth_td_image.pfm", depth_td_array)
        depth_td_normalized = (depth_td_array - depth_td_array.min()) / (depth_td_array.max() - depth_td_array.min()) * 255
        depth_td_img = depth_td_normalized.astype(np.uint8)


        filename_prefix = f"{environment_name}_{environment_number}_{run_number}_{image_number}"      
        cv2.imwrite(os.path.join(run_dir, 'rgb', f"{filename_prefix}_FV_rgb.png"), rgb_image)
        cv2.imwrite(os.path.join(run_dir, 'seg', f"{filename_prefix}_FV_segmentation.png"), seg_image)
        cv2.imwrite(os.path.join(run_dir, 'depth', f"{filename_prefix}_FV_depth.png"), depth_img)
        cv2.imwrite(os.path.join(run_dir, 'rgb', f"{filename_prefix}_TD_rgb.png"), rgb_td_image)
        cv2.imwrite(os.path.join(run_dir, 'seg', f"{filename_prefix}_TD_segmentation.png"), seg_td_image)
        cv2.imwrite(os.path.join(run_dir, 'depth', f"{filename_prefix}_TD_depth.png"), depth_td_img)
    

        bounding_boxes = self.get_bounding_boxes(seg_image, rgb_image, self.color_to_object_map)
        td_bounding_boxes = self.get_bounding_boxes(seg_td_image, rgb_td_image, self.color_to_object_map)

        for bbox in bounding_boxes:
            x_min = bbox['bounding_box']['x_min']
            y_min = bbox['bounding_box']['y_min']
            x_max = bbox['bounding_box']['x_max']
            y_max = bbox['bounding_box']['y_max']
            cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            label = f"{bbox['object_name']}"  
            cv2.putText(rgb_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        for bbox in td_bounding_boxes:
            x_min = bbox['bounding_box']['x_min']
            y_min = bbox['bounding_box']['y_min']
            x_max = bbox['bounding_box']['x_max']
            y_max = bbox['bounding_box']['y_max']

            cv2.rectangle(rgb_td_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            label = f"{bbox['object_name']}"  
            cv2.putText(rgb_td_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imwrite(os.path.join(run_dir, 'rgb_bb', f"{filename_prefix}_FV_rgb_bb.png"), rgb_image)
        cv2.imwrite(os.path.join(run_dir, 'rgb_bb', f"{filename_prefix}_TD_rgb_bb.png"), rgb_td_image)

        with open(os.path.join(run_dir, 'label', f"{filename_prefix}_FV.json"), 'w') as f:
            json.dump(bounding_boxes, f)
        
        with open(os.path.join(run_dir, 'label', f"{filename_prefix}_TD.json"), 'w') as f:
            json.dump(td_bounding_boxes, f)
        self.snapshot_index += 1
        self.start_time = time.time() 

    def get_bounding_boxes(self, segmentation_image, rgb_image, color_to_object_map):

        segmentation_image_resized = cv2.resize(segmentation_image, (rgb_image.shape[1], rgb_image.shape[0]))

        segmentation_image_resized = cv2.cvtColor(segmentation_image_resized, cv2.COLOR_BGR2RGB)

        bounding_boxes = []

        for target_segmentation_color, object_name in color_to_object_map.items():

            mask = np.all(segmentation_image_resized == np.array(target_segmentation_color), axis=-1)


            non_zero_coords = np.argwhere(mask)


            if non_zero_coords.size > 0:
                y_min, x_min = np.min(non_zero_coords, axis=0)
                y_max, x_max = np.max(non_zero_coords, axis=0)


                bounding_boxes.append({
                    "object_name": object_name, 
                    "object_id": target_segmentation_color, 
                    "bounding_box": {
                        "x_min": int(x_min),
                        "y_min": int(y_min),
                        "x_max": int(x_max),
                        "y_max": int(y_max)
                    }
                })

        return bounding_boxes


    def sign(self, s):
        if s < 0: 
            return -1
        return 1

    def get_environment_number(self, environment_name):
        try:
            with open('environment_mapping.txt', 'r') as file:
                mapping = file.readlines()
            mapping_dict = {line.split(",")[0].strip(): line.split(",")[1].strip() for line in mapping}
        except FileNotFoundError:
            mapping_dict = {}

        if environment_name not in mapping_dict:

            new_num = f"{len(mapping_dict):02d}"
            with open('environment_mapping.txt', 'a') as file:
                file.write(f"{environment_name},{new_num}\n")
            mapping_dict[environment_name] = new_num

        return mapping_dict[environment_name]
    
    def create_directories(self, environment_name, run_number):

        environment_number = self.get_environment_number(environment_name)
        base_dir = f"{environment_name}_{environment_number}"
        run_dir = os.path.join(base_dir, f"Run_{run_number}")

        os.makedirs(os.path.join(run_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'rgb_bb'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'seg'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'label'), exist_ok=True)

        return run_dir, environment_number
if __name__ == "__main__":
    environment_name = input("Enter the environment name: ")
    run_number = input("Enter the run number: ")
    try:
        run_number = int(run_number)
    except ValueError:
        print("Invalid run number. Please enter a valid integer.")
        sys.exit(1)
    args = sys.argv
    args.pop(0)
    arg_parser = argparse.ArgumentParser("Orbit.py makes drone fly in a circle with camera pointed at the given center vector")
    arg_parser.add_argument("--radius", type=float, help="radius of the orbit", default=2)
    arg_parser.add_argument("--altitude", type=float, help="altitude of orbit (in positive meters)", default=0.3)
    arg_parser.add_argument("--speed", type=float, help="speed of orbit (in meters/second)", default=3)
    arg_parser.add_argument("--center", help="x,y direction vector pointing to center of orbit from current starting position (default 0,0)", default="0,0")
    arg_parser.add_argument("--iterations", type=float, help="number of 360 degree orbits (default 3)", default=3)
    arg_parser.add_argument("--snapshots", type=float, help="number of FPV snapshots to take during orbit (default 30)", default=30)    
    arg_parser.add_argument("--environment",type=str, help="name of the environment", default="Test")
    arg_parser.add_argument("--run", type=int, help="run number", default=1)
    args = arg_parser.parse_args(args)    
    nav = OrbitNavigator(args.radius, args.altitude, args.speed, args.iterations, args.center.split(','), args.snapshots, environment_name, run_number)
    nav.start()