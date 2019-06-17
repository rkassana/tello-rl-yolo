from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import time
import os
import json
from utils.utils import get_yolo_boxes
from utils.bbox import draw_boxes
from keras.models import load_model
import logging
import threading
import queue
from agents.rl_drone import RLAgent
from agents.drone_sim_env import drone_sim

LOG_FILENAME = 'output.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)



# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 25

config_path = './zoo/config_voc.json'
with open(config_path) as config_buffer:
    config = json.load(config_buffer)


###############################
#   Set some parameter
###############################
net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
obj_thresh, nms_thresh = 0.90, 0.45

# Create queue for thread to find box in frame
box_q = queue.Queue()
frame_q = queue.Queue()
frame_glob = []
box_glob = []




def box_thread():
    """."""
    ###############################
    #   Load the model for YOLO detection
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Load the model for RL agent
    ###############################
    ENV_NAME = 'drone'
    env = drone_sim()
    agent = RLAgent(env)
    agent.agent.load_weights('ddpg_{}_weights.h5f'.format(ENV_NAME))

    while True:
        try:
            print('thread size: ' + str(len(frame_glob)))
            boxes = \
                get_yolo_boxes(infer_model, [frame_glob],
                               net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
            print('ok')
            if len(boxes) > 0:
                filter_boxes = []
                actions_f = []
                for box in boxes:
                    if box.get_label() == 14:
                        # logging.debug('box located xmin : %d, xmax : %d, ymin: %d, ymax: %d' %
                        # (box.xmin, box.xmax, box.ymin, box.ymax))
                        box_x = int((box.xmax - box.xmin) / 2) + box.xmin
                        box_y = int((box.ymax - box.ymin) / 2) + box.ymin
                        actions = agent.agent.forward([box_x, box_y])
                        actions_f.append(actions)
                        filter_boxes.append(box)
                box_q.put(([filter_boxes[0]], [actions_f[0]]))
                print('thread : box, cmd sent')
        except Exception as e:
            print(e)


thread_box = threading.Thread(target=box_thread, args=())
thread_box.setDaemon(True)
thread_box.start()



class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 60

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

        # Run thread to find box in frame
        print('init done')



    def run(self):

        global frame_glob
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 25.0, (960, 720))
        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False

        print('loop started')

        last_yaw=0
        box_cnt = 0

        while not should_stop:

            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)
            if frame_read.stopped:
                frame_read.stop()
                break

            self.screen.fill([0, 0, 0])
            frame = frame_read.frame
            frame_glob = frame
            box_list = []
            try:
                box_list, actions_list = box_q.get_nowait()
                box = box_list[0]
                actions = actions_list[0]
            except Exception:
                pass
            box_x = 0
            box_y = 0

            if len(box_list) > 0:
                box_cnt = 0
                area = (box.xmax - box.xmin) * (box.ymax - box.ymin)
                area_p = (area / 691200.) * 100.0
                draw_boxes(frame, box_list, config['model']['labels'], obj_thresh)
                self.yaw_velocity = -int(actions[0])
                last_yaw = self.yaw_velocity
                self.up_down_velocity = int(actions[1])
                if area_p < 30:
                    self.for_back_velocity = 60
                elif area_p > 50:
                    self.for_back_velocity = -60
                else:
                    self.for_back_velocity = 0
                frame = cv2.circle(frame, (box_x, box_y), 5, (255, 0, 0), -1)

            else:
                box_cnt += 1
                if box_cnt > 25:
                    self.yaw_velocity = last_yaw
                else:
                    self.yaw_velocity = 0
                    self.up_down_velocity = 0
                    self.for_back_velocity = 0

            frame = cv2.circle(frame, (480, 360), 5,(0, 0, 255),-1)
            out.write(frame)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()
        out.release()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def reset_speed(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(0, 0, 0, 0)


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
