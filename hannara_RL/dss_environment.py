import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import random

PhotoImage = ImageTk.PhotoImage
UNIT = 60  # 픽셀 수
HEIGHT = 14  # 그리드 세로
WIDTH = 25  # 그리드 가로
maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
rec_lo = [21, 12]
rand_lo=[[4, 12],
    [7, 11],
    [1, 8],
    [18, 7],
    [12, 7],
    [10, 10],
    [19, 10]]
fire_lo = random.choice(rand_lo)
# fire_lo = [4, 12]
liftboat_lo = [3, 7]
np.random.seed(1)


class Env(tk.Tk):
    def __init__(self, render_speed=10000):
        super(Env, self).__init__()
        self.render_speed = render_speed
        self.action_space = ['s', 'u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('DeepSARSA')
        self.geometry('{0}x{1}'.format(WIDTH * UNIT, HEIGHT * UNIT + 50))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        # 장애물 설정
        # self.set_reward([fire_lo[0], fire_lo[1]], -2)
        # self.set_reward([fire_lo[0], fire_lo[1]], -1)
        # self.set_reward([fire_lo[0], fire_lo[1]], -1)
        # self.set_reward([1, 2], -1)
        # self.set_reward([2, 3], -1)
        # 목표 지점 설정
        self.set_reward([liftboat_lo[0], liftboat_lo[1]], 5)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        canvas.create_image(WIDTH * UNIT / 2, HEIGHT * UNIT / 2, image=self.shapes[5])
        # 그리드 생성
        for col in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = col, 0, col, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, row, WIDTH * UNIT, row
            canvas.create_line(x0, y0, x1, y1)

        for y in range(HEIGHT):
            for x in range(WIDTH):
                if maze[y][x] == 1:
                    canvas.create_image(UNIT * x + (UNIT * 0.5), UNIT * y + (UNIT * 0.5), image=self.shapes[6])

        self.rewards = []
        self.goal = []
        # 캔버스에 이미지 추가
        # x, y = UNIT / 2, UNIT / 2
        self.rectangle = canvas.create_image(UNIT * rec_lo[0] + (UNIT * 0.5), UNIT * rec_lo[1] + (UNIT * 0.5),
                                             image=self.shapes[0])

        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(Image.open("../img/rectangle.png").resize((UNIT - 10, UNIT - 10)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((UNIT - 10, UNIT - 10)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((UNIT - 10, UNIT - 10)))
        block = PhotoImage(Image.open("../img/block.png").resize((UNIT, UNIT)))
        FIRE = PhotoImage(Image.open("../img/FIRE.png").resize((UNIT - 5, UNIT - 5)))
        ship = PhotoImage(Image.open("../img/hannara_skele.png").resize((UNIT * WIDTH, UNIT * HEIGHT)))
        lifeboat = PhotoImage(Image.open("../img/lifeboat.png").resize((UNIT - 5, UNIT - 5)))
        return rectangle, triangle, circle, FIRE, lifeboat, ship, block

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()

        fire_lo = random.choice(rand_lo)
        self.set_reward([fire_lo[0], fire_lo[1]], -1)
        a = random.randint(0, 2)
        if a == 0:
            fire_lo[1] -= 1
        elif a == 1:
            fire_lo[0] -= 1
        elif a == 2:
            fire_lo[0] += 1
        self.set_reward([fire_lo[0], fire_lo[1]], -2)
        b = random.randint(0, 2)
        if b == 0:
            fire_lo[1] -= 1
        elif b == 1:
            fire_lo[0] -= 1
        elif b == 2:
            fire_lo[0] += 1
        self.set_reward([fire_lo[0], fire_lo[1]], -3)
        # #goal
        self.set_reward([liftboat_lo[0], liftboat_lo[1]], 5)

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])

        temp = {}

        if reward > 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[4])

            self.goal.append(temp['figure'])
        # 보상이 1이면 출력 4개

        elif reward == -1:  # -1
            # temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])
        elif reward == -2:  # -2
            # temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])

        elif reward == -3:  # -3
            # temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

    # goal에 도착했는지 확인
    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']
                if reward['reward'] == 5:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        # 이동한 만큼 다시 원점으로 되돌리기 0,0으로

        self.canvas.move(self.rectangle, rec_lo[0] * UNIT + UNIT / 2 - x, rec_lo[1] * UNIT + UNIT / 2 - y)
        self.reset_reward()
        return self.get_state()

    # not done (미도착) 이면 step 할때마다 다음 위치, 보상, 도착여부(done) 확인
    def step(self, action):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action)
        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_goal']
        reward = check['rewards']

        self.canvas.tag_raise(self.rectangle)

        s_ = self.get_state()

        return s_, reward, done

    def get_state(self):

        location = self.coords_to_state(self.canvas.coords(self.rectangle))
        agent_x = location[0]
        agent_y = location[1]

        states = list()

        for reward in self.rewards:  # 총4개 (장애물 3개 * 4, 목표 1개 * 3 = 15개)
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y)
            if reward['reward'] < 0:
                states.append(-1)
                # states.append(reward['direction'])
            else:
                states.append(1)

        return states

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 1 or temp['reward'] == -2:
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)  # 불 움직임
            temp['state'] = self.coords_to_state(temp['coords'])

            new_rewards.append(temp)
        return new_rewards

    # 불 움직이기, FIGURE는
    def move_const(self, target):
        self.canvas.coords(target['figure'])  # 움직이는 불
        # s = self.canvas.coords(target['figure'])  # 움직이는 불
        # base_action = np.array([0, 0])
        # ss = ((int(s[0]) - UNIT / 2) / UNIT, (int(s[1]) - UNIT / 2) / UNIT)
        #
        # # self.canvas.create_image((UNIT * target['state'][0]) + UNIT / 2,
        # #                                            (UNIT * target['state'][1]) + UNIT / 2,
        # #                                            image=self.shapes[4])
        # # if target['direction'] == -1 :
        # if maze[int(ss[1])][int(ss[0]) + 1] == 1 or maze[int(ss[1])][int(ss[0]) - 1] == 1:  # 좌 혹은 우가 막히면 위로
        #     target['direction'] = 0  # 방향 위로 바꾸기
        #
        # if maze[int(ss[1]) - 1][int(ss[0])] == 1:  # 위가 막혔으때,
        #     a = random.choice([-1, 1])
        #     target['direction'] = a
        #     if maze[int(ss[1])][int(ss[0]) - 1] == 1:  # 위+좌 막혔을때,
        #         target['direction'] = 1  # 우로
        #     elif maze[int(ss[1])][int(ss[0]) + 1] == 1:  # 위+우 막혔을때,
        #         target['direction'] = -1  # 좌로
        #     elif maze[int(ss[1])][int(ss[0]) + 1] == 1 and maze[int(ss[1])][int(ss[0]) - 1] == 1:
        #         base_action = np.array([0, 0])
        #
        # if target['direction'] == 1:  # 우측으로
        #     base_action[0] += UNIT
        # elif target['direction'] == -1:  # 좌측으로
        #     base_action[0] -= UNIT
        # elif target['direction'] == 0:  # 위로
        #     base_action[1] -= UNIT

        #
        # if (target['figure'] is not self.rectangle
        #         and s == [(liftboat_lo[0]) * UNIT, (liftboat_lo[1]) * UNIT]):
        #     base_action = np.array([0, 0])
        #
        # # if target['direction'] == 0:
        # #     self.set_reward([ss[0], ss[1] - 1], -1)
        #
        # self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def find_rectangle(self):
        temp = self.canvas.coords(self.rectangle)
        # x = (temp[0] / 100) - 0.5
        # y = (temp[1] / 100) - 0.5
        x = ((temp[0] - UNIT / 2) / UNIT)
        y = ((temp[1] - UNIT / 2) / UNIT)

        return int(x), int(y)

    def move(self, target, action):  # target=rectangle
        s = self.canvas.coords(target)
        base_action = np.array([0, 0])
        location = self.find_rectangle()

        if action == 1 and location[1] > 0 and maze[location[1] - 1][location[0]] == 0:  # 상 location[0], y 좌표가 0보다 크고
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 2 and location[1] < HEIGHT - 1 and maze[location[1] + 1][location[0]] == 0:  # 하
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 3 and location[0] > 0 and maze[location[1]][location[0] - 1] == 0:  # 좌
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] -= UNIT
        elif action == 4 and location[0] < WIDTH - 1 and maze[location[1]][location[0] + 1] == 0:  # 우
            if s[0] > UNIT:
                base_action[0] += UNIT

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
